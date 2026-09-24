#!/usr/bin/env bash
# ============================================================================
# wedge-watchdog-eslane.sh — side-lane (es-lane) wedge watchdog
# Adapted from wedge-watchdog.sh (production, md5 3b7f1384). DIFF DISCIPLINE:
# every changed line is one of — container name, port, served model id, log
# source (docker logs instead of repo .run/server.log), restart command
# (start.sh --launch kept as the interface), boot-grace comment, or a comment.
# Nothing else differs. The Xe2 Level-Zero wedge signatures are UNCHANGED
# (same silicon, same Level Zero stack).
#
# The Xe2 Level-Zero wedge kills the serving job every 2-6 h under load
# (kernel signature: "Engine reset: engine_class=ccs|bcs",
#  "Fault response: Unsuccessful", "guc_exec_queue_timedout_job"). Only a
# container restart recovers; in-flight requests are lost.
#
# Differences from the production watchdog (all intentional):
#   - container es-lane, port 8022, served id qwen-256k
#   - engine log source is `docker logs es-lane` (the side lane has no host
#     log-tail follower); log_size/log_tail read the container log directly
#     (bounded: last 2000 lines / last 200 lines)
#   - boot grace: es-lane boots in ~4-5 min (vs ~8-9 production). The same
#     cold-load guard applies — no stall counting until /v1/models answers
#     once — so no separate grace constant is needed.
#   - restart command: ESLANE_RESTART_CMD (default: "$SCRIPT_DIR/start.sh"
#     --launch). ON-BOX DEPLOY NOTE: the deploy directory carries its own
#     copy (acceptance-v2 pattern). Restore scripts MUST use the deploy-dir
#     watchdog path, NOT /home/bonz/wedge-watchdog.sh — that path does not
#     exist and a restore launched it silently today (ledger-corrected).
#
# Liveness = GEN-PROBE. A 1-token chat completion proves the executor
# actually advances (a wedged engine can hold /v1/models up). On wedge
# detection it: captures last log lines + py-spy python-frame dumps of TP
# workers + EngineCore BEFORE any restart (capture-first discipline), kills
# the container group, restarts via the restart command (bounded retries,
# default 3), gives up LOUDLY after WEDGE_WATCHDOG_RETRIES.
#
# DISABLE (double opt-out, non-negotiable, same as production):
# WEDGE_WATCHDOG_DISABLE=1 AND PREFLIGHT_SKIPPED=1 together.
#
# State lives under .run/ in this repo only (watchdog-eslane.log,
# wedge-<ts>.log).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; }
red()  { echo -e "\033[1;31m$*\033[0m"; }

WEDGE_WATCHDOG_DISABLE="${WEDGE_WATCHDOG_DISABLE:-0}"
PREFLIGHT_SKIPPED="${PREFLIGHT_SKIPPED:-0}"
if [[ "$WEDGE_WATCHDOG_DISABLE" == "1" && "$PREFLIGHT_SKIPPED" == "1" ]]; then
    red ""
    red "  =================================================================="
    red "   WEDGE WATCHDOG NOT RUNNING (WEDGE_WATCHDOG_DISABLE=1 + --no-preflight)"
    red "  =================================================================="
    red "   The rig WILL wedge unattended within 2-6 h under load. Xe2 Level-Zero"
    red "   wedge signature: 'Engine reset: engine_class=ccs|bcs', 'Fault response:"
    red "   Unsuccessful', 'guc_exec_queue_timedout_job'. Only a container"
    red "   restart recovers; in-flight requests are lost. A human must watch it."
    red "  =================================================================="
    exit 0
fi
if [[ "$WEDGE_WATCHDOG_DISABLE" == "1" && "$PREFLIGHT_SKIPPED" != "1" ]]; then
    warn "WEDGE_WATCHDOG_DISABLE=1 without --no-preflight is not a valid opt-out — supervising anyway."
fi

INTERVAL="${WEDGE_WATCHDOG_INTERVAL:-60}"
RETRIES="${WEDGE_WATCHDOG_RETRIES:-3}"
CONTAINER_NAME="${CONTAINER_NAME:-es-lane}"
PORT="${PORT:-8022}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-256k}"
PREFLIGHT_XPU_COUNT="${PREFLIGHT_XPU_COUNT:-4}"
STALL_LIMIT=3
RESTART_CMD="${ESLANE_RESTART_CMD:-$SCRIPT_DIR/../start.sh --launch}"

RUN_DIR="$SCRIPT_DIR/.run/eslane"
WATCHDOG_LOG="$RUN_DIR/watchdog-eslane.log"
mkdir -p "$RUN_DIR"

device_count() {
    if command -v sycl-ls >/dev/null 2>&1; then
        local c
        c=$(sycl-ls 2>/dev/null | grep -c 'level_zero:gpu' || true)
        [[ -n "$c" && "$c" -gt 0 ]] && { echo "$c"; return; }
    fi
    local d
    d=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
    [[ "$d" -gt 0 ]] && { echo "$d"; return; }
    if command -v xpu-smi >/dev/null 2>&1; then
        if xpu-smi discovery >/dev/null 2>&1; then
            echo 4
            return
        fi
    fi
    echo 0
}

container_running() {
    docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER_NAME"
}

health_ok() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsS -m 8 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_MODEL_NAME"
    else
        python3 -c 'import urllib.request,sys;d=urllib.request.urlopen("http://localhost:'"$PORT"'/v1/models",timeout=8).read().decode()' 2>/dev/null | grep -q "$SERVED_MODEL_NAME"
    fi
}

gen_probe_ok() {
    # Executor liveness — 1 token must actually generate. es-lane answers in
    # seconds when healthy (MML 262144, admission open).
    if command -v curl >/dev/null 2>&1; then
        curl -fsS -m 60 -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"temperature\":0}" 2>/dev/null | grep -q "choices"
    fi
}

log_size() {
    # Bounded container-log size (last 2000 lines) — proxy for "log advanced".
    docker logs --tail 2000 "$CONTAINER_NAME" 2>/dev/null | wc -c || echo 0
}

log_tail() {
    docker logs --tail 200 "$CONTAINER_NAME" 2>/dev/null || true
}

WEDGE_PATTERN='Engine reset: engine_class=ccs|bcs|Fault response: Unsuccessful|guc_exec_queue_timedout_job|EngineDeadError|RPC call to sample_tokens timed out'

capture_and_kill() {
    local reason="$1" ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    local wedge_log="$RUN_DIR/wedge-${ts}.log"
    {
        echo "=== WEDGE DETECTED $(date -u) ==="
        echo "reason : $reason"
        echo "retry  : $((retries_used + 1))/$RETRIES  interval=${INTERVAL}s"
        echo ""
        echo "--- docker logs --tail 200 ($CONTAINER_NAME) ---"
        log_tail
    } > "$wedge_log"
    red "WEDGE DETECTED ($reason). Captured -> $wedge_log"

    # Capture-first: py-spy stacks BEFORE any restart.
    {
        echo ""
        echo "--- py-spy python-frame dumps + top-TID table before kill ---"
        PYSPY_BIN="$HOME/.local/bin/py-spy"
        [[ -x "$PYSPY_BIN" ]] || PYSPY_BIN="$(command -v py-spy || true)"
        if [[ -n "$PYSPY_BIN" && -x "$PYSPY_BIN" ]]; then
            while read -r wpid wargs; do
                echo "=== pid=$wpid $wargs"
                sudo -n "$PYSPY_BIN" dump --pid "$wpid" 2>&1 | head -80 || \
                    "$PYSPY_BIN" dump --pid "$wpid" 2>&1 | head -80
            done < <(docker top "$CONTAINER_NAME" -eo pid,args 2>/dev/null \
                     | grep -E "Worker_TP|EngineCore|multiprocessing\.spawn" | grep -v grep)
            echo "--- top-TID CPU per TP worker (utime+stime, lifetime) ---"
            for wpid in $(docker top "$CONTAINER_NAME" -eo pid,args 2>/dev/null \
                          | grep "Worker_TP" | grep -v grep | awk '{print $1}'); do
                echo "worker $wpid:"
                for t in $(ls /proc/$wpid/task 2>/dev/null); do
                    set -- $(awk '{print $14, $15}' /proc/$wpid/task/$t/stat 2>/dev/null)
                    echo "$(( $1 + $2 )) $t"
                done | sort -rn | head -3 | awk '{print "  tid="$2" ticks="$1}'
            done
        else
            echo "py-spy absent at $HOME/.local/bin/py-spy"
        fi
    } >> "$wedge_log"

    # Kill the hung process group: container main PID group, then docker rm -f.
    local cpid
    cpid=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER_NAME" 2>/dev/null || true)
    if [[ -n "$cpid" && "$cpid" != "0" ]] && kill -0 -- "-$cpid" 2>/dev/null; then
        kill -TERM -- "-$cpid" 2>/dev/null || true
        sleep 2
        kill -KILL -- "-$cpid" 2>/dev/null || true
    fi
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    echo "$wedge_log"
}

trap 'info "Watchdog stopped (signal). Wedge logs: $RUN_DIR/wedge-*.log"; exit 0' TERM INT

retries_used=0
stall_count=0
ever_running=false
ready_once=false   # cold-load guard: no stall counting until /v1/models answers once
last_log_size=$(log_size)

info "Wedge watchdog (es-lane) starting: interval=${INTERVAL}s retries=${RETRIES} container=$CONTAINER_NAME port=$PORT model=$SERVED_MODEL_NAME"
ok "Watchdog live. Liveness = gen-probe (1-token completion). Restart via: $RESTART_CMD (max ${RETRIES}x). py-spy captures before any restart." | tee -a "$WATCHDOG_LOG"

while :; do
    sleep "$INTERVAL"

    if container_running; then
        ever_running=true
    fi

    if health_ok && gen_probe_ok; then
        ready_once=true
        stall_count=0
        last_log_size=$(log_size)
        continue
    fi

    local_size=$(log_size)
    if [[ "$local_size" -gt "$last_log_size" ]]; then
        last_log_size=$local_size
        stall_count=0
        continue
    fi

    if ! container_running; then
        if [[ "$ever_running" == "true" ]]; then
            if [[ "$retries_used" -ge "$RETRIES" ]]; then
                red "  =================================================================="
                red "   WEDGE WATCHDOG GAVE UP after ${retries_used} restart attempts."
                red "   The server wedged again (Xe2 Level-Zero). A human must intervene."
                red "   Wedge evidence: ls $RUN_DIR/wedge-*.log"
                red "  =================================================================="
                exit 1
            fi
            wedge_log=$(capture_and_kill "container '$CONTAINER_NAME' no longer running after being up")
            retries_used=$((retries_used + 1))
            info "Restart attempt ${retries_used}/${RETRIES} via $RESTART_CMD ..."
            if WEDGE_WATCHDOG_ALREADY_RUNNING=1 $RESTART_CMD; then
                ok "Restart ${retries_used}/${RETRIES} succeeded ($wedge_log)."
                ever_running=false
                stall_count=0
                last_log_size=$(log_size)
            else
                err "Restart attempt ${retries_used}/${RETRIES} FAILED (restart command non-zero). Will retry on next probe."
            fi
            continue
        fi
        continue
    fi

    dev=$(device_count)
    sig=$(log_tail | grep -iE "$WEDGE_PATTERN" | tail -1 || true)
    if [[ "$dev" -lt "$PREFLIGHT_XPU_COUNT" ]]; then
        if [[ "$retries_used" -ge "$RETRIES" ]]; then
            red "  =================================================================="
            red "   WEDGE WATCHDOG GAVE UP: device health degraded (${dev}/${PREFLIGHT_XPU_COUNT} XPU visible) after ${retries_used} restarts. Human intervention required. Evidence: $RUN_DIR/wedge-*.log"
            red "  =================================================================="
            exit 1
        fi
        wedge_log=$(capture_and_kill "device health ${dev}/${PREFLIGHT_XPU_COUNT} XPU visible (sycl-ls//dev/dri)")
        retries_used=$((retries_used + 1))
        info "Restart attempt ${retries_used}/${RETRIES} via $RESTART_CMD ..."
        if WEDGE_WATCHDOG_ALREADY_RUNNING=1 $RESTART_CMD; then
            ok "Restart ${retries_used}/${RETRIES} succeeded ($wedge_log)."
            ever_running=false
            stall_count=0
            last_log_size=$(log_size)
        else
            err "Restart attempt ${retries_used}/${RETRIES} FAILED (restart command non-zero)."
        fi
        continue
    fi
    if [[ -n "$sig" ]]; then
        if [[ "$retries_used" -ge "$RETRIES" ]]; then
            red "  =================================================================="
            red "   WEDGE WATCHDOG GAVE UP after ${retries_used} restarts (wedge signature in log)."
            red "   Evidence: $RUN_DIR/wedge-*.log   Human intervention required."
            red "  =================================================================="
            exit 1
        fi
        wedge_log=$(capture_and_kill "wedge signature: $sig")
        retries_used=$((retries_used + 1))
        info "Restart attempt ${retries_used}/${RETRIES} via $RESTART_CMD ..."
        if WEDGE_WATCHDOG_ALREADY_RUNNING=1 $RESTART_CMD; then
            ok "Restart ${retries_used}/${RETRIES} succeeded ($wedge_log)."
            ever_running=false
            stall_count=0
            last_log_size=$(log_size)
        else
            err "Restart attempt ${retries_used}/${RETRIES} FAILED (restart command non-zero)."
        fi
        continue
    fi

    if [[ "$ready_once" != "true" ]]; then
        warn "[$(date -u +%H:%M:%SZ)] pre-READY hold: models endpoint has not answered yet - stall NOT counted (cold-load guard)"
        continue
    fi
    stall_count=$((stall_count + 1))
    warn "[$(date -u +%H:%M:%SZ)] no health/gen-probe + no log growth for ${stall_count}/${STALL_LIMIT} probes (dev=${dev})"
    if [[ "$stall_count" -ge "$STALL_LIMIT" ]]; then
        if [[ "$retries_used" -ge "$RETRIES" ]]; then
            red "  =================================================================="
            red "   WEDGE WATCHDOG GAVE UP after ${retries_used} restarts (server hung, no progress)."
            red "   Evidence: $RUN_DIR/wedge-*.log   Human intervention required."
            red "  =================================================================="
            exit 1
        fi
        wedge_log=$(capture_and_kill "hung server (no health/gen-probe, no log growth for ${STALL_LIMIT} probes)")
        retries_used=$((retries_used + 1))
        info "Restart attempt ${retries_used}/${RETRIES} via $RESTART_CMD ..."
        if WEDGE_WATCHDOG_ALREADY_RUNNING=1 $RESTART_CMD; then
            ok "Restart ${retries_used}/${RETRIES} succeeded ($wedge_log)."
            ever_running=false
            stall_count=0
            last_log_size=$(log_size)
        else
            err "Restart attempt ${retries_used}/${RETRIES} FAILED (restart command non-zero)."
        fi
    fi
done

#!/usr/bin/env bash
# ============================================================================
# wedge-watchdog.sh — production wedge watchdog (spawned by start.sh; the
# es-lane side-lane lineage this was adapted from is retired — this IS the
# production watchdog, lane-agnostic via .env / LANE_DIR).
# SINGLE-INSTANCE GUARD IS HOST-WIDE BY DESIGN: start.sh's
# `pgrep -f 'wedge-watchdog\.sh$'` allows exactly ONE watchdog per host —
# one rig, one serving lane. LANE_DIR separates runtime STATE (logs,
# pidfiles, wedge captures), not concurrency: two lanes on one rig would
# fight over the same GPUs, so a second watchdog is a bug, not a feature.
#
# The Xe2 Level-Zero wedge kills the serving job every 2-6 h under load
# (kernel signature: "Engine reset: engine_class=ccs|bcs",
#  "Fault response: Unsuccessful", "guc_exec_queue_timedout_job"). Only a
# container restart recovers; in-flight requests are lost.
#
# Design notes:
#   - container/port/served-id come from .env (CONTAINER_NAME/PORT/
#     SERVED_MODEL_NAME); defaults below match the production lane.
#   - engine log source is `docker logs $CONTAINER_NAME` (no host log-tail
#     follower); log_size/log_tail read the container log directly
#     (bounded: last 2000 lines / last 200 lines)
#   - boot grace: the lane boots in ~4-5 min. The same cold-load guard
#     applies — no stall counting until /v1/models answers once — so no
#     separate grace constant is needed.
#   - restart command: PROD_RESTART_CMD (default: "$SCRIPT_DIR/start.sh"
#     --launch; start.sh exports exactly this name when it spawns the
#     watchdog). ESLANE_RESTART_CMD is accepted as a deprecated alias.
#     ON-BOX DEPLOY NOTE: the deploy directory carries its own copy
#     (acceptance-v2 pattern). Restore scripts MUST use the deploy-dir
#     watchdog path, NOT <watchdog-path-in-operator-home> — that path does not
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
# State lives under the lane's .run/ ($LANE_DIR/.run, default repo root):
# watchdog.log, wedge-<ts>.log — the same dir status.sh and stop.sh read.
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
CONTAINER_NAME="${CONTAINER_NAME:-b70-lumnus-prod}"
PORT="${PORT:-8022}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-256k}"
PREFLIGHT_XPU_COUNT="${PREFLIGHT_XPU_COUNT:-4}"
STALL_LIMIT="${STALL_LIMIT:-3}"
# LIVENESS_CMD: test hook — when set, it REPLACES health_ok+gen_probe_ok as
# the liveness signal (exit 0 = alive). Default empty = the real curl probes.
LIVENESS_CMD="${LIVENESS_CMD:-}"
# PROD_RESTART_CMD is the name start.sh exports (start.sh watchdog block) —
# the watchdog MUST read the name its spawner writes. ESLANE_RESTART_CMD is a
# deprecated alias (old side-lane deployments). Fallback resolves to
# scripts/start.sh (this script's own dir), NOT ../start.sh — the deploy-dir
# path trap (a restore once launched a nonexistent on-box path silently).
RESTART_CMD="${PROD_RESTART_CMD:-${ESLANE_RESTART_CMD:-$SCRIPT_DIR/start.sh --launch}}"

# State lives in the LANE's .run/ — same derivation as start.sh/status.sh/
# stop.sh (LANE_DIR default = repo root = this script's parent), so
# status.sh tails this log and stop.sh's "wedge evidence" glob finds the
# captures. (Was: $SCRIPT_DIR/.run/eslane — invisible to both.)
RUN_DIR="${LANE_DIR:-$SCRIPT_DIR/..}/.run"
WATCHDOG_LOG="$RUN_DIR/watchdog.log"
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

WD_STATE_CMD="${WD_STATE_CMD:-}"   # test hook: echoes running|stalled|dead (see tests/watchdog-restart-test.sh)
container_running() {
    if [[ -n "$WD_STATE_CMD" ]]; then
        # Test hook: the stub's verdict IS the answer (never consult docker).
        [[ "$($WD_STATE_CMD)" == "running" ]]
        return
    fi
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
    # Executor liveness — 1 token must actually generate. The lane answers in
    # seconds when healthy (MML 262144, admission open).
    if command -v curl >/dev/null 2>&1; then
        curl -fsS -m 60 -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"temperature\":0}" 2>/dev/null | grep -q "choices"
    fi
}

log_size() {
    # Test hook: stub log-size source (keeps the numeric-only contract).
    # Test hook: stub log-size source — RUN the command (test passes 'echo 0').
    [[ -n "$WD_STATE_CMD" ]] && { eval "${WD_LOG_SIZE_CMD:-echo 0}"; return; }
    # Bounded container-log size (last 2000 lines) — proxy for "log advanced".
    # Numeric-only: docker CLI failure text (multi-line) must never reach the
    # -gt comparison (2026-10-03 line-215 "[[: 0\n0: syntax error" crash).
    _ls=$(docker logs --tail 2000 "$CONTAINER_NAME" 2>/dev/null | wc -c | tr -cd '0-9')
    echo "${_ls:-0}"
}

metrics_sample() {
    # Summed vLLM counters: running requests, generated tokens, prefilled tokens.
    curl -fsS -m 8 "http://localhost:$PORT/metrics" 2>/dev/null | awk '
        /^vllm:num_requests_running/ {r+=$NF}
        /^vllm:generation_tokens_total/ {g+=$NF}
        /^vllm:prompt_tokens_total/ {p+=$NF}
        END {printf "%d %d %d\n", r+0, g+0, p+0}'
}

engine_progressing() {
    # Load-aware gate (2026-10-03): the watchdog bounced a healthy engine saturated by
    # 8x~120K/200K-token prefills - every gen-probe queued out past its 60s window and
    # log_size is constant once the log exceeds the 2000-line tail, so stalls accumulated
    # into a restart. Under load a 1-token probe is NOT liveness evidence. Two /metrics
    # samples LOAD_SAMPLE_S apart: requests running AND either counter advanced =
    # busy, not wedged. Prompt tokens count as progress: a long prefill generates
    # ZERO output tokens for minutes, so the generation counter alone misreads a
    # busy engine as wedged (2026-10-03 23:53Z restart was that, not a real wedge).
    # Fail-open: metrics unavailable -> this path never restarts.
    local s1 s2
    s1=$(metrics_sample) || return 0
    sleep "${LOAD_SAMPLE_S:-30}"
    s2=$(metrics_sample) || return 0
    read -r r1 g1 p1 <<<"$s1"
    read -r r2 g2 p2 <<<"$s2"
    info "load-guard samples: s1=["$s1"] s2=["$s2"] (running gen prompt)"
    [[ "${r2:-0}" -gt 0 ]] || return 1
    [[ "${g2:-0}" -gt "${g1:-0}" || "${p2:-0}" -gt "${p1:-0}" ]] && return 0
    return 1
}

log_tail() {
    docker logs --tail 200 "$CONTAINER_NAME" 2>/dev/null || true
}

WEDGE_PATTERN='Engine reset: engine_class=ccs|bcs|Fault response: Unsuccessful|guc_exec_queue_timedout_job|EngineDeadError|RPC call to sample_tokens timed out'

capture_and_kill() {
    local reason="$1" ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    local wedge_log="$RUN_DIR/wedge-${ts}.log"
    # Test hook: stub lane has no docker container to capture/kill.
    if [[ -n "$WD_STATE_CMD" ]]; then
        { echo "=== WEDGE DETECTED $(date -u) ==="; echo "reason : $reason"; echo "retry  : $((retries_used + 1))/$RETRIES  interval=${INTERVAL}s"; } > "$wedge_log"
        echo "$wedge_log"
        return
    fi
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

info "Wedge watchdog starting: interval=${INTERVAL}s retries=${RETRIES} container=$CONTAINER_NAME port=$PORT model=$SERVED_MODEL_NAME state=$RUN_DIR"
ok "Watchdog live. Liveness = gen-probe (1-token completion). Restart via: $RESTART_CMD (max ${RETRIES}x). py-spy captures before any restart." | tee -a "$WATCHDOG_LOG"

while :; do
    sleep "$INTERVAL"
    # xe engine-reset monitor: log dmesg "Engine reset" count each cycle
    _rc_total=$(sudo -n dmesg 2>/dev/null | grep -acE "Engine reset" || true)
    _rc_total=${_rc_total:-0}
    info "xe engine resets: total=$_rc_total new_this_cycle=$(( _rc_total - ${_rc_last:-0} ))"
    _rc_last=$_rc_total

    if container_running; then
        ever_running=true
    fi

    if [[ -n "$LIVENESS_CMD" ]]; then
        _alive=false; if eval "$LIVENESS_CMD" >/dev/null 2>&1; then _alive=true; fi
    elif health_ok && gen_probe_ok; then
        _alive=true
    else
        _alive=false
    fi
    if [[ "$_alive" == "true" ]]; then
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
    if engine_progressing; then
        warn "[$(date -u +%H:%M:%SZ)] probe failed but engine is busy (requests running, tokens advancing) - stall NOT counted (load guard)"
        stall_count=0
        last_log_size=$(log_size)
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

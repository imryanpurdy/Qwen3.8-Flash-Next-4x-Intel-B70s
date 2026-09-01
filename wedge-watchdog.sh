#!/usr/bin/env bash
# ============================================================================
# wedge-watchdog.sh — MANDATORY Xe2 Level-Zero wedge watchdog
#                     (docs/lanes/deploy-kit-contract.md D3)
#
# The Xe2 Level-Zero wedge kills the serving job every 2-6 h under load
# (kernel signature: "Engine reset: engine_class=ccs|bcs",
#  "Fault response: Unsuccessful", "guc_exec_queue_timedout_job"). Only a
# container/engine restart recovers; in-flight requests are lost.
#
# The watchdog probes the container (docker inspect), the OpenAI health
# endpoint, log-tail liveness and device health (sycl-ls / xpu-smi /dev/dri
# render nodes) every WEDGE_WATCHDOG_INTERVAL (default 60 s). On wedge
# detection it:
#   1. captures the last 200 log lines + a timestamp to .run/wedge-<ts>.log
#   2. kills the hung process group (container group + docker rm -f)
#   3. restarts via ./start.sh --launch (bounded retries, default 3)
#   4. gives up LOUDLY after WEDGE_WATCHDOG_RETRIES with a red banner.
#
# DISABLE (double opt-out, non-negotiable): this watchdog only stands down
# when BOTH WEDGE_WATCHDOG_DISABLE=1 (in .env) AND PREFLIGHT_SKIPPED=1
# (--no-preflight) are set. start.sh refuses to launch otherwise.
#
# State lives under .run/ in this repo only.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; }
red()  { echo -e "\033[1;31m$*\033[0m"; }

# ---------------------------------------------------------------------------
# Double opt-out — both explicit acknowledgements required to stand down
# ---------------------------------------------------------------------------
WEDGE_WATCHDOG_DISABLE="${WEDGE_WATCHDOG_DISABLE:-0}"
PREFLIGHT_SKIPPED="${PREFLIGHT_SKIPPED:-0}"
if [[ "$WEDGE_WATCHDOG_DISABLE" == "1" && "$PREFLIGHT_SKIPPED" == "1" ]]; then
    red ""
    red "  =================================================================="
    red "   WEDGE WATCHDOG NOT RUNNING (WEDGE_WATCHDOG_DISABLE=1 + --no-preflight)"
    red "  =================================================================="
    red "   The rig WILL wedge unattended within 2-6 h under load. Xe2 Level-Zero"
    red "   wedge signature: 'Engine reset: engine_class=ccs|bcs', 'Fault response:"
    red "   Unsuccessful', 'guc_exec_queue_timedout_job'. Only a container/engine"
    red "   restart recovers; in-flight requests are lost. A human must watch it."
    red "  =================================================================="
    exit 0
fi
if [[ "$WEDGE_WATCHDOG_DISABLE" == "1" && "$PREFLIGHT_SKIPPED" != "1" ]]; then
    # Defensive: start.sh refuses this path, but if spawned anyway, supervise.
    warn "WEDGE_WATCHDOG_DISABLE=1 without --no-preflight is not a valid opt-out — supervising anyway."
fi

INTERVAL="${WEDGE_WATCHDOG_INTERVAL:-60}"
RETRIES="${WEDGE_WATCHDOG_RETRIES:-3}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
PREFLIGHT_XPU_COUNT="${PREFLIGHT_XPU_COUNT:-4}"
STALL_LIMIT=3                   # probes of log silence (no health, no growth) before declaring a hang

SERVER_LOG="$SCRIPT_DIR/.run/server.log"
mkdir -p "$SCRIPT_DIR/.run"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
device_count() {
    # Prefer sycl-ls (level_zero:gpu), fall back /dev/dri/renderD*, then xpu-smi.
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
            echo 4   # xpu-smi reports a live device stack; count not parseable portably
            return
        fi
    fi
    echo 0
}

container_running() {
    docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER_NAME"
}

health_ok() {
    # OpenAI /v1/models must answer with the served name.
    if command -v curl >/dev/null 2>&1; then
        curl -fsS -m 8 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_MODEL_NAME"
    else
        python3 -c 'import urllib.request,sys;d=urllib.request.urlopen("http://localhost:'"$PORT"'/v1/models",timeout=8).read().decode()' 2>/dev/null | grep -q "$SERVED_MODEL_NAME"
    fi
}

log_size() {
    [[ -f "$SERVER_LOG" ]] && wc -c < "$SERVER_LOG" 2>/dev/null || echo 0
}

log_tail() {
    [[ -f "$SERVER_LOG" ]] && tail -n 200 "$SERVER_LOG" 2>/dev/null || true
}

# Wedge signatures from the cookbook + shared context (docs/lanes/00 §Xe2 wedge).
WEDGE_PATTERN='Engine reset: engine_class=ccs|bcs|Fault response: Unsuccessful|guc_exec_queue_timedout_job|EngineDeadError|RPC call to sample_tokens timed out'

capture_and_kill() {
    local reason="$1" ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    local wedge_log="$SCRIPT_DIR/.run/wedge-${ts}.log"
    {
        echo "=== WEDGE DETECTED $(date -u) ==="
        echo "reason : $reason"
        echo "retry  : $((retries_used + 1))/$RETRIES  interval=${INTERVAL}s"
        echo "--- last 200 lines: .run/server.log ---"
        log_tail
        echo ""
        echo "--- docker logs --tail 200 ($CONTAINER_NAME) ---"
        docker logs --tail 200 "$CONTAINER_NAME" 2>/dev/null || true
    } > "$wedge_log"
    red "WEDGE DETECTED ($reason). Captured -> $wedge_log"

    # Kill the hung process group: container main PID group, then docker rm -f
    # (docker rm -f is graceful: SIGTERM, then SIGKILL after the stop timeout).
    local cpid
    cpid=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER_NAME" 2>/dev/null || true)
    if [[ -n "$cpid" && "$cpid" != "0" ]] && kill -0 -- "-$cpid" 2>/dev/null; then
        kill -TERM -- "-$cpid" 2>/dev/null || true
        sleep 2
        kill -KILL -- "-$cpid" 2>/dev/null || true
    fi
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    # Stop the log tail follower on the host so restarts re-tee cleanly.
    if [[ -f "$SCRIPT_DIR/.run/logtail.pid" ]]; then
        kill "$(cat "$SCRIPT_DIR/.run/logtail.pid")" 2>/dev/null || true
        rm -f "$SCRIPT_DIR/.run/logtail.pid"
    fi
    echo "$wedge_log"
}

# Graceful shutdown on signal (stop.sh sends TERM)
trap 'info "Watchdog stopped (signal). Wedge logs: .run/wedge-*.log"; exit 0' TERM INT

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
retries_used=0
stall_count=0
ever_running=false
last_log_size=$(log_size)

info "Wedge watchdog starting: interval=${INTERVAL}s retries=${RETRIES} container=$CONTAINER_NAME (probe cadence ${INTERVAL}s)"
ok "Watchdog live. Rig WILL be restarted (max ${RETRIES}x) if it wedges."

while :; do
    sleep "$INTERVAL"

    if container_running; then
        ever_running=true
    fi

    # Health endpoint answers -> healthy, reset stall.
    if health_ok; then
        stall_count=0
        last_log_size=$(log_size)
        continue
    fi

    # Log advanced since the last probe -> still loading/working.
    local_size=$(log_size)
    if [[ "$local_size" -gt "$last_log_size" ]]; then
        last_log_size=$local_size
        stall_count=0
        continue
    fi

    # No health, no log growth.
    if ! container_running; then
        if [[ "$ever_running" == "true" ]]; then
            # Was up, now gone -> wedge / crash.
            if [[ "$retries_used" -ge "$RETRIES" ]]; then
                red "  =================================================================="
                red "   WEDGE WATCHDOG GAVE UP after ${retries_used} restart attempts."
                red "   The server wedged again (Xe2 Level-Zero, recurs every 2-6 h under load)."
                red "   A human must intervene. Wedge evidence: ls .run/wedge-*.log"
                red "   DO NOT relaunch unattended — the wedge will recur."
                red "  =================================================================="
                exit 1
            fi
            wedge_log=$(capture_and_kill "container '$CONTAINER_NAME' no longer running after being up")
            retries_used=$((retries_used + 1))
            info "Restart attempt ${retries_used}/${RETRIES} via ./start.sh --launch ..."
            if WEDGE_WATCHDOG_ALREADY_RUNNING=1 "$SCRIPT_DIR/start.sh" --launch; then
                ok "Restart ${retries_used}/${RETRIES} succeeded ($wedge_log)."
                ever_running=false
                stall_count=0
                last_log_size=$(log_size)
            else
                err "Restart attempt ${retries_used}/${RETRIES} FAILED (start.sh --launch non-zero). Will retry on next probe."
            fi
            continue
        fi
        # Never observed running yet (start window) — wait.
        continue
    fi

    # Container running but stalled: probe device health + recent-log signature.
    dev=$(device_count)
    sig=$(log_tail | grep -iE "$WEDGE_PATTERN" | tail -1 || true)
    if [[ "$dev" -lt "$PREFLIGHT_XPU_COUNT" ]]; then
        if [[ "$retries_used" -ge "$RETRIES" ]]; then
            red "  =================================================================="
            red "   WEDGE WATCHDOG GAVE UP: device health degraded (${dev}/${PREFLIGHT_XPU_COUNT} XPU visible) after ${retries_used} restarts. Human intervention required. Evidence: .run/wedge-*.log"
            red "  =================================================================="
            exit 1
        fi
        wedge_log=$(capture_and_kill "device health ${dev}/${PREFLIGHT_XPU_COUNT} XPU visible (sycl-ls//dev/dri)")
        retries_used=$((retries_used + 1))
        info "Restart attempt ${retries_used}/${RETRIES} via ./start.sh --launch ..."
        if WEDGE_WATCHDOG_ALREADY_RUNNING=1 "$SCRIPT_DIR/start.sh" --launch; then
            ok "Restart ${retries_used}/${RETRIES} succeeded ($wedge_log)."
            ever_running=false
            stall_count=0
            last_log_size=$(log_size)
        else
            err "Restart attempt ${retries_used}/${RETRIES} FAILED (start.sh --launch non-zero)."
        fi
        continue
    fi
    if [[ -n "$sig" ]]; then
        if [[ "$retries_used" -ge "$RETRIES" ]]; then
            red "  =================================================================="
            red "   WEDGE WATCHDOG GAVE UP after ${retries_used} restarts (wedge signature in log)."
            red "   Evidence: .run/wedge-*.log   Human intervention required."
            red "  =================================================================="
            exit 1
        fi
        wedge_log=$(capture_and_kill "wedge signature: $sig")
        retries_used=$((retries_used + 1))
        info "Restart attempt ${retries_used}/${RETRIES} via ./start.sh --launch ..."
        if WEDGE_WATCHDOG_ALREADY_RUNNING=1 "$SCRIPT_DIR/start.sh" --launch; then
            ok "Restart ${retries_used}/${RETRIES} succeeded ($wedge_log)."
            ever_running=false
            stall_count=0
            last_log_size=$(log_size)
        else
            err "Restart attempt ${retries_used}/${RETRIES} FAILED (start.sh --launch non-zero)."
        fi
        continue
    fi

    # Stalled with the container alive and no signature yet: count, then wedge on a hang.
    stall_count=$((stall_count + 1))
    warn "[$(date -u +%H:%M:%SZ)] no health + no log growth for ${stall_count}/${STALL_LIMIT} probes (dev=${dev})"
    if [[ "$stall_count" -ge "$STALL_LIMIT" ]]; then
        if [[ "$retries_used" -ge "$RETRIES" ]]; then
            red "  =================================================================="
            red "   WEDGE WATCHDOG GAVE UP after ${retries_used} restarts (server hung, no progress)."
            red "   Evidence: .run/wedge-*.log   Human intervention required."
            red "  =================================================================="
            exit 1
        fi
        wedge_log=$(capture_and_kill "hung server (no health, no log growth for ${STALL_LIMIT} probes)")
        retries_used=$((retries_used + 1))
        info "Restart attempt ${retries_used}/${RETRIES} via ./start.sh --launch ..."
        if WEDGE_WATCHDOG_ALREADY_RUNNING=1 "$SCRIPT_DIR/start.sh" --launch; then
            ok "Restart ${retries_used}/${RETRIES} succeeded ($wedge_log)."
            ever_running=false
            stall_count=0
            last_log_size=$(log_size)
        else
            err "Restart attempt ${retries_used}/${RETRIES} FAILED (start.sh --launch non-zero)."
        fi
    fi
done

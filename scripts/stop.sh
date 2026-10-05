#!/usr/bin/env bash
# ============================================================================
# stop.sh — graceful stop: wedge watchdog FIRST, then the container.
#
# Order matters (the stop-order contract): TERM the watchdog so it cannot
# "detect a wedge" and restart the server mid-teardown, then `docker rm -f`
# the container (graceful: SIGTERM, SIGKILL after the stop timeout).
#
# Single-instance note: the watchdog is guarded host-wide (start.sh's pgrep
# guard allows exactly one per host), so killing by pidfile + the pidfile
# path sweep below is safe even when several lanes share one tree — each lane
# owns its own $LANE_DIR/.run/watchdog.pid.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

# .env is optional here (we only need CONTAINER_NAME / LANE_DIR defaults), but
# load it when present so custom names/paths are honored.
if [[ -f .env ]]; then
    # shellcheck source=.env
    source .env
fi
CONTAINER_NAME="${CONTAINER_NAME:-b70-lumnus-prod}"
LANE_DIR="${LANE_DIR:-$PWD}"
RUN_DIR="$LANE_DIR/.run"

# ---------------------------------------------------------------------------
# 1. Stop the wedge watchdog (TERM, graceful; KILL only if it won't die)
# ---------------------------------------------------------------------------
if [[ -f "$RUN_DIR/watchdog.pid" ]]; then
    WDPID=$(cat "$RUN_DIR/watchdog.pid")
    if [[ -n "$WDPID" ]] && kill -0 "$WDPID" 2>/dev/null; then
        info "Stopping wedge watchdog (PID $WDPID)..."
        kill -TERM "$WDPID" 2>/dev/null || true
        # Give it up to ~5 s to exit cleanly, then force.
        for _ in 1 2 3 4 5; do
            kill -0 "$WDPID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$WDPID" 2>/dev/null; then
            warn "Watchdog did not exit on TERM — sending KILL."
            kill -KILL "$WDPID" 2>/dev/null || true
        else
            ok "Watchdog stopped."
        fi
    else
        warn "No live watchdog at PID ${WDPID:-<empty>} (stale $RUN_DIR/watchdog.pid)."
    fi
    rm -f "$RUN_DIR/watchdog.pid"
else
    info "No $RUN_DIR/watchdog.pid — watchdog not running (or already stopped)."
fi

# Safety net: a watchdog spawned without a pidfile (manual run, crashed
# writer). The pattern is end-anchored so it cannot match a shell whose
# command string merely contains the name; the host-wide single-instance
# guard means at most one watchdog exists anyway.
for WDPID in $(pgrep -f 'wedge-watchdog\.sh$' 2>/dev/null); do
    warn "Lingering watchdog PID $WDPID without a live pidfile — terminating."
    kill -TERM "$WDPID" 2>/dev/null || true
done
sleep 1
pgrep -f 'wedge-watchdog\.sh$' >/dev/null 2>&1 && pkill -KILL -f 'wedge-watchdog\.sh$' 2>/dev/null || true

# Stop any lingering docker-logs follower on the host (not created by the
# watchdog; kept for manual `docker logs -f` sessions started earlier).
if [[ -f "$RUN_DIR/logtail.pid" ]]; then
    kill "$(cat "$RUN_DIR/logtail.pid")" 2>/dev/null || true
    rm -f "$RUN_DIR/logtail.pid"
fi

# ---------------------------------------------------------------------------
# 2. Stop the vLLM container (graceful rm -f: TERM then KILL after timeout)
# ---------------------------------------------------------------------------
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER_NAME"; then
    info "Stopping container '$CONTAINER_NAME' (docker rm -f, graceful)..."
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 && ok "Container '$CONTAINER_NAME' removed." || warn "docker rm -f '$CONTAINER_NAME' returned non-zero."
else
    info "Container '$CONTAINER_NAME' not present — nothing to stop."
fi

ok "Stopped. (Wedge evidence, if any: ls $RUN_DIR/wedge-*.log)"

# 

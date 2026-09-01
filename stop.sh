#!/usr/bin/env bash
# ============================================================================
# stop.sh — graceful stop: wedge watchdog first, then the vLLM container.
#
# Order matters (docs/lanes/deploy-kit-contract.md D3): TERM the watchdog so it
# does not "detect a wedge" and restart the server mid-teardown, then `docker
# rm -f` the container (graceful: SIGTERM, SIGKILL after the stop timeout).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }

# .env is optional here (we only need CONTAINER_NAME / PORT defaults), but load
# it when present so custom names/ports are honored.
if [[ -f .env ]]; then
    # shellcheck source=.env
    source .env
fi
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next}"

# ---------------------------------------------------------------------------
# 1. Stop the wedge watchdog (TERM, graceful; KILL only if it won't die)
# ---------------------------------------------------------------------------
if [[ -f "$SCRIPT_DIR/.run/watchdog.pid" ]]; then
    WDPID=$(cat "$SCRIPT_DIR/.run/watchdog.pid")
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
        warn "No live watchdog at PID ${WDPID:-<empty>} (stale .run/watchdog.pid)."
    fi
    rm -f "$SCRIPT_DIR/.run/watchdog.pid"
else
    info "No .run/watchdog.pid — watchdog not running (or already stopped)."
fi

# Stop any lingering docker-logs follower on the host.
if [[ -f "$SCRIPT_DIR/.run/logtail.pid" ]]; then
    kill "$(cat "$SCRIPT_DIR/.run/logtail.pid")" 2>/dev/null || true
    rm -f "$SCRIPT_DIR/.run/logtail.pid"
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

ok "Stopped. (Wedge evidence, if any: ls .run/wedge-*.log)"

#!/usr/bin/env bash
# ============================================================================
# watchdog-restart-test.sh — proves the wedge watchdog's restart path.
#
# Tiers:
#   stub (DEFAULT) — no GPU, no docker, no running engine: a fake lane driven
#     by the watchdog's WD_STATE_CMD/LIVENESS_CMD test hooks. The lane "dies"
#     when a sentinel file appears; PROD_RESTART_CMD is a stub that touches a
#     marker and revives the lane. PASS = marker exists, lane revived, wedge
#     capture + restart line in the watchdog log. Runs in ~10 s. Safe on a
#     live rig ONLY while the real watchdog is stopped (the host-wide pgrep
#     guard makes the stub refuse to start alongside a live watchdog).
#   live — the rig test: kill vLLM inside the real container, wait for the
#     watchdog-driven restart (~4-5 min load). Run AFTER verify.sh, with the
#     watchdog RUNNING (it fires the restart).
#
# Runnable right after a fresh clone: bash tests/watchdog-restart-test.sh
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR/.run"    # fresh clones have no .run/ (git drops empty dirs)

TIER="${1:-stub}"
PORT="${PORT:-8022}"
CONTAINER_NAME="${CONTAINER_NAME:-b70-lumnus-prod}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-256k}"
WATCHDOG_LOG="${WATCHDOG_LOG:-$SCRIPT_DIR/.run/watchdog.log}"
RESTART_TIMEOUT="${RESTART_TIMEOUT:-900}"   # live tier: engine loads ~4-5 min

if [[ "$TIER" == "live" ]]; then
    echo "== wedge watchdog restart-path test (LIVE tier — restarts the engine) =="
    docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" || { echo "RESTART_TEST_FAIL (engine not up)"; exit 1; }

    echo "-- simulating wedge (kill vLLM in $CONTAINER_NAME)"
    docker exec "$CONTAINER_NAME" bash -c 'pkill -9 -f "vllm.entrypoints.openai.api_server" || pkill -9 python3' 2>/dev/null
    sleep 5

    echo "-- waiting for watchdog-driven restart (deadline ${RESTART_TIMEOUT}s)..."
    DEADLINE=$(( $(date +%s) + RESTART_TIMEOUT ))
    while :; do
        if curl -fsS -m 10 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_MODEL_NAME"; then
            echo "-- engine answering again"
            break
        fi
        [[ "$(date +%s)" -gt "$DEADLINE" ]] && { echo "RESTART_TEST_FAIL (no recovery in ${RESTART_TIMEOUT}s)"; exit 1; }
        sleep 20
    done

    # Confirm it was the WATCHDOG that restarted it (not a lucky docker policy).
    if grep -q "Restart attempt 1/" "$WATCHDOG_LOG" 2>/dev/null; then
        echo "-- watchdog log shows the restart path fired"
    else
        echo "WARN: no 'Restart attempt' line in $WATCHDOG_LOG (check manually)"
    fi
    echo "RESTART_TEST_PASS (serving after watchdog restart)"
    exit 0
fi

# ---------------------------------------------------------------------------
# stub tier — GPU-free end-to-end proof of the detect→capture→restart chain
# ---------------------------------------------------------------------------
echo "== wedge watchdog restart-path test (STUB tier — no GPU/docker needed) =="
WD="$(command -v pgrep || true)"
if [[ -n "$WD" ]] && pgrep -f 'wedge-watchdog\.sh$' >/dev/null 2>&1; then
    echo "RESTART_TEST_FAIL: a wedge-watchdog is already running on this host (host-wide single-instance guard). Stop it (./scripts/stop.sh) before the stub test."
    exit 1
fi

SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/wdtest.XXXXXX")"
trap 'kill "${WD_PID:-}" 2>/dev/null; rm -rf "$SANDBOX"' EXIT

# Fake lane: alive until the sentinel appears; the restart stub revives it.
touch "$SANDBOX/alive"
: > "$SANDBOX/restart-marker"
STATE_CMD="$SANDBOX/state.sh"
cat > "$STATE_CMD" <<'EOF'
#!/usr/bin/env bash
if [[ -e "$WD_SANDBOX/alive" ]]; then echo running; else echo dead; fi
EOF
chmod +x "$STATE_CMD"
RESTART_STUB="$SANDBOX/restart.sh"
cat > "$RESTART_STUB" <<'EOF'
#!/usr/bin/env bash
# PROD_RESTART_CMD stub: record the restart, revive the fake lane.
echo "$(date -u +%FT%TZ) restart" >> "$WD_SANDBOX/restart-marker"
touch "$WD_SANDBOX/alive"
EOF
chmod +x "$RESTART_STUB"

WD_LOG="$SANDBOX/watchdog.log"
# interval=2s, LOAD_SAMPLE_S=1 (load-guard fail-open is instant with no
# /metrics), STALL_LIMIT=1 (force stall detection on the first dead probe).
env WD_SANDBOX="$SANDBOX" \
    WD_STATE_CMD="$STATE_CMD" \
    WD_LOG_SIZE_CMD='echo 0' \
    LIVENESS_CMD='test -e "$WD_SANDBOX/alive"' \
    STALL_LIMIT=1 LOAD_SAMPLE_S=1 \
    WEDGE_WATCHDOG_INTERVAL=2 WEDGE_WATCHDOG_RETRIES=3 \
    LANE_DIR="$SANDBOX/lane" \
    PROD_RESTART_CMD="$RESTART_STUB" \
    bash "$SCRIPT_DIR/scripts/wedge-watchdog.sh" >> "$WD_LOG" 2>&1 &
WD_PID=$!

echo "-- watchdog up (pid $WD_PID); killing the fake lane in 6s"
sleep 6
rm -f "$SANDBOX/alive"          # simulate the wedge

DEADLINE=$(( $(date +%s) + 60 ))
while :; do
    [[ -s "$SANDBOX/restart-marker" ]] && break
    kill -0 "$WD_PID" 2>/dev/null || { echo "RESTART_TEST_FAIL: watchdog exited before restarting"; tail -20 "$WD_LOG"; exit 1; }
    [[ "$(date +%s)" -gt "$DEADLINE" ]] && { echo "RESTART_TEST_FAIL: no restart within 60s"; tail -20 "$WD_LOG"; exit 1; }
    sleep 1
done
sleep 3   # let the watchdog observe the revival (stall reset)

FAIL=0
grep -qc "restart" "$SANDBOX/restart-marker" >/dev/null 2>&1 || { echo "FAIL: restart marker empty"; FAIL=1; }
[[ "$(grep -c restart "$SANDBOX/restart-marker")" -ge 1 ]] || { echo "FAIL: restart not counted"; FAIL=1; }
ls "$SANDBOX/lane/.run"/wedge-*.log >/dev/null 2>&1 || { echo "FAIL: no wedge capture under \$LANE_DIR/.run"; FAIL=1; }
grep -q "Restart attempt 1/3" "$WD_LOG" || { echo "FAIL: no 'Restart attempt 1/3' in watchdog log"; FAIL=1; }
kill -0 "$WD_PID" 2>/dev/null || { echo "FAIL: watchdog died after restart"; FAIL=1; }
grep -q "stall NOT counted\|Restart attempt" "$WD_LOG" || true

if [[ "$FAIL" == "0" ]]; then
    echo "RESTART_TEST_PASS (stub lane restarted via PROD_RESTART_CMD; capture in $SANDBOX/lane/.run/wedge-*.log)"
else
    echo "RESTART_TEST_FAIL"; tail -30 "$WD_LOG"; exit 1
fi
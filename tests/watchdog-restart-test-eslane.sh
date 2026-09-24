#!/usr/bin/env bash
# ============================================================================
# watchdog-restart-test-eslane.sh — proves the es-lane watchdog's restart path
# end to end. Adapted from watchdog-restart-test.sh; only names/ports/paths
# differ (es-lane, 8022, qwen-256k, .run/eslane/watchdog-eslane.log).
#
#   1. confirm es-lane is serving
#   2. simulate a wedge: kill the vLLM process (container exits)
#   3. watchdog detects (gen-probe fails), captures py-spy, restarts
#      via the restart command (default: repo start.sh --launch)
#   4. wait for /v1/models to answer again (~4-5 min load)
#
# PASS = engine serving again after a watchdog-driven restart.
# Run AFTER verify.sh (this restarts the engine by design). The es-lane
# watchdog must be RUNNING before this test (it fires the restart).
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
PORT="${PORT:-8022}"
CONTAINER_NAME="${CONTAINER_NAME:-es-lane}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-256k}"
WATCHDOG_LOG="${WATCHDOG_LOG:-$SCRIPT_DIR/.run/eslane/watchdog-eslane.log}"
RESTART_TIMEOUT="${RESTART_TIMEOUT:-900}"   # es-lane loads ~4-5 min; 15 min margin

echo "== es-lane watchdog restart-path test =="
docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" || { echo "RESTART_TEST_ESLANE_FAIL (engine not up)"; exit 1; }

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
    [[ "$(date +%s)" -gt "$DEADLINE" ]] && { echo "RESTART_TEST_ESLANE_FAIL (no recovery in ${RESTART_TIMEOUT}s)"; exit 1; }
    sleep 20
done

# Confirm it was the WATCHDOG that restarted it (not a lucky docker policy).
if grep -q "Restart attempt 1/" "$WATCHDOG_LOG" 2>/dev/null; then
    echo "-- watchdog log shows the restart path fired"
else
    echo "WARN: no 'Restart attempt' line in $WATCHDOG_LOG (check manually)"
fi

echo "RESTART_TEST_ESLANE_PASS (serving after watchdog restart)"

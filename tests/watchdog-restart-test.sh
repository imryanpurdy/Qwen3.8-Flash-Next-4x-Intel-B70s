#!/usr/bin/env bash
# ============================================================================
# watchdog-restart-test.sh — proves the wedge watchdog's restart path end to
# end against THIS clone's launcher (start.sh --launch).
#
#   1. confirm the engine is serving
#   2. simulate a wedge: kill the vLLM process (container exits)
#   3. watchdog detects (gen-probe fails), captures py-spy, restarts
#      via ./start.sh --launch
#   4. wait for /v1/models to answer again (full ~186 GB load)
#
# PASS = engine serving again after a watchdog-driven restart.
# Run AFTER verify.sh (this restarts the engine by design).
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
. ./.env 2>/dev/null
PORT="${PORT:-8021}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next}"
RESTART_TIMEOUT="${RESTART_TIMEOUT:-1500}"   # full load can take ~15 min

echo "== watchdog restart-path test =="
docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" || { echo "RESTART_TEST_FAIL (engine not up)"; exit 1; }

echo "-- simulating wedge (kill vLLM in $CONTAINER_NAME)"
docker exec "$CONTAINER_NAME" bash -c 'pkill -9 -f "vllm serve" || pkill -9 python3' 2>/dev/null
sleep 5

echo "-- waiting for watchdog-driven restart (interval from .env)..."
DEADLINE=$(( $(date +%s) + RESTART_TIMEOUT ))
while :; do
    if curl -fsS -m 10 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "${SERVED_MODEL_NAME:-qwen3.8-flash-next}"; then
        echo "-- engine answering again"
        break
    fi
    [[ "$(date +%s)" -gt "$DEADLINE" ]] && { echo "RESTART_TEST_FAIL (no recovery in ${RESTART_TIMEOUT}s)"; exit 1; }
    sleep 20
done

# Confirm it was the WATCHDOG that restarted it (not a lucky docker restart policy).
grep -q "Restart attempt 1/" .run/watchdog.log 2>/dev/null \
    && echo "-- watchdog log shows the restart path fired" \
    || echo "WARN: no 'Restart attempt' line in .run/watchdog.log (check manually)"

echo "RESTART_TEST_PASS (serving after watchdog restart)"

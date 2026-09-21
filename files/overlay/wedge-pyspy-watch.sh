#!/usr/bin/env bash
# wedge-pyspy-watch.sh — Friday-signature stall detector, capture-before-restart
# Ryan directive 2026-09-21: the 0.2 tok/s wedge is a FINDING, not an obstacle.
# Standing rule: on wedge signature, py-spy BEFORE any restart.
# Signature: >=3 consecutive 60s samples with Running>=1 AND gen-throughput<=0.3
#   (idle samples have Running 0 — do not count).
# On trigger: dump EngineCore + all Worker_TP* stacks (all threads — connector
#   thread visible) into .run/wedge-dumps/, echo WEDGE_CAPTURED, exit.
# P14 v2 suspicion: pre-v3 blocking-copy shape in the connector thread.
set -uo pipefail
cd /home/bonz/fn-recipe-int4
OUT=.run/wedge-dumps; mkdir -p "$OUT"
PYSPY=${PYSPY:-$HOME/bin/py-spy}
STREAK=0
for i in $(seq 1 150); do   # up to 2.5h
  sleep 60
  line=$(grep "loggers.py:310" .run/server.log | tail -1)
  gen=$(printf '%s' "$line" | grep -oE "generation throughput: [0-9.]+" | grep -oE "[0-9.]+")
  run=$(printf '%s' "$line" | grep -oE "Running: [0-9]+" | grep -oE "[0-9]+")
  if [ -n "$gen" ] && [ -n "$run" ] && [ "$run" -ge 1 ] && awk "BEGIN{exit !($gen <= 0.3)}"; then
    STREAK=$((STREAK+1))
  else
    STREAK=0
  fi
  if [ "$STREAK" -ge 3 ]; then
    echo "WEDGE_SIGNATURE t=$(date -Is) gen=$gen run=$run streak=$STREAK" | tee "$OUT/wedge.txt"
    EC=$(pgrep -f "EngineCore" | head -1)
    [ -n "$EC" ] && sudo "$PYSPY" dump --pid "$EC" > "$OUT/enginecore-$EC.txt" 2>&1 || true
    for w in $(pgrep -f "Worker_TP" | head -6); do
      sudo "$PYSPY" dump --pid "$w" > "$OUT/worker-$w.txt" 2>&1 || true
    done
    ls -la "$OUT" >> "$OUT/wedge.txt"
    echo WEDGE_CAPTURED
    exit 0
  fi
done
echo NO_WEDGE_IN_WINDOW

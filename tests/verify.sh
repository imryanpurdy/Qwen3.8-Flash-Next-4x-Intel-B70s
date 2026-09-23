#!/usr/bin/env bash
# ============================================================================
# verify.sh — acceptance + measurement of record for the v1 production line
#
#   ./tests/verify.sh            # full gate: tool-calls, 98K needle, sustained,
#                                # short-burst (labeled secondary), single-stream
#
# Metric discipline (2026-09-23 reconciliation — docs/rebuild/2026-09-23-
# measurement-reconciliation-soakfix-vs-bench-harness.md):
#   * EVERY number prints with its harness, aggregate formula, and prompt shape.
#   * The PRODUCTION metric is soakfix.py r2-r4 sustained (sum of completion
#     tokens over wall; open-ended prompt that runs to the token cap).
#   * bench_harness.py short-burst is labeled SECONDARY: same formula, but its
#     prompt EOS-stops near ~110-130 tokens regardless of the cap, so it
#     measures a ramp+drain slice, not sustained decode. NEVER compare a
#     soakfix number to a bench_harness number.
#   * Medians, not best, with spread.
#   * Single-stream: discard the first (cold) measurement.
#
# Gates (all must PASS): tool-call check, 98K needle, sustained threshold,
# graph-capture lines present. Fails loudly; exit code carries the result.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
. ./.env 2>/dev/null || { echo "FATAL: .env missing (cp .env.example .env)"; exit 2; }
PORT="${PORT:-8021}"
BASE="http://localhost:$PORT"
OUT="$SCRIPT_DIR/.run/verify.out"
: > "$OUT"
log() { echo "$*" | tee -a "$OUT"; }
hr()  { log "----------------------------------------------------------------"; }

PASS=0; FAIL=0
gate() {  # gate <name> <ok:0|1>
    if [[ "$2" == "0" ]]; then log "  [PASS] $1"; PASS=$((PASS+1));
    else log "  [FAIL] $1"; FAIL=$((FAIL+1)); fi
}

command -v docker >/dev/null 2>&1 || { echo "docker missing"; exit 2; }
docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME:-qwen38-flash-next}" \
    || { echo "FATAL: engine not running (./start.sh)"; exit 2; }

log "# VERIFY RUN $(date -u +%FT%TZ)"
log "# engine: $(docker inspect -f '{{.Config.Image}}' ${CONTAINER_NAME:-qwen38-flash-next})"
hr

# ---------------------------------------------------------------------------
# 1. Tool-call check — 5 sequential chat calls with a tool defined; every
#    answer must carry a well-formed tool_calls block (B3w payload).
# ---------------------------------------------------------------------------
log "## 1. Tool-call check (5 sequential, B3w payload)"
log "   harness: raw POST /v1/chat/completions, temperature=0, max_tokens=96"
TC_OK=0
for i in 1 2 3 4 5; do
    R=$(python3 - "$PORT" <<'PYT'
import json, sys, urllib.request
body = {"model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
        "tools": [{"type": "function", "function": {"name": "get_weather",
            "description": "Get the current weather conditions for a city.",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
            "required": ["location"]}}}],
        "tool_choice": "auto", "max_tokens": 96, "temperature": 0}
req = urllib.request.Request("http://localhost:%s/v1/chat/completions" % sys.argv[1],
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=180) as r:
    o = json.loads(r.read())
tc = o.get("choices", [{}])[0].get("message", {}).get("tool_calls")
print("OK" if tc and tc[0].get("function", {}).get("name") == "get_weather" else "MISS")
PYT
) 2>/dev/null
    [[ "$R" == "OK" ]] && TC_OK=$((TC_OK+1))
done
log "   result: $TC_OK/5 well-formed tool_calls"
gate "tool-call check ($TC_OK/5)" "$([[ $TC_OK -eq 5 ]] && echo 0 || echo 1)"
hr

# ---------------------------------------------------------------------------
# 2. 98K needle — scripts/needle_probe.py at 98,288+ tokens; PASS = answer
#    returned, content correct, staging_new == 0 (no PLE staging timeouts).
# ---------------------------------------------------------------------------
log "## 2. 98K needle probe"
log "   harness: scripts/needle_probe.py (single long-context request)"
log "   gates:   CORRECT=YES and STAGING_NEW=0 (no PLE staging timeouts)"
NEEDLE=$(python3 scripts/needle_probe.py --port "$PORT" --log "$SCRIPT_DIR/.run/server.log" --target-tokens 98288 2>&1)
echo "$NEEDLE" | tee -a "$OUT"
N_CORRECT=$(echo "$NEEDLE"  | grep -c '^CORRECT=YES')
N_STAGING=$(echo "$NEEDLE"  | grep -oE '^STAGING_NEW=[0-9]+' | head -1 | cut -d= -f2)
N_TOKENS=$(echo "$NEEDLE"   | grep -oE '^prompt_tokens=[0-9]+' | head -1 | cut -d= -f2)
log "   tokens=$N_TOKENS staging_new=${N_STAGING:-?}"
gate "98K needle (CORRECT=YES + STAGING_NEW=0)" "$([[ ${N_CORRECT:-0} -ge 1 && ${N_STAGING:-1} -eq 0 ]] && echo 0 || echo 1)"
hr

# ---------------------------------------------------------------------------
# 3. PRODUCTION METRIC — soakfix.py sustained (r2-r4)
# ---------------------------------------------------------------------------
log "## 3. Sustained throughput — PRODUCTION METRIC"
log "   harness: soakfix.py"
log "   formula: agg = sum(completion_tokens) / round_wall; sustained = mean(r2..r4)"
log "   prompt:  open-ended technical-essay instruction; runs TO the token cap"
log "   note:    r1 is warmup and is excluded by the harness itself"
SF=$(python3 scripts/soakfix.py 16 2>&1)
echo "$SF" | tee -a "$OUT"
SF_SUS=$(echo "$SF" | grep -oE 'sustained_agg=[0-9.]+' | cut -d= -f2)
SF_MIN=$(echo "$SF"  | grep -oE 'min_agg=[0-9.]+'    | cut -d= -f2)
SF_MAX=$(echo "$SF"  | grep -oE 'max_agg=[0-9.]+'    | cut -d= -f2)
log "   16x600 sustained: $SF_SUS tok/s (spread $SF_MIN-$SF_MAX)"
gate "16-way sustained >= 280 (of record band 294-320)" \
     "$(python3 -c "print(0 if float('${SF_SUS:-0}') >= 280 else 1)")"

SF8=$(python3 scripts/soakfix.py 8 2>&1)
echo "$SF8" | tee -a "$OUT"
SF8_SUS=$(echo "$SF8" | grep -oE 'sustained_agg=[0-9.]+' | cut -d= -f2)
log "   8x600 sustained: $SF8_SUS tok/s"
hr

# ---------------------------------------------------------------------------
# 4. Short-burst — bench_harness.py (SECONDARY; do not compare to row 3)
# ---------------------------------------------------------------------------
log "## 4. Short-burst — SECONDARY (different regime, see formula + prompt)"
log "   harness: bench_harness.py burst"
log "   formula: agg = sum(completion_tokens) / round_wall (identical formula)"
log "   prompt:  'Write a short thank-you note of exactly three sentences'"
log "   note:    EOS-stops near ~110-130 tok/stream -> ramp+drain slice; NOT"
log "            comparable to row 3 (2026-09-23 reconciliation)"
BH=$(python3 scripts/bench_harness.py burst --workers 16 --rounds 4 --max-tokens 320 2>&1)
echo "$BH" | tee -a "$OUT"
BH_MED=$(echo "$BH" | grep -oE 'ROUND_[0-9]+_aggregate_tokps=[0-9.]+' | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END{if(NR%2)print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2}')
log "   16x320 short-burst median: $BH_MED tok/s (secondary)"
hr

# ---------------------------------------------------------------------------
# 5. Single-stream at N=20, first measurement discarded (cold)
# ---------------------------------------------------------------------------
log "## 5. Single-stream (N=20, first discarded)"
log "   formula: completion_tokens / wall per request; median over 19"
SS=$(python3 - "$PORT" <<'PYS'
import json, sys, time, urllib.request
lat = []
for i in range(20):
    body = {"model": "qwen3.8-flash-next",
            "messages": [{"role": "user", "content": "Summarize the plot of Romeo and Juliet in a few paragraphs."}],
            "max_tokens": 600, "temperature": 0}
    req = urllib.request.Request("http://localhost:%s/v1/chat/completions" % sys.argv[1],
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        o = json.loads(r.read())
    wall = time.time() - t0
    ct = o["usage"]["completion_tokens"]
    lat.append(ct / wall)
disc, rest = lat[0], lat[1:]
rest.sort()
n = len(rest)
med = rest[n//2] if n % 2 else (rest[n//2-1]+rest[n//2])/2
print("discarded_first=%.1f median19=%.1f min=%.1f max=%.1f" % (disc, med, rest[0], rest[-1]))
PYS
) 2>&1
echo "$SS" | tee -a "$OUT"
SS_MED=$(echo "$SS" | grep -oE 'median19=[0-9.]+' | cut -d= -f2)
log "   single-stream median: $SS_MED tok/s"
hr

# ---------------------------------------------------------------------------
# 6. Graph-capture gate — decode graphs actually captured at boot
# ---------------------------------------------------------------------------
log "## 6. Graph-capture gate"
CAPN=$(docker logs "${CONTAINER_NAME:-qwen38-flash-next}" 2>&1 | grep -c "Graph capturing finished")
log "   'Graph capturing finished' lines: $CAPN"
gate "graph capture present" "$([[ ${CAPN:-0} -ge 1 ]] && echo 0 || echo 1)"

# ---------------------------------------------------------------------------
log ""
log "# SUMMARY"
log "  16x600 sustained (PRODUCTION): $SF_SUS tok/s (spread $SF_MIN-$SF_MAX)"
log "   8x600 sustained (PRODUCTION): $SF8_SUS tok/s"
log "  16x320 short-burst (secondary): $BH_MED tok/s"
log "  single-stream median:           $SS_MED tok/s"
log "  gates: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]] && log "  VERIFY: PASS" || log "  VERIFY: FAIL"
exit "$FAIL"

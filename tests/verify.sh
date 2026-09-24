#!/usr/bin/env bash
# ============================================================================
# verify.sh — acceptance + measurement of record for the SIDE-LANE line
# (their stack: vLLM devan-carlin/vllm@xpu-qwen4exp a69fba21; 2026-09-23)
#
#   ./tests/verify.sh            # full gate: tool-calls (structural), 97K
#                                # needle, sustained, single-stream, receipt
#
# Metric discipline (comparability law — every number prints its harness,
# aggregate formula, and prompt shape):
#   * The GATE metric is sidelane-soakfix.py 16x600: agg = Σcompletion_tokens
#     ÷ round_wall per round; open-ended prompt that runs TO the token cap;
#     r1 discarded as warmup by the harness. Gate = MEDIAN r2..r15 computed
#     from the JSON rounds (the harness's printed sustained_agg is the MEAN
#     of the same window — both are reported).
#   * Medians, not best, with spread.
#   * Single-stream: discard the first (cold) measurement.
#   * Tool calls compared STRUCTURALLY (name + argument JSON as objects),
#     never by raw text (2026-09-23 comparator law).
#
# Gates (all must PASS): tool-call structural 20/20 + multitool + nested,
# 97K needle, sustained median within 10% of 626.3, single-stream, receipt.
# Fails loudly; exit code carries the result.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
. ./.env 2>/dev/null || { echo "FATAL: .env missing (cp .env.example .env)"; exit 2; }
PORT="${PORT:-8022}"
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
docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME:-es-lane}" \
    || { echo "FATAL: engine not running (./start.sh)"; exit 2; }

log "# VERIFY RUN $(date -u +%FT%TZ)"
log "# engine: $(docker inspect -f '{{.Config.Image}}' ${CONTAINER_NAME:-es-lane})"
log "# line:   TP4+EP MML=${MAX_MODEL_LEN:-262144} MNS=${MAX_NUM_SEQS:-16} kv=fp8 (side-lane their-stack line)"
hr

# ---------------------------------------------------------------------------
# 1. Tool-call check — 20 calls, STRUCTURAL compare (name + args as objects)
#    harness: scripts/sidelane/sidelane-toolcall.py (promotion battery:
#    20 calls incl. multi-tool pick + nested-arguments case)
# ---------------------------------------------------------------------------
log "## 1. Tool-call check — 20 structural (incl. multi-tool + nested args)"
log "   harness: scripts/sidelane/sidelane-toolcall.py"
log "   compare: function name + argument JSON as OBJECTS (canonicalized),"
log "            never raw text (comparator law 2026-09-23)"
TC=$(python3 scripts/sidelane/sidelane-toolcall.py --port "$PORT" --count 20 2>&1)
echo "$TC" | tee -a "$OUT"
TC_V=$(echo "$TC"   | grep -oE 'TOOLCALL20_VERDICT=[0-9]+/[0-9]+' | head -1 | cut -d= -f2)
TC_OK=${TC_V%%/*}; TC_TOT=${TC_V##*/}
MT=$(echo "$TC"     | grep -oE 'MULTITOOL_VERDICT=(CORRECT_PICK|CHECK_RAW)' | head -1 | cut -d= -f2)
NV=$(echo "$TC"     | grep -oE 'NESTED_VERDICT=(PASS|FAIL)' | head -1 | cut -d= -f2)
gate "tool-call structural ${TC_OK:-?}/${TC_TOT:-?} + multitool(${MT:-MISS}) + nested(${NV:-MISS})" \
     "$([[ "${TC_OK:-0}" -eq 20 && "${TC_TOT:-0}" -eq 20 && "${MT:-}" == "CORRECT_PICK" && "${NV:-}" == "PASS" ]] && echo 0 || echo 1)"
hr

# ---------------------------------------------------------------------------
# 2. 97K needle — sidelane-needle-probe.py (v2.1 wrapper over needle_probe,
#    engine-calibrated sizing); PASS = CORRECT + SIZE_OK >= 97000 engine tokens
# ---------------------------------------------------------------------------
log "## 2. 97K needle probe"
log "   harness: scripts/sidelane/sidelane-needle-probe.py v2.1 (engine-calibrated:"
log "            sizes to the tokenizer via usage.prompt_tokens; gate number is the"
log "            ENGINE-confirmed token count, never the estimate)"
log "   gates:   CORRECT=YES + SIZE_OK=YES (>=97000 engine tokens)"
NEEDLE=$(python3 scripts/sidelane/sidelane-needle-probe.py --port "$PORT" \
            --target-tokens 97800 --min-prompt-tokens 97000 \
            --salt "verify-97k-$(date -u +%s)" 2>&1)
echo "$NEEDLE" | tee -a "$OUT"
N_CORRECT=$(echo "$NEEDLE" | grep -c '^CORRECT=YES')
N_SIZE=$(echo "$NEEDLE"    | grep -c '^SIZE_OK=YES')
N_TOKENS=$(echo "$NEEDLE"  | grep -oE '^ENGINE_PROMPT_TOKENS=[0-9]+' | head -1 | cut -d= -f2)
log "   engine_prompt_tokens=$N_TOKENS"
gate "97K needle (CORRECT + SIZE>=97000)" "$([[ ${N_CORRECT:-0} -ge 1 && ${N_SIZE:-0} -ge 1 ]] && echo 0 || echo 1)"
hr

# ---------------------------------------------------------------------------
# 3. Sustained — sidelane-soakfix.py 16x600, 15 rounds — GATE METRIC
#    printed sustained_agg = MEAN r2..r15; GATE = MEDIAN r2..r15 (from JSON)
# ---------------------------------------------------------------------------
log "## 3. Sustained throughput — GATE METRIC (16x600 soakfix)"
log "   harness: scripts/sidelane/sidelane-soakfix.py"
log "   formula: agg = sum(completion_tokens)/round_wall per round;"
log "            r1 warmup discarded; sustained_agg printed = MEAN r2..r15;"
log "            GATE = MEDIAN r2..r15 (computed from JSON rounds)"
log "   prompt:  open-ended technical-essay instruction; runs TO the token cap"
SF=$(python3 scripts/sidelane/sidelane-soakfix.py --n 16 --max-tokens 600 --rounds 15 \
        --json-out .run/verify-soakfix16.json 2>&1)
echo "$SF" | tail -20 | tee -a "$OUT"
SF_SUS=$(echo "$SF" | grep -oE 'sustained_agg=[0-9.]+' | cut -d= -f2)
SF_MIN=$(echo "$SF" | grep -oE 'min_agg=[0-9.]+'      | cut -d= -f2)
SF_MAX=$(echo "$SF" | grep -oE 'max_agg=[0-9.]+'      | cut -d= -f2)
SF_CLEAN=$(echo "$SF" | grep -oE 'clean=(YES|NO)'     | cut -d= -f2)
SF_MED=$(python3 - <<'PYM'
import json
try:
    with open(".run/verify-soakfix16.json") as fh:
        aggs = [r["agg"] for r in json.load(fh)["rounds"]][1:]   # r2..r15
    aggs.sort()
    n = len(aggs)
    print("%.1f" % (aggs[n//2] if n % 2 else (aggs[n//2-1]+aggs[n//2])/2) if n else 0)
except Exception:
    print(0)
PYM
)
log "   16x600 sustained: median r2..r15 = $SF_MED tok/s (harness mean $SF_SUS; spread $SF_MIN-$SF_MAX)"
gate "16x600 sustained median within 10% of 626.3 (band 563.7-688.9)" \
     "$(python3 -c "print(0 if 563.7 <= float('${SF_MED:-0}') <= 688.9 else 1)")"
gate "soak clean (0 errors, post-check OK)" "$([[ "${SF_CLEAN:-NO}" == "YES" ]] && echo 0 || echo 1)"
hr

# ---------------------------------------------------------------------------
# 4. Single-stream at N=20, first measurement discarded (cold)
# ---------------------------------------------------------------------------
log "## 4. Single-stream (N=20, first discarded)"
log "   harness: scripts/sidelane/sidelane-single-stream.py"
log "   formula: completion_tokens / wall per request; median over 19"
SS=$(python3 scripts/sidelane/sidelane-single-stream.py --port "$PORT" 2>&1)
echo "$SS" | tee -a "$OUT"
SS_MED=$(echo "$SS" | grep -oE 'median19=[0-9.]+' | cut -d= -f2)
log "   single-stream median: $SS_MED tok/s"
gate "single-stream median >= 40 (validated band 52.3-52.8)" \
     "$(python3 -c "print(0 if float('${SS_MED:-0}') >= 40 else 1)")"
hr

# ---------------------------------------------------------------------------
# 5. Boot-receipt gate — 'Application startup complete' (the verified READY
#    receipt of every boot of record; graphs capture inside torch.compile
#    before it on this stack)
# ---------------------------------------------------------------------------
log "## 5. Boot-receipt gate"
RCPT=$(docker logs "${CONTAINER_NAME:-es-lane}" 2>&1 | grep -c "Application startup complete")
log "   'Application startup complete' lines: $RCPT"
gate "startup receipt present" "$([[ ${RCPT:-0} -ge 1 ]] && echo 0 || echo 1)"

# ---------------------------------------------------------------------------
log ""
log "# SUMMARY"
log "  tool-call structural:          ${TC_OK:-?}/${TC_TOT:-?} + multitool=${MT:-MISS} + nested=${NV:-MISS}"
log "  97K needle:                    CORRECT=${N_CORRECT:-0} SIZE_OK=${N_SIZE:-0} (engine tokens ${N_TOKENS:-?})"
log "  16x600 sustained (GATE):       median ${SF_MED:-?} / mean ${SF_SUS:-?} tok/s (spread ${SF_MIN:-?}-${SF_MAX:-?}, clean=${SF_CLEAN:-?})"
log "  single-stream median:          ${SS_MED:-?} tok/s"
log "  startup receipt lines:         ${RCPT:-0}"
log "  gates: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]] && log "  VERIFY: PASS" || log "  VERIFY: FAIL"
exit "$FAIL"

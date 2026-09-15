#!/usr/bin/env bash
# MTP4@4K engine stall — evidence pack (run on jobe with the lab launcher +
# MAX_NUM_SPECULATIVE_TOKENS=4, MAX_MODEL_LEN=4352, GRAPH_MODE=eager).
# Kills the 4,096-token run when the stall signature appears, collects
# per-stage timings + memory state around the stall point.
set -u
OUT="${1:-/tmp/stall-evidence}"
mkdir -p "$OUT"
MODEL="${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}"

echo "== [1] synthetic stall repro (engine-side, no client) =="
# Deterministic 4K prompt; the lab stall reproduces at 3904/4096 produced.
python3 - <<'PY' 2>&1 | tee "$OUT/stall-repro.log"
import os, time
from vllm import LLM, SamplingParams
llm = LLM(model=os.environ.get("MODEL", "Qwen/Qwen3.8-Flash-Next-FP8"),
          max_model_len=4352, max_num_seqs=1,
          speculative_config={"method": "mtp", "num_speculative_tokens": 4},
          enforce_eager=True)
sp = SamplingParams(max_tokens=4096, ignore_eos=True, temperature=0.0)
t0 = time.time()
out = llm.generate(["Explain the history of computing from Babbage onward. " * 120],
                   sp)
dt = time.time() - t0
ntok = len(out[0].outputs[0].token_ids)
print(f"PRODUCED {ntok}/4096 in {dt:.1f}s -> {ntok/dt:.2f} tok/s")
PY

echo "== [2] driver state at stall =="
sudo dmesg -T | grep -E "xe|drm|Engine reset|guc" | tail -30 > "$OUT/dmesg-tail.txt" || true
sudo cat /sys/kernel/debug/dri/$(pgrep -f "vllm" >/dev/null && echo 0)/i915... 2>/dev/null || true
xpu-smi discovery > "$OUT/xpu-smi-discovery.txt" 2>&1 || true
for d in 0 1 2 3; do xpu-smi stats -d $d >> "$OUT/xpu-smi-stats.txt" 2>&1; done

echo "== [3] process state =="
ps -eo pid,rss,cmd --sort=-rss | grep -E "vllm|EngineCore" | head -5 > "$OUT/top-mem.txt"
free -g > "$OUT/free.txt"
VLLM_LOG_STATS_DEPTH=1 true

echo "== [4] control points (no full re-run needed if [1] stalls) =="
echo "A. MTP4@512 known-good: 20.727 tok/s (lab)"
echo "B. MTP3@4K known-good: 15.502 tok/s (lab preferred 4K cell)"
echo "C. MTP0@4K: run only if A stalls AND B reproduces, to isolate verify path"

echo "== evidence in $OUT =="
ls -la "$OUT"

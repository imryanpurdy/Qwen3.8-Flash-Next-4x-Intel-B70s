#!/usr/bin/env bash
# ============================================================================
# p16-qsa-indexer-prefill-smem.sh — P16: B70-sized prefill indexer launch config
#
# Boot #14 evidence (2026-09-21): compile-time warmup dies in
# _qsa_mqa_paged_prefill_kernel (_prefill_logits, qsa_indexer.py) with
#   OutOfResources: shared memory, Required: 278528, Hardware limit: 131072
# Smem arithmetic: q tile = TILE_R(64) x NUM_HEADS_PADDED(16) x HEAD_DIM(128)
# x 2 B = 262144 B, + one BLOCK_N(64) x 128 x 2 B keys stage = 16384 B
# => 278528 B exactly. The fork sized TILE_R for >=256 KiB-smem GPUs; B70
# has 128 KiB. Old box never hit this: v24h2 has no qsa_indexer.py at all
# (the fork rewrote the indexer; the old overlay ran an older design).
#
# Fix: TILE_R 64 -> 16 at the _prefill_logits launch site (one line).
#   q tile drops to 65536 B; + keys stages 32768 B ~= 96 KiB <= 128 KiB.
#   Grid dim 1 = cdiv(num_queries, TILE_R) auto-compensates; kernel logic
#   is constexpr-parameterized (lanes = arange(TILE_R), m = TILE_R*NH_PAD).
#   Correctness identical: rows partition across more programs.
# Idempotent, drift-failing. P9-P15 series pattern.
# ============================================================================
set -euo pipefail

F=/opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py

python - "$F" <<'PYEOF'
import sys
p = sys.argv[1]
src = open(p).read()

old = """    TILE_R = 64
    BLOCK_N = 64
    K_TILES = 16"""
new = """    # P16 (2026-09-21): B70 shared memory is 128 KiB; TILE_R=64 makes the
    # resident q tile 64*NUM_HEADS_PADDED*HEAD_DIM*2 = 256 KiB -> OutOfResources
    # at compile (_init_handles). 16 rows fit (~96 KiB with keys stages); the
    # grid's row dimension auto-compensates (cdiv(num_queries, TILE_R)).
    TILE_R = 16
    BLOCK_N = 64
    K_TILES = 16"""

if "P16 (2026-09-21)" in src:
    print("P16: already applied")
elif old not in src:
    print("P16: anchor NOT FOUND - drift, refusing", file=sys.stderr)
    sys.exit(1)
else:
    src = src.replace(old, new, 1)
    open(p, "w").write(src)
    print("P16: applied")
PYEOF

# ---- Assert: smem budget for the launch config on B70 ----------------------
python - <<'PYEOF'
NH_PAD, HEAD_DIM, TILE_R, BLOCK_N = 16, 128, 16, 64
q_tile = TILE_R * NH_PAD * HEAD_DIM * 2          # resident before loop
keys_stage = BLOCK_N * HEAD_DIM * 2              # per pipeline stage
required = q_tile + keys_stage * 2               # STAGES=2
LIMIT = 131072
assert required <= LIMIT, (required, LIMIT)
print(f"P16 assert OK: prefill smem {required} B <= {LIMIT} B (TILE_R=16, NH_PAD={NH_PAD})")
PYEOF

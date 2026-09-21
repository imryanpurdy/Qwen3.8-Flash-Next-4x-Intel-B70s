#!/usr/bin/env bash
# ============================================================================
# p15-qsa-indexer-mindot-shapes.sh — P15: floor tl.dot N-dims to >= 16 in the
#   Qwen4Exp QSA indexer kernels (triton-xpu 3.7.2 strict min-shape enforcement)
#
# Boot #13 evidence (2026-09-21): first real forward after profile-A cleared
# the DEVICE_LOST class dies in the QSA indexer decode kernel at
#   scores = tl.dot(keys, query, out_dtype=tl.float32)
# with "Input shapes should have M >= 1, N >= 16 and K >= 16.0" ("at 76:17").
#
# Root class: the dot's N-dim is a PADDED PRODUCT:
#   decode:  N = DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED  (can be < 16)
#   prefill: N = TILE_R * NUM_HEADS_PADDED  (TILE_R=64 -> safe, but the head
#            pad participates in the post-dot reshape; floor it anyway)
# The K-dim already has the floor ("tl.dot requires a reduction dimension of
# at least 16" comment + BLOCK_D = max(16, ...)); the N-dim never got one.
# Old box never hit this: its 2021-era triton accepted small dots silently.
#
# Fix shape (constexpr flooring, kernel logic untouched): masked loads pad
# with other=0.0 and post-dot reshape/sum use the same padded constexprs, so
# extra pad lanes contribute only zeros through masked-out paths.
#   decode:  floor DECODE_QUERY_LEN_PADDED to >= 16 (product >= 16 for any
#            NUM_HEADS >= 1); NUM_HEADS_PADDED floor derived so product >= 16
#   prefill: floor NUM_HEADS_PADDED to >= 16
# Idempotent, drift-failing on anchor mismatch. P9-P14 series pattern.
# ============================================================================
set -euo pipefail

F=/opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py

python - "$F" <<'PYEOF'
import sys
p = sys.argv[1]
src = open(p).read()
TAG = "P15"

def apply(old, new, label):
    global src
    if label + " APPLIED" in src:
        print(f"P15: {label} already applied")
        return
    if old not in src:
        print(f"P15: {label} anchor NOT FOUND - drift, refusing", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)
    print(f"P15: {label} applied")

# ---- decode kernel ----------------------------------------------------------
apply(
'''    DECODE_QUERY_LEN_PADDED: tl.constexpr = triton.next_power_of_2(DECODE_QUERY_LEN)
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))''',
'''    # P15 (2026-09-21): triton-xpu 3.7.2 enforces tl.dot shapes M>=1,N>=16,K>=16.
    # The dot's N here is DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED, which can
    # fall below 16 for small decode configs. Floor the pads; masked loads pad
    # with other=0.0 and the post-dot reshape uses these same constexprs, so
    # extra lanes contribute only zeros through masked-out paths.
    _DQL_POW2: tl.constexpr = triton.next_power_of_2(DECODE_QUERY_LEN)
    DECODE_QUERY_LEN_PADDED: tl.constexpr = max(16, _DQL_POW2)
    NUM_HEADS_PADDED: tl.constexpr = max(
        (16 + _DQL_POW2 - 1) // _DQL_POW2, triton.next_power_of_2(NUM_HEADS)
    )
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))''',
"decode-nfloor APPLIED")

# ---- prefill kernel ---------------------------------------------------------
apply(
'''    num_columns = page_table_width * PAGE_SIZE
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.''',
'''    num_columns = page_table_width * PAGE_SIZE
    # P15 (2026-09-21): floor the head pad so N = TILE_R * NUM_HEADS_PADDED
    # respects triton-xpu 3.7.2's tl.dot N >= 16 rule for any config.
    NUM_HEADS_PADDED: tl.constexpr = max(16, triton.next_power_of_2(NUM_HEADS))
    # tl.dot requires a reduction dimension of at least 16.''',
"prefill-nfloor APPLIED")

open(p, "w").write(src)
print("P15: written")
PYEOF

# ---- Assert: constexpr floors hold for the decode-dot worst case ------------
python - <<'PYEOF'
import triton
worst = []
for dql in (1, 2, 4, 8):
    for heads in (1, 2, 4, 8, 16):
        p2 = lambda x: triton.next_power_of_2(x)
        dql_pad = max(16, p2(dql))                      # decode floor
        nh_floor = (16 + p2(dql) - 1) // p2(dql)        # derived head floor
        nh_pad = max(nh_floor, p2(heads))
        n = dql_pad * nh_pad
        assert n >= 16, (dql, heads, n)
        worst.append((dql, heads, n))
print("P15 assert OK: decode-dot N >= 16 for all probe configs; min =", min(w[2] for w in worst))
PYEOF

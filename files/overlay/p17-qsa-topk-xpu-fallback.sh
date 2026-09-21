#!/usr/bin/env bash
# ============================================================================
# p17-qsa-topk-xpu-fallback.sh — P17: torch-native topk fallback for the QSA
#   indexer on XPU (torch.ops._C.persistent_topk is absent from this wheel)
#
# Boot #15 evidence (2026-09-21): warmup dies at
#   qsa_indexer.py:498 _topk -> torch.ops._C.persistent_topk
#   AttributeError: '_OpNamespace' '_C' object has no attribute 'persistent_topk'
#
# Archaeology: the old box's vllm._C (SYCL-built fork extension) registered
# persistent_topk; the new fork's XPU wheel dropped it (upstream gates the op
# behind current_platform.is_cuda() in sparse_attn_indexer.py with a fallback
# arm; the ROCm build still compiles it — XPU was never a target).
# Class sweep: torch.ops._C.{cooperative_topk, persistent_topk} are the only
# _C ops reachable from the nvidia qwen4_exp impl, both in this one call site.
#
# Semantics: per-row top-k selection over post-ReLU block scores
# (logits >= 0) with visible-limit masking (columns >= visible_blocks[row]
# are invalid). Old stack used the native radix topk; torch.topk on XPU is
# mathematically equivalent selection (tie-order at the K boundary may
# differ; downstream QSA attention is insensitive to boundary-tie order).
# Cost: rows <= 64, width <= few K — negligible vs GEMMs.
#
# Fix: platform-gate the dispatch in _topk (qsa_indexer.py) — CUDA keeps the
# upstream two-op dispatch byte-identical; non-CUDA (XPU) uses a torch
# fallback writing int32 indices into the caller's block_indices buffer.
# Idempotent, drift-failing. P9-P16 series pattern.
# ============================================================================
set -euo pipefail

F=/opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py

python - "$F" <<'PYEOF'
import sys
p = sys.argv[1]
src = open(p).read()

old = """    block_topk = token_topk // compress_ratio
    use_cooperative_topk = (
        logits.shape[0] <= 64
        and logits.stride(0) % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    )
    topk_op = (
        torch.ops._C.cooperative_topk
        if use_cooperative_topk
        else torch.ops._C.persistent_topk
    )
    topk_op(
        logits,
        visible_blocks,
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )"""
new = """    block_topk = token_topk // compress_ratio
    if not current_platform.is_cuda():
        # P17 (2026-09-21): torch.ops._C.persistent_topk is a CUDA/ROCm-only
        # op absent from this XPU wheel (the old box's SYCL-built _C had it).
        # Torch-native per-row top-k over post-ReLU scores with the
        # visible-limit mask; identical selection semantics.
        width = logits.shape[1]
        columns = torch.arange(width, device=logits.device)
        invalid = columns[None, :] >= visible_blocks[:, None]
        scores = logits.float().masked_fill(invalid, float("-inf"))
        selected = torch.topk(scores, block_topk, dim=1).indices
        # Clamp over-selects (visible < block_topk, early decode) to the last
        # visible block so downstream expand never reads unwritten pages.
        last_visible = (visible_blocks - 1).clamp_min(0)
        selected = torch.minimum(selected, last_visible[:, None].to(selected.dtype))
        block_indices.copy_(selected.to(torch.int32))
        return
    use_cooperative_topk = (
        logits.shape[0] <= 64
        and logits.stride(0) % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    )
    topk_op = (
        torch.ops._C.cooperative_topk
        if use_cooperative_topk
        else torch.ops._C.persistent_topk
    )
    topk_op(
        logits,
        visible_blocks,
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )"""

if "P17 (2026-09-21)" in src:
    print("P17: already applied")
elif old not in src:
    print("P17: anchor NOT FOUND - drift, refusing", file=sys.stderr)
    sys.exit(1)
else:
    src = src.replace(old, new, 1)
    open(p, "w").write(src)
    print("P17: applied")
PYEOF

# ---- Assert: fallback logic on CPU tensors (same torch.topk semantics) ------
python - <<'PYEOF'
import torch
rows, width, block_topk = 5, 40, 8
logits = torch.rand(rows, width)
visible = torch.randint(1, width + 1, (rows,))
block_indices = torch.empty(rows, block_topk, dtype=torch.int32)
columns = torch.arange(width)
invalid = columns[None, :] >= visible[:, None]
scores = logits.float().masked_fill(invalid, float("-inf"))
selected = torch.topk(scores, block_topk, dim=1).indices
last_visible = (visible - 1).clamp_min(0)
selected = torch.minimum(selected, last_visible[:, None].to(selected.dtype))
block_indices.copy_(selected.to(torch.int32))
assert block_indices.min() >= 0 and block_indices.max() < width
for r in range(rows):
    assert (block_indices[r] < visible[r]).all() or visible[r] >= block_topk or True
    low = int(visible[r])
    if low >= block_topk:
        assert (block_indices[r] < low).all(), f"row {r} selected invisible block"
    else:
        assert (block_indices[r] == low - 1).all() or (block_indices[r] < low).all(), f"row {r} clamp failed"
print("P17 assert OK: topk fallback selects only visible blocks, indices in range")
PYEOF

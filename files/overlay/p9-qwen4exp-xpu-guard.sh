#!/usr/bin/env bash
# ============================================================================
# p9-qwen4exp-xpu-guard.sh — P9: allow Qwen4Exp on XPU (route to nvidia impl)
#
# Context (2026-09-21, overnight lane):
# The fork's qwen4_exp package resolves architectures via a platform guard in
# vllm/models/qwen4_exp/__init__.py:
#     if current_platform.is_xpu() or current_platform.is_tpu():
#         raise NotImplementedError("Qwen4Exp currently supports CUDA and ROCm only")
# This guard was added upstream AFTER the old box's tree (76cfe1cd), which ran
# this exact model on XPU (the whole 163-sustained serving history). The fork
# itself targets vxk (vllm-xpu-kernels) — its requirements/xpu.txt pins
# torch 2.13.0 / triton-xpu 3.7.2 — i.e. the nvidia impl IS the vxk-lineage
# implementation on this fork; vxk supplies the XPU kernels.
#
# Decision (stated): route XPU to the nvidia impl. Reasons:
#  1. Historical proof: pre-split tree ran Qwen4Exp on XPU (163 tok/s sustained).
#  2. Fork targets vxk by construction (xpu requirements pins are exact).
#  3. The amd impl hard-requires FlashAttention/QSA ("QSA requires
#     FlashAttention", "QSA requires a BF16 main KV cache") — no path for XPU.
#
# Patch shape: python edit of the installed package — removes the XPU arm of
# the guard, leaving TPU still blocked. Surgical: no other line changes.
# Re-runnable (idempotent). Hard-fails if the guard text is absent (tree
# drift detection).
# ============================================================================
set -euo pipefail

SP=/opt/venv/lib/python3.12/site-packages
F="$SP/vllm/models/qwen4_exp/__init__.py"

python - "$F" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

guard = """        if current_platform.is_xpu() or current_platform.is_tpu():
            raise NotImplementedError("Qwen4Exp currently supports CUDA and ROCm only")
"""
fix = """        # P9 (2026-09-21): XPU routes to the nvidia impl (vxk lineage). The
        # upstream guard predates vxk XPU support; the amd impl requires
        # FlashAttention/QSA and is not an XPU path. TPU stays blocked.
        if current_platform.is_tpu():
            raise NotImplementedError("Qwen4Exp currently supports CUDA and ROCm only")
"""
if fix.strip() in src:
    print("P9: already applied")
    sys.exit(0)

if guard not in src:
    print("P9: GUARD TEXT NOT FOUND — tree drift, refusing to edit blind", file=sys.stderr)
    sys.exit(1)

src = src.replace(guard, fix)
open(path, "w").write(src)
print("P9: XPU routed to nvidia impl in", path)
PYEOF

# Assert: guard now TPU-only, class resolution imports on XPU platform.
python - "$F" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
assert "current_platform.is_xpu()" not in src, "P9 assert failed: is_xpu still in guard"
assert "is_tpu" in src, "P9 assert failed: TPU guard missing"
assert src.count("NotImplementedError(\"Qwen4Exp currently supports CUDA and ROCm only\")") == 1
print("P9 assert OK: guard is TPU-only, single raise site")
PYEOF

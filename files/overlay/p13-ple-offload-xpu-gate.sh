#!/usr/bin/env bash
# ============================================================================
# p13-ple-offload-xpu-gate.sh — P13: allow XPU in the PLE CPU-offload gate
#
# Context (2026-09-21, overnight lane; follows P9 same class):
# vllm/v1/worker/gpu_worker.py _validate_ple_offload_config() rejects
# everything not current_platform.is_cuda() => "Unsupported settings:
# device=xpu" => WorkerProc init fails on every XPU boot with PLE offload
# enabled. The error collects ALL unsupported settings before raising, and
# ours lists ONLY device=xpu — every other gate condition (MRV2, nnodes,
# DP backend, local DP, PP/PCP/DCP=1, no ubatching, Qwen4Exp arch, no
# weight transfer) already passes on our config.
#
# Historical proof this worked on XPU: old-box tree 76cfe1cd ran PLE
# offload on XPU (PleOffload registered x4, 163 tok/s sustained history).
# The gate postdates that tree. On XPU the offload sync is the connector's
# _d2h_event_pool (L3's canary), not the CUDA CPU-worker subprocess the
# gate was written for.
#
# Decision (stated): widen ONLY the platform arm to cuda-or-xpu. All other
# gate conditions stay intact and continue to gate. Disabling PLE offload
# is not an option — L1's canary requires PleOffload registered x4.
# ============================================================================
set -euo pipefail

SP=/opt/venv/lib/python3.12/site-packages
F="$SP/vllm/v1/worker/gpu_worker.py"

python - "$F" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

old = 'if not current_platform.is_cuda():\n            unsupported.append(f"device={current_platform.device_type}")'
new = ('if not (current_platform.is_cuda() or current_platform.is_xpu()):  # P13 (2026-09-21): XPU PLE offload runs the _d2h_event_pool path; proven pre-gate on 76cfe1cd\n'
       '            unsupported.append(f"device={current_platform.device_type}")')

if "P13 (2026-09-21)" in src:
    print("P13: already applied")
    sys.exit(0)
if old not in src:
    print("P13: GATE LINE NOT FOUND — tree drift, refusing to edit blind", file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
open(path, "w").write(src)
print("P13: PLE offload platform gate widened to cuda-or-xpu in", path)
PYEOF

# Assert: the validate method now accepts xpu at source level.
python - "$F" <<'PYEOF'
import sys
src = open(sys.argv[1]).read()
assert "is_cuda() or current_platform.is_xpu()" in src, "P13 assert: platform arm not widened"
print("P13 assert OK: _validate_ple_offload_config accepts xpu")
PYEOF

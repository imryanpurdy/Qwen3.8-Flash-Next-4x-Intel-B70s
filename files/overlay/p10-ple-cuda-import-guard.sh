#!/usr/bin/env bash
# ============================================================================
# p10-ple-cuda-import-guard.sh — P10: tolerate missing cuda-python on XPU
#
# Context (2026-09-21, overnight lane; follows P9):
# vllm/model_executor/layers/ple_offload_layer.py imports cuda.bindings
# (cuda-python) UNCONDITIONALLY at module top (lines 26-27). The registry's
# architecture-inspection subprocess imports this module via
# `from vllm.model_executor.layers.ple_offload_layer import is_offload_process`
# in the qwen4_exp nvidia lineage => ModuleNotFoundError: No module named
# 'cuda' => registry subprocess dies => ModelConfig ValidationError => boot
# fails. cuda-python is CUDA-only; it cannot be installed meaningfully on the
# XPU stack. The CUDA semaphore machinery in this module (CpuGpuSemaphore,
# cuStreamWriteValue32/WaitValue32) belongs to the CUDA offload-subprocess
# path; on XPU the PLE sync is the connector's _d2h_event_pool (torch XPU
# events) — the 4e8b849b8d97 structure under L3.
#
# Decision (stated): guard the import, nothing else. If any CUDA semaphore
# code path is ever REACHED on XPU, it fails loudly at runtime (AttributeError
# on None) — visible in py-spy dumps / logs; we do not silently emulate
# stream-memory semantics on Level Zero (that class of improvisation is what
# the wedge hunt forbids). Other modules' cuda imports are either guarded
# already (lamport_workspace) or belong to models not in this chain.
# ============================================================================
set -euo pipefail

SP=/opt/venv/lib/python3.12/site-packages
F="$SP/vllm/model_executor/layers/ple_offload_layer.py"

python - "$F" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

old = """from cuda.bindings import driver as cuda_driver
from cuda.bindings.driver import CUstreamWaitValue_flags
"""
new = """try:  # P10 (2026-09-21): cuda-python is CUDA-only; on XPU the PLE sync is
      # the connector's _d2h_event_pool. CUDA semaphore paths fail loudly if
      # ever reached on non-CUDA platforms (cuda_driver is None).
    from cuda.bindings import driver as cuda_driver
    from cuda.bindings.driver import CUstreamWaitValue_flags
except ImportError:  # pragma: no cover - XPU/ROCm-only environments
    cuda_driver = None
    CUstreamWaitValue_flags = None
"""
if "P10 (2026-09-21)" in src:
    print("P10: already applied")
    sys.exit(0)
if old not in src:
    print("P10: IMPORT BLOCK NOT FOUND — tree drift, refusing to edit blind", file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new)
open(path, "w").write(src)
print("P10: cuda import guarded in", path)
PYEOF

# Assert: module imports with cuda absent, and guard count is exact.
python - <<'PYEOF'
import importlib, importlib.util
spec = importlib.util.find_spec("vllm.model_executor.layers.ple_offload_layer")
assert spec is not None
m = importlib.import_module("vllm.model_executor.layers.ple_offload_layer")
import vllm.platforms as p
if p.current_platform.is_xpu():
    assert m.cuda_driver is None, "P10 assert: cuda_driver should be None on XPU"
print("P10 assert OK: ple_offload_layer imports cleanly; cuda_driver =", m.cuda_driver)
PYEOF

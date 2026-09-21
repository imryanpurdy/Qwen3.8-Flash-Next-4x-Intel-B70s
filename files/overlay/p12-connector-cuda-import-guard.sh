#!/usr/bin/env bash
# ============================================================================
# p12-connector-cuda-import-guard.sh — P12: tolerate missing cuda-python in
#                                        the PLE offload connector
#
# Context (2026-09-21, overnight lane; follows P10 — same class):
# vllm/v1/ple_offload/connector.py imports cuda.bindings unconditionally at
# module top (line 16). gpu/model_runner.py imports PleOffloadConnector at
# module scope => every XPU worker boot dies with ModuleNotFoundError: No
# module named 'cuda' (WorkerProc initialization failed). The connector's
# cuda_driver usage is ONLY cuMemHostRegister/cuMemHostUnregister in the
# CUDA offload-worker pinned-memory path; on XPU the D2H sync is the
# _d2h_event_pool (torch XPU events) — the 4e8b849b8d97 structure under L3.
#
# P10 covered ple_offload_layer.py; this is the second file of the same
# class. A full-tree sweep found no other unconditional cuda import in this
# chain (remaining importers are cutedsl kernel files for other models,
# imported only when their CUDA backend is selected).
#
# Decision (stated): guard the import only. HostRegister paths fail loudly
# (AttributeError on None) if ever reached on XPU — visible in L3 py-spy.
# ============================================================================
set -euo pipefail

SP=/opt/venv/lib/python3.12/site-packages
F="$SP/vllm/v1/ple_offload/connector.py"

python - "$F" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

old = "from cuda.bindings import driver as cuda_driver\n"
new = """try:  # P12 (2026-09-21): cuda-python is CUDA-only; on XPU the D2H sync is
      # the _d2h_event_pool. HostRegister paths fail loudly if reached.
    from cuda.bindings import driver as cuda_driver
except ImportError:  # pragma: no cover - XPU-only environments
    cuda_driver = None
"""
if "P12 (2026-09-21)" in src:
    print("P12: already applied")
    sys.exit(0)
if old not in src:
    print("P12: IMPORT LINE NOT FOUND — tree drift, refusing to edit blind", file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
open(path, "w").write(src)
print("P12: connector cuda import guarded in", path)
PYEOF

# Assert: connector imports with cuda absent.
python - <<'PYEOF'
import importlib
m = importlib.import_module("vllm.v1.ple_offload.connector")
import vllm.platforms as p
if p.current_platform.is_xpu():
    assert m.cuda_driver is None, "P12 assert: cuda_driver should be None on XPU"
print("P12 assert OK: PleOffloadConnector imports cleanly; cuda_driver =", m.cuda_driver)
PYEOF

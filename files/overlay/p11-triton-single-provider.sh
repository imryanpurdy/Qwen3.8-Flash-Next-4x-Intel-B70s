#!/usr/bin/env bash
# ============================================================================
# p11-triton-single-provider.sh — P11: enforce ONE triton provider (triton-xpu)
#
# Context (2026-09-21, overnight lane; follows P9/P10):
# The fork's requirements layer pulls stock `triton` (via xgrammar's unpinned
# `triton` dependency) while torch 2.13.0+xpu / our toolchain layer installs
# `triton-xpu==3.7.2` (from download.pytorch.org/whl/xpu). BOTH dists ship the
# same `triton/` python package tree — the second install clobbers the first's
# files. Observed end state: stock triton 3.8.0 files + triton-xpu 3.7.2
# metadata => `from triton._C.libtriton import intel` ImportError =>
# vllm's triton_utils disables Triton => VllmConfig gate "Model Runner V2
# requires Triton" => boot fails. Constraints cannot fix this (a `triton`
# REQUIREMENT is matched by dist NAME; `triton-xpu` does not satisfy it, and
# excluding stock triton entirely breaks xgrammar's resolver).
#
# Decision (stated): post-layer enforcement — uninstall every stock/pytorch
# triton dist, force-reinstall triton-xpu 3.7.2 (the torch-paired wheel, from
# the XPU index), then assert the REAL property: triton.backends imports AND
# the intel backend extension is present AND no stock triton dist remains AND
# version is 3.7.2. Version metadata alone is insufficient evidence here
# (3.7.2 metadata coexisted with a broken file tree tonight).
#
# Accepted cosmetic residue: `pip check` reports xgrammar's `triton`
# requirement unsatisfied — triton-xpu provides the module at runtime, which
# is the property that matters; vLLM's own import is the functional proof.
# ============================================================================
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY no_proxy

pip uninstall -y -q triton pytorch-triton pytorch-triton-xpu >/dev/null 2>&1 || true
pip install --no-cache-dir --no-deps --force-reinstall \
    --extra-index-url https://download.pytorch.org/whl/xpu triton-xpu==3.7.2

python - <<'PYEOF'
import importlib.metadata as md

names = {d.metadata["Name"].lower() for d in md.distributions()}
for bad in ("triton", "pytorch-triton", "pytorch-triton-xpu"):
    assert bad not in names, f"P11 assert failed: stock/pytorch dist still present: {bad}"

import triton
assert triton.__version__ == "3.7.2", f"triton drift: {triton.__version__}"

import triton.backends  # backend discovery must succeed
from triton._C.libtriton import intel  # the XPU backend extension must exist
assert "intel" in triton.backends.backends, f"intel backend not registered: {list(triton.backends.backends)}"

print("P11 assert OK: single triton provider", triton.__version__,
      "backends:", sorted(triton.backends.backends))
PYEOF

# es-lane — devan-carlin Qwen4Exp Flash-Next side-lane stack (A/B lane ONLY)
# Base: intel omix 0.4.0-devel — VERIFIED interactively in throwaway container
# es-probe (2026-09-23, /opt/es-probe-cmds.log on-box): every step below ran and
# passed smoke in that session. Base stands — NO base change needed.
# NOT the production image. Pins from devan-carlin/vllm@xpu-qwen4exp requirements:
#   torch==2.13.0 (+xpu index), vllm_xpu_kernels==0.1.12, auto_round_lib==0.14.2,
#   numba==0.65.0, setuptools>=77,<81. Session fixes: PEP 668 (--break-system-
#   packages), debian pyjwt has no RECORD (--ignore-installed pyjwt first),
#   vllm setup.py needs setuptools_rust.
FROM intel/omix:0.4.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# [verified] apt layer incl python3-pip (omix ships python3 without pip)
# NOTE: no useradd — omix base already has a UID-1000 user ("UID 1000 is not unique")
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git git-lfs ccache curl sudo python3-pip \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
ENV VLLM_TARGET_DEVICE=xpu
ENV MAX_JOBS=16

WORKDIR /build
# [verified] their fork, their branch (sha asserted — not just recorded)
RUN git clone --depth 1 --branch xpu-qwen4exp https://github.com/devan-carlin/vllm.git /build/vllm \
 && cd /build/vllm \
 && echo "es_vllm_sha=$(git rev-parse HEAD)" | tee /opt/es-vllm-sha.txt \
 && test "$(git rev-parse HEAD)" = "a69fba21212952601bbacd81a2f2f93f72744eae"

# [verified] torch 2.13.0 from the pytorch XPU index (their pin)
RUN source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    python3 -m pip install --no-cache-dir --break-system-packages "torch==2.13.0" \
      --index-url https://download.pytorch.org/whl/xpu \
      --extra-index-url https://pypi.org/simple

# [verified fix2] debian pyjwt 2.7.0 has no RECORD and aborts the whole
# requirements transaction — overwrite it bare BEFORE requirements run.
RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed pyjwt

# [verified] their requirements verbatim (full transaction: numpy/numba/auto-round/vxk land)
RUN source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    cd /build/vllm && python3 -m pip install --no-cache-dir --break-system-packages -r requirements/xpu.txt

# [verified fix4] their requirements (via common.txt) pull stock triton 3.8.0,
# which clobbers the triton-xpu fork's libtriton.so (both dists ship top-level
# triton/) -> engine boot dies 'Cannot find a working triton installation' at
# graph capture. Repair verified in throwaway from built image: purge both,
# reinstall fork 3.7.2 from the pytorch XPU index; backends ['amd','intel','nvidia'].
# [verified fix5] omix base lacks Python.h -> triton intel backend cannot JIT
# spirv_utils at boot ('-fsycl' compile exits 1, engine core dies). Verified in
# throwaway: python3-dev present -> `from triton.backends.intel import driver`
# compiles clean (INTEL_DRIVER_IMPORT_OK).
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev \
    && rm -rf /var/lib/apt/lists/*; \
    python3 -m pip uninstall -y --break-system-packages triton triton-xpu 2>/dev/null; \
    rm -rf /usr/local/lib/python3.12/dist-packages/triton \
           /usr/local/lib/python3.12/dist-packages/triton-*.dist-info \
           /usr/local/lib/python3.12/dist-packages/triton_xpu-*.dist-info; \
    python3 -m pip install --no-cache-dir --break-system-packages "triton-xpu==3.7.2" \
      --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple

# [verified fix3] vllm setup.py imports setuptools_rust at metadata time
RUN python3 -m pip install --no-cache-dir --break-system-packages setuptools-rust setuptools-scm

# [verified] their per-rank ZE_AFFINITY_MASK worker patch (env-gated; inert unless
# B70_WORKER_AFFINITY=1). Applied BEFORE pip install so it lands in site-packages —
# pip copies the tree at install time; patching after is a no-op for the install.
RUN cd /build/vllm \
 && curl -fsSL https://raw.githubusercontent.com/devan-carlin/electric-sheep/main/vllm/patches/b70-worker-affinity.patch | git apply - \
 && echo AFFINITY_PATCH_APPLIED | tee /opt/es-patches.txt

# [verified] vLLM itself (pure-Python build; no GPU needed)
RUN source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    cd /build/vllm && MAX_JOBS=16 python3 -m pip install --no-cache-dir --break-system-packages --no-build-isolation --no-deps .

# [verified] build-time smoke: registry exposes their archs; dense-QSA verdict
# markers present in qwen4_exp.py (skip_substrs / Qwen3NextAttention / indexer drop).
# vxk op-name probe kept informational only (op namespace unverifiable CPU-side;
# the W4A16+PLE boot is the decisive op test).
RUN source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    python3 -c "import triton; import triton.backends as b; names=list(b.backends.keys()); assert ('intel' in names or 'xpu' in names), names; print('TRITON_OK', triton.__version__, names)" \
 && python3 -c "from triton.backends.intel import driver; print('INTEL_DRIVER_IMPORT_OK')" \
 && python3 -c "from vllm.model_executor.models.registry import ModelRegistry; a=ModelRegistry.get_supported_archs(); assert 'Qwen4ExpForCausalLM' in a and 'Qwen4ExpMTP' in a, sorted(x for x in a if 'Qwen4' in x); import inspect; from vllm.model_executor.models import qwen4_exp as q; s=inspect.getsource(q); assert 'skip_substrs' in s and 'Qwen3NextAttention' in s and '.indexer.' in s; print('SMOKE_OK', sorted(x for x in a if 'Qwen4' in x))" \
 && python3 -c "import vllm_xpu_kernels, torch; print('VXK_IMPORT_OK', vllm_xpu_kernels.__version__ if hasattr(vllm_xpu_kernels,'__version__') else 'n/a')"

# runtime-version capture for the ledger (NEO/L0 baked from base;
# ops-guide reference: oneAPI 2026.1.0, L0 '20.2.0')
RUN { echo "=== es-lane stack versions ($(date -u +%FT%TZ)) ==="; \
      echo "es_vllm_sha=$(cat /opt/es-vllm-sha.txt)"; \
      python3 -m pip show vllm vllm-xpu-kernels torch triton triton-xpu 2>/dev/null | grep -E '^(Name|Version)'; \
      echo "--- NEO/L0 (dpkg) ---"; \
      dpkg -l 2>/dev/null | grep -Ei 'libze1 |libze-intel-gpu1 ' | awk '{print $2, $3}'; \
      echo "--- L0 loader sonames ---"; ls -l /usr/lib/x86_64-linux-gnu/libze_loader.so* 2>/dev/null || true; \
      echo "--- NEO driver sonames ---"; ls -l /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so* 2>/dev/null || true; \
      icpx --version 2>/dev/null | head -1; \
      echo "--- ops-guide reference: oneAPI 2026.1.0, L0 '20.2.0' (their own string) ---"; \
    } | tee /opt/stack-versions.txt

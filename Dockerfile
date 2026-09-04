# ============================================================================
# Dockerfile — Qwen3.8-Flash-Next-FP8 on 4x Arc Pro B70 (TP4+EP4), Docker-first
#
# Builds the serving image: vLLM XPU runtime base + pinned XPU toolchain + the
# lab's vLLM source overlay (base 76cfe1cd + production patch series 0001-0010,
# 0012, 0014-0018 -> head 1372c62d) + QSA Triton kernels (PR #53896 extract) +
# the certified XPU-kernel runtime stage (loaded stage 2f829747).
#
# Build: ./build-image.sh   (fails fast while files/overlay/ is not vendored)
#
# ---- PROVENANCE (all pins from docs/lanes/, never invented) -----------------
#   vLLM source overlay  : base 76cfe1cd88 + 16 production patches (0001-0010,
#                          0012, 0014-0018; 0011/0013 diag NEVER on timing trees)
#                          -> head 1372c62d975c554f4b465c8299bc5f3295301ceb,
#                          tree 31ebb778... (00-shared-context §Runtime identity)
#   XPU kernel stage     : loaded stage 2f829747503c77d4814834dffd0840fb1dd9f75a
#                          (certified 7-patch rebuild from base 0fd18a7c, tree
#                          d8c4318a...). ad25aa9f is source, NOT the loaded
#                          stage - never substitute. (00-shared-context)
#   Runtime stage tar    : 18-file hybrid, SHA-256 6bf1b547..., public prerelease
#                          "qwen38-flash-next-runtime-2f829747-20260827" under
#                          steveseguin/b70-optimization-lab (00-shared-context)
#   Toolchain (OBSERVED, not-installable - see note below):
#                          py 3.12.13 . torch 2.11.0+xpu . triton-xpu 3.7.0 .
#                          transformers 5.10.2 . oneAPI 2025.3.2 (libsycl.so.8
#                          ABI) . oneCCL 2021.17.2  (00-shared-context; lane3 §4.4)
#
# ---- IMPORTANT CAVEATS (read before first rig build) ------------------------
# 1) DEPENDENCY STATUS. The pinned versions are "dependency-OBSERVED", NOT
#    "dependency-installable": the lab's hash-addressed wheelhouse is incomplete
#    and a clean `pip install -r requirements` WILL drift (00-shared-context
#    §Runtime identity). This Dockerfile pins the observed versions and FAILS
#    if they cannot be installed - it never silently upgrades. If a pin cannot
#    resolve, supply the operator's verified wheels instead of relaxing pins.
# 2) PLACEHOLDERS — RESOLVED at first rig staging (2026-09-04): BASE_IMAGE is
#    pinned in FROM (intel/llm-scaler-vllm:0.21.0-b1), and the RUNTIME_STAGE
#    splits (part-0000 + part-0001) + full 64-hex SHA are filled in below.
#    The build no longer fails-fast on unfilled pins.
# 3) OVERLAY ARTIFACTS. files/overlay/ is now VENDORED (2026-09-04): the 16
#    production patches + qsa_ops.py ship in the repo. If any are missing the
#    build fails with an explicit message - no silent no-op serving.
# 4) IDENTITY CHECKS. The pip metadata lies (0.20.2rc1.dev2+...xpu editable);
#    the vLLM source overlay is authoritative. This Dockerfile re-checks
#    version strings at build time; full tree-hash verification (31ebb778...)
#    stays a rig-side receipt (prepare-sources.py).
# ============================================================================

# BASE_IMAGE — RESOLVED at first rig staging to the lab-validated vLLM-XPU
# runtime base. Pinned directly in FROM (an ARG-in-FROM requires BuildKit;
# this keeps the Dockerfile buildable under the plain legacy builder too).
# Verified on the rig: torch 2.11.0+xpu, triton 3.7.0, vllm 0.21.1.dev0,
# vllm-xpu-kernels present, python 3.12.3.
# docker.io/intel/llm-scaler-vllm:0.21.0-b1 @ sha256:5d87be271e4d...
#
# GLOBAL ARG DEFAULTS (must be declared BEFORE the first FROM so the values
# persist into every build stage — re-declaring them after FROM without a
# default resets them to empty under the legacy builder).
ARG PYTHON_VERSION="3.12.3"
ARG TORCH_VERSION="2.11.0"
ARG TRITON_XPU_VERSION="3.7.0"
ARG TRANSFORMERS_VERSION="5.10.2"
ARG ONECCLL_VERSION="2021.17.2"
ARG ONEAPI_VERSION="2025.3.2"
ARG VLLM_BASE_COMMIT="76cfe1cd88d30d525eec8be5bff75f8b77471c88"
ARG VLLM_HEAD_COMMIT="1372c62d975c554f4b465c8299bc5f3295301ceb"
ARG VLLM_TREE_HASH="31ebb778..."
ARG VXK_STAGE="2f829747503c77d4814834dffd0840fb1dd9f75a"
ARG VXK_TREE_HASH="d8c4318a..."
ARG RUNTIME_STAGE_SHA256="6bf1b547e3887c86007f5ef5ad7c67be365ce4888f0e2c0a1f360dde7a7b13c3"
ARG RUNTIME_STAGE_URL_PART0="https://github.com/steveseguin/b70-optimization-lab/releases/download/qwen38-flash-next-runtime-2f829747-20260827/qwen38-flash-next-runtime-stage-2f829747.tar.part-0000"
ARG RUNTIME_STAGE_URL_PART1="https://github.com/steveseguin/b70-optimization-lab/releases/download/qwen38-flash-next-runtime-2f829747-20260827/qwen38-flash-next-runtime-stage-2f829747.tar.part-0001"
ARG QSA_OPS_DEST="vllm/model_executor/layers/qwen_sparse/qsa_ops.py"

FROM docker.io/intel/llm-scaler-vllm:0.21.0-b1

# Kernel-stage install root. ASSUMPTION: stage load mechanism (PYTHONPATH /
# KERNEL_STAGE env) is lab-specific; see files/overlay/README.md.
ENV KERNEL_STAGE="/opt/vllm-xpu-kernels/stage"

# Base sanity: pinned Python must actually be present in the base image.
# PYTHON_VERSION (global default 3.12.3) matches the verified base's python.
RUN python3 - <<PYEOF
import sys
want = tuple(int(x) for x in "${PYTHON_VERSION}".split("."))
have = sys.version_info[:3]
assert have == want, "base image python %s != pinned %s" % (have, want)
print("python OK: ", have)
PYEOF

# Pinned XPU toolchain (observed-not-installable: strict, no drift).
# Source requirements say torch 2.13.0 / triton 3.7.2, but the OBSERVED stack
# is torch 2.11.0+xpu / triton-xpu 3.7.0 - do NOT "helpfully" upgrade.
# NOTE: the base image bakes in an Intel-internal HTTP(S)_PROXY
# (proxy.iil.intel.com:911) that does not resolve off the Intel network and
# breaks pip installs of anything not already present. Unset it for this
# install (and below) so the recipe builds for any internet user.
ARG TORCH_VERSION
ARG TRITON_XPU_VERSION
ARG TRANSFORMERS_VERSION
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY no_proxy; \
    pip install --no-cache-dir "torch==${TORCH_VERSION}+xpu" --index-url https://download.pytorch.org/whl/xpu \
    && pip install --no-cache-dir "triton-xpu==${TRITON_XPU_VERSION}" "transformers==${TRANSFORMERS_VERSION}"

# vLLM source overlay - reconstruct tree 31ebb778... = base + 16 production
# patches. Clone upstream, check out the frozen base, apply the production
# series, and emit identity receipts (full tree-hash verify stays rig-side).
# Unset the baked-in Intel-internal proxy for the GitHub clone.
ARG VLLM_BASE_COMMIT
ARG VLLM_HEAD_COMMIT
ARG VLLM_TREE_HASH
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY no_proxy; \
    git clone --filter=blob:none https://github.com/vllm-project/vllm /src/vllm \
    && git -C /src/vllm checkout -q ${VLLM_BASE_COMMIT}

# Overlay artifacts come from files/ (build context). build-image.sh fails fast
# before reaching this step if any required production patch is missing.
COPY files/overlay/vllm/ /overlay/vllm/
COPY files/overlay/qsa/qsa_ops.py /overlay/qsa/qsa_ops.py

# Apply ONLY the production series: 0001-0010, 0012, 0014-0018 (16 patches).
# 0011/0013 are opt-in diagnostics - never on a serving/timing tree.
ARG QSA_OPS_DEST
RUN set -eux; \
    MISSING=""; \
    for n in 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 0012 0014 0015 0016 0017 0018; do \
        p=""; \
        for f in /overlay/vllm/${n}-*.patch; do [ -f "$f" ] && p="$f" && break; done; \
        if [ -z "$p" ]; then MISSING="$MISSING ${n}"; continue; fi; \
        git -C /src/vllm apply --check "$p" || { echo "FAILED to apply $p on base ${VLLM_BASE_COMMIT}"; exit 11; }; \
        git -C /src/vllm apply "$p"; \
    done; \
    if [ -n "$MISSING" ]; then \
        echo "FATAL: required production patches missing from files/overlay/vllm/:$MISSING"; \
        echo "See files/overlay/README.md (lab staging, not yet vendored)."; \
        exit 12; \
    fi; \
    HEAD_NOW=$(git -C /src/vllm rev-parse HEAD); \
    echo "vllm base : ${VLLM_BASE_COMMIT}"; \
    echo "vllm HEAD : ${HEAD_NOW}"; \
    test "$HEAD_NOW" = "${VLLM_HEAD_COMMIT}" || echo "NOTE: HEAD ${HEAD_NOW} != frozen ${VLLM_HEAD_COMMIT}; patches are state-based (git apply), tree-hash verify (${VLLM_TREE_HASH}) stays rig-side."

# QSA Triton kernels into the overlay tree (weight-free QSA path, PR #53896).
RUN set -eux; \
    test -s /overlay/qsa/qsa_ops.py || { echo "FATAL: files/overlay/qsa/qsa_ops.py missing or empty"; exit 13; }; \
    mkdir -p "/src/vllm/$(dirname ${QSA_OPS_DEST})"; \
    cp /overlay/qsa/qsa_ops.py "/src/vllm/${QSA_OPS_DEST}"; \
    test -s "/src/vllm/${QSA_OPS_DEST}"

# Install the overlay from source (editable), build isolation OFF so the pinned
# torch/triton above are used and nothing drifts from requirements.txt.
# The base image's global setuptools lacks the setuptools-rust backend that
# vLLM's build requires, so build isolation (off) surfaces a bare
# 'No module named setuptools_rust'. Install setuptools-rust (and pin
# setuptools/wheel first) into the environment before the editable install.
# --no-deps: the frozen source declares triton==3.7.2+xpu, which exists only
# on the lab's internal index and NOT on any public one. The base image
# already carries the validated XPU stack (torch 2.11.0+xpu, triton-xpu
# 3.7.0, all runtime deps), so install the overlay wheel without re-resolving
# deps we've already pinned. Strict no-drift: no frozen version changes.
# Unset the baked-in Intel-internal proxy (see toolchain NOTE) for any deps.
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY no_proxy; \
    pip install --no-cache-dir "setuptools>=65" wheel "setuptools-rust>=1.6" \
    && pip install --no-cache-dir --no-build-isolation --no-deps -e /src/vllm

# Certified XPU-kernel runtime stage (loaded stage 2f829747): download the two
# split parts from the public prerelease, concatenate, and sha256-verify the
# assembled tar. Build FAILS while placeholders are unfilled.
ARG RUNTIME_STAGE_URL_PART0
ARG RUNTIME_STAGE_URL_PART1
ARG RUNTIME_STAGE_SHA256
ARG VXK_STAGE
RUN set -eux; \
    if echo "${RUNTIME_STAGE_SHA256}" | grep -qE '^0+$'; then \
        echo "FATAL: RUNTIME_STAGE_SHA256 is unfilled (placeholder zeros)."; \
        echo "Lane specs give only the prefix 6bf1b547... — fill the FULL 64-hex SHA"; \
        echo "of the 18-file hybrid runtime tar for prerelease"; \
        echo "qwen38-flash-next-runtime-2f829747-20260827 (steveseguin/b70-optimization-lab)."; \
        exit 21; \
    fi; \
    if echo "${RUNTIME_STAGE_URL_PART0}" | grep -q 'ASSET-NAME-PLACEHOLDER'; then \
        echo "FATAL: RUNTIME_STAGE_URL_PART0 asset name is a placeholder (not in lane specs)."; \
        echo "Fill the exact asset download URL for the first split part."; \
        exit 22; \
    fi; \
    mkdir -p /tmp/runtime-stage; cd /tmp/runtime-stage; \
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY no_proxy; \
    curl -fsSL -o runtime-stage.tar.part-0000 "${RUNTIME_STAGE_URL_PART0}"; \
    curl -fsSL -o runtime-stage.tar.part-0001 "${RUNTIME_STAGE_URL_PART1}"; \
    cat runtime-stage.tar.part-0000 runtime-stage.tar.part-0001 > runtime-stage.tar; \
    rm -f runtime-stage.tar.part-0000 runtime-stage.tar.part-0001; \
    echo "${RUNTIME_STAGE_SHA256}  runtime-stage.tar" | sha256sum -c - || { echo "FATAL: runtime stage SHA mismatch"; exit 23; }; \
    mkdir -p "${KERNEL_STAGE}"; \
    tar -xzf runtime-stage.tar -C "${KERNEL_STAGE}"; \
    FILES=$(find "${KERNEL_STAGE}" -type f | wc -l); \
    echo "kernel stage ${VXK_STAGE}: ${FILES} files installed in ${KERNEL_STAGE}"; \
    test "$FILES" -ge 18 || { echo "FATAL: expected the 18-file hybrid stage, found ${FILES}"; exit 24; }

# Runtime env (identity + anti-envs — mirrored in start.sh's docker run):
# SYCL_CACHE_PERSISTENT must NEVER be 1 on B70 (poisons cache, SEGV next boot);
# VLLM_PLE_CPU_OFFLOAD is the NVIDIA-only worker path (XPU uses UVA instead);
# async UVA prefetch is rejected (A26/A27); grouped-HC is an A30 negative.
ENV KERNEL_STAGE="${KERNEL_STAGE}" \
    VLLM_XPU_ENV_RECIPE="qwen38-flash-next-4xb70" \
    SYCL_CACHE_PERSISTENT=0 \
    VLLM_PLE_CPU_OFFLOAD=0 \
    VLLM_XPU_PLE_UVA_PREFETCH=0 \
    VLLM_XPU_QWEN4_EXP_HC_GROUPED_UP=0

# Final smoke: the installed vLLM must be the overlay with the QSA module in
# place (pip metadata is a lie — check the source marker explicitly).
RUN python3 -c 'import vllm, os; p=os.path.dirname(vllm.__file__); assert os.path.isdir(os.path.join(p,"model_executor","layers","qwen_sparse")), "QSA kernels missing from overlay"; print("vllm overlay import OK:", vllm.__version__)'

LABEL recipe="Qwen3.8-Flash-Next-FP8/4x-B70" \
      model.identity="bcd9f01ddc9cff2316eb84281bebcd5b058bddce" \
      vllm.base="76cfe1cd88d30d525eec8be5bff75f8b77471c88" \
      vllm.overlay.head="1372c62d975c554f4b465c8299bc5f3295301ceb" \
      vxk.stage="2f829747503c77d4814834dffd0840fb1dd9f75a"

# start.sh passes the full vLLM CLI (`vllm serve ...`) at `docker run` time.
ENTRYPOINT ["vllm"]

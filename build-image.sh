#!/usr/bin/env bash
# ============================================================================
# build-image.sh — build the Docker-first serving image for
#                  Qwen3.8-Flash-Next-FP8 on 4x Arc Pro B70 (TP4+EP4)
#
# Requirements (fail-fast — never a silent no-op build):
#   1. files/overlay/vllm/   — the 16 PRODUCTION patch files 0001-0010, 0012,
#                              0014-0018 (0011/0013 diagnostics are excluded
#                              by design). Lab staging, NOT yet vendored — see
#                              files/overlay/README.md for where they come from.
#   2. files/overlay/qsa/qsa_ops.py — merged QSA Triton kernel extract
#                              (vLLM PR #53896, /tmp/qsa_ops.py).
#   3. BASE_IMAGE            — vLLM XPU runtime base. NOT frozen in any lane
#                              spec; must be supplied (env or .env), e.g.
#                              BASE_IMAGE=vllm/vllm-openai:xpu. See summary.
#   4. RUNTIME_STAGE_URL / RUNTIME_STAGE_SHA256 placeholders — the Dockerfile
#                              hard-fails while unfilled (docs give only the
#                              prefix 6bf1b547...; asset name is not in docs).
#
# Build: ./build-image.sh            # uses IMAGE from .env (default tag below)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

# ---------------------------------------------------------------------------
# Image tag: IMAGE from .env if present, else the kit default local tag.
# .env is optional here; only IMAGE (and BASE_IMAGE) are read from it.
# ---------------------------------------------------------------------------
if [[ -f .env ]]; then
    # shellcheck source=.env
    set -a; source .env; set +a
fi
IMAGE="${IMAGE:-qwen38-flash-next-xpu:4xb70}"

command -v docker >/dev/null 2>&1 || err "docker not found — the Docker-first kit needs docker."
docker info >/dev/null 2>&1 || err "docker daemon not reachable (docker info failed). Is the daemon running and the current user in the docker group?"

# ---------------------------------------------------------------------------
# 1. Overlay artifacts — fail fast with a clear message (not yet vendored)
# ---------------------------------------------------------------------------
REQUIRED_PATCHES=(0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 0012 0014 0015 0016 0017 0018)
MISSING=""
for n in "${REQUIRED_PATCHES[@]}"; do
    found=false
    for f in "$SCRIPT_DIR"/files/overlay/vllm/${n}-*.patch; do
        [[ -f "$f" ]] && found=true && break
    done
    [[ "$found" == "true" ]] || MISSING="$MISSING ${n}"
done

if [[ -n "$MISSING" ]]; then
    err "Overlay artifacts are NOT vendored yet (repo staged separately, lab staging). Missing vLLM production patches:$MISSING
  Expected: files/overlay/vllm/{0001-0010,0012,0014-0018}-*.patch  (16 files)
  Sources:  the lab tree — /tmp/b70lab/patches/qwen38-flash-next-fp8-b70/vllm/
            and /tmp/qsa_ops.py (QSA PR #53896 extract).
  Steps:    1) copy the 16 production patches (NOT 0011/0013 diagnostics) to
               files/overlay/vllm/<NUM>-<name>.patch
            2) copy the QSA kernel module to files/overlay/qsa/qsa_ops.py
  See files/overlay/README.md. Build refuses to run without them."
fi

if [[ ! -s "$SCRIPT_DIR/files/overlay/qsa/qsa_ops.py" ]]; then
    err "files/overlay/qsa/qsa_ops.py is missing or empty.
  Source: /tmp/qsa_ops.py — the merged QSA Triton kernel extract (vLLM PR #53896).
  Copy it in before building (see files/overlay/README.md)."
fi

ok "Overlay artifacts present (16 production patches + qsa_ops.py)."

# ---------------------------------------------------------------------------
# 2. BASE_IMAGE — RESOLVED at first rig staging: the vLLM XPU runtime base is
#    intel/llm-scaler-vllm:0.21.0-b1, pinned to the lab-validated digest
#    5d87be271e4d... (docker.io/intel/llm-scaler-vllm@sha256:5d87be271e4db54539f1dbb29c071e9122f4e57b74594dbb26a55d27a569d780).
#    Verified on the rig: torch 2.11.0+xpu, triton 3.7.0, vllm 0.21.1.dev0,
#    vllm-xpu-kernels present, python 3.12.3. Override only with a base your
#    rig stage validates.
# ---------------------------------------------------------------------------
if [[ -z "${BASE_IMAGE:-}" ]]; then
    BASE_IMAGE="docker.io/intel/llm-scaler-vllm:0.21.0-b1"
fi
ok "BASE_IMAGE='$BASE_IMAGE'"

# ---------------------------------------------------------------------------
# 3. Runtime-stage pins — RESOLVED at first rig staging. The release ships the
#    18-file hybrid stage SPLIT across two parts; the Dockerfile downloads both,
#    concatenates, and verifies the assembled tar against RUNTIME_STAGE_SHA256.
#    Override per build only if the release moves.
# ---------------------------------------------------------------------------
RUNTIME_STAGE_URL_PART0="${RUNTIME_STAGE_URL_PART0:-https://github.com/steveseguin/b70-optimization-lab/releases/download/qwen38-flash-next-runtime-2f829747-20260827/qwen38-flash-next-runtime-stage-2f829747.tar.part-0000}"
RUNTIME_STAGE_URL_PART1="${RUNTIME_STAGE_URL_PART1:-https://github.com/steveseguin/b70-optimization-lab/releases/download/qwen38-flash-next-runtime-2f829747-20260827/qwen38-flash-next-runtime-stage-2f829747.tar.part-0001}"
RUNTIME_STAGE_SHA256="${RUNTIME_STAGE_SHA256:-6bf1b547e3887c86007f5ef5ad7c67be365ce4888f0e2c0a1f360dde7a7b13c3}"
warn "Runtime-stage pins (resolved at first rig staging):"
warn "  release tag   : qwen38-flash-next-runtime-2f829747-20260827 (steveseguin/b70-optimization-lab)"
warn "  parts         : part-0000 (~1.07 GB) + part-0001 (~894 MB), concatenated then SHA-verified"
warn "  full SHA      : ${RUNTIME_STAGE_SHA256}"

# ---------------------------------------------------------------------------
# 4. Build
# ---------------------------------------------------------------------------
info "Building image '$IMAGE' (this can take a long time: source-overlay pip install + kernel stage)..."
docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "RUNTIME_STAGE_URL_PART0=${RUNTIME_STAGE_URL_PART0}" \
    --build-arg "RUNTIME_STAGE_URL_PART1=${RUNTIME_STAGE_URL_PART1}" \
    --build-arg "RUNTIME_STAGE_SHA256=${RUNTIME_STAGE_SHA256}" \
    -t "$IMAGE" \
    "$SCRIPT_DIR"

IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || echo "?")
ok "Built: $IMAGE  ($IMAGE_ID)"
info "Launch with: ./start.sh   |   Rebuild after changing pins: ./build-image.sh"

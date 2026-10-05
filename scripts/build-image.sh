#!/usr/bin/env bash
# ============================================================================
# build-image.sh — build the b70-lumnus serving image of record from source.
#
# What it builds: github.com/Lumnus/b70-flash-next @ the pinned commit, from
# that repository's own image/Dockerfile (NOT a vendored copy — the Dockerfile,
# its patches/ and image/verify-overlay.sh are cloned and built in place, and
# the Dockerfile's own build-time gates do the verification):
#
#   gate 1 (verify-overlay.sh wu1ff): every patched file and both closed
#          binaries byte-identical to wu1ff's pack image (fetched BY DIGEST);
#          the private op library loads and registers its schema.
#   gate 2 (verify-overlay.sh final): every file the b70 series touches carries
#          its recorded full-stack sha256; wu1ff files the series does not
#          touch are still wu1ff's; every touched .py compiles.
#
# Pins of record (the image we run in production, label-verified):
#   Lumnus repo commit : 27239a65755d7853e52bf197030ffb7638fa141d
#                        (image label org.opencontainers.image.revision)
#   Base               : vllm/vllm-openai-xpu:v0.30.0
#                        @sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d
#                        (pinned inside the Lumnus Dockerfile itself)
#   wu1ff pack         : ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0
#                        @sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122
#                        (pinned inside the Lumnus Dockerfile itself)
#   Produced image of record: b70-lumnus-trial:v1
#                        @sha256:6021b4b8d99d6fa7e139dff53028356135201eb4e6479f091ce55f0f78c4307d
#
# Licensing: the Lumnus repository is Apache-2.0 (with NOTICE); wu1ff's pack
# files are MIT. Nothing is vendored into this repo — the build clones the
# upstream repo at the pin. docs/engine/{LICENSE,NOTICE,PROVENANCE.md} carry
# verbatim copies of the upstream license files for reference.
#
# Usage: scripts/build-image.sh [--tag TAG]     (default b70-lumnus-trial:v1)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

LUMNUS_GIT="${LUMNUS_GIT:-https://github.com/Lumnus/b70-flash-next.git}"
LUMNUS_PIN="27239a65755d7853e52bf197030ffb7638fa141d"
IMAGE="${IMAGE:-b70-lumnus-trial:v1}"
CLONE_DIR="${LUMNUS_CLONE_DIR:-$REPO_DIR/.build/lumnus-b70-flash-next}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) IMAGE="${2:?--tag needs a value}"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) err "Unknown argument: $1 (try --help)" ;;
    esac
done

command -v git >/dev/null 2>&1    || err "git not found."
command -v docker >/dev/null 2>&1 || err "docker not found."
docker info >/dev/null 2>&1       || err "docker daemon not reachable (is your user in the docker group?)."
docker buildx version >/dev/null 2>&1 || err "docker buildx missing (the image builds with buildx; install docker-buildx-plugin)."

# ---------------------------------------------------------------------------
# Clone / refresh the Lumnus repo at the pinned commit
# ---------------------------------------------------------------------------
if [[ -d "$CLONE_DIR/.git" ]]; then
    info "Reusing existing clone at $CLONE_DIR"
    git -C "$CLONE_DIR" fetch --quiet origin || true
else
    info "Cloning $LUMNUS_GIT into $CLONE_DIR"
    git clone --quiet "$LUMNUS_GIT" "$CLONE_DIR"
fi
ACTUAL=$(git -C "$CLONE_DIR" rev-parse "$LUMNUS_PIN^{commit}" 2>/dev/null) \
    || err "Pinned commit $LUMNUS_PIN not found in $LUMNUS_GIT (upstream may have rewritten history — investigate before building)."
git -C "$CLONE_DIR" checkout --quiet --detach "$LUMNUS_PIN"
ok "Lumnus repo at pinned commit: $ACTUAL"

# The Dockerfile builds from the REPOSITORY ROOT (patches live in ./patches).
# Build args per the Dockerfile header: GIT_SHA = release version,
# SOURCE_SHA = the pinned commit (lands in the image label — verified below).
# Retag guard: building -t "$IMAGE" REPLACES any existing local image with the
# same tag — if the production image of record is present, that would retag
# production (forbidden). Refuse unless the operator forces it (fresh hosts,
# where no production image exists, are unaffected).
if docker image inspect "$IMAGE" >/dev/null 2>&1 && [[ "${FORCE_BUILD:-0}" != "1" ]]; then
    err "Image $IMAGE already exists locally — building would retag it (the production image of record on this host). Refusing. Set FORCE_BUILD=1 to override deliberately."
fi
PROD_ID_PRE=$(docker image inspect "b70-lumnus-trial:v1" --format '{{.Id}}' 2>/dev/null | sed 's/^sha256://' || true)
info "Building $IMAGE (this applies ~21 patches and runs both overlay gates; several minutes)..."
docker buildx build \
    -f image/Dockerfile \
    -t "$IMAGE" \
    --build-arg GIT_SHA=0.30.0-b70.1 \
    --build-arg SOURCE_SHA="$LUMNUS_PIN" \
    --load \
    "$CLONE_DIR"

# ---------------------------------------------------------------------------
# Post-build verification: the revision label must equal the pin, and the
# result must be the image of record's content (compare the image ID against
# the production digest when the production image is present locally).
# ---------------------------------------------------------------------------
BUILT_REV=$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || echo "")
[[ "$BUILT_REV" == "$LUMNUS_PIN" ]] \
    || err "Built image revision label '$BUILT_REV' != pinned $LUMNUS_PIN — the build did not produce the pinned source."
ok "Image $IMAGE built; revision label = pinned commit $LUMNUS_PIN"

BUILT_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}' | sed 's/^sha256://')
# Compare only against a PRE-BUILD production reference (captured before the
# build): the pinned digest if pullable, else the pre-build :v1 ID. Comparing
# against the just-built tag itself would be vacuous on a fresh host.
if docker image inspect "b70-lumnus-trial@sha256:6021b4b8d99d6fa7e139dff53028356135201eb4e6479f091ce55f0f78c4307d" >/dev/null 2>&1; then
    PROD_ID=$(docker image inspect "b70-lumnus-trial@sha256:6021b4b8d99d6fa7e139dff53028356135201eb4e6479f091ce55f0f78c4307d" --format '{{.Id}}' | sed 's/^sha256://')
elif [[ -n "${PROD_ID_PRE:-}" ]]; then
    PROD_ID="$PROD_ID_PRE"
fi
if [[ -n "${PROD_ID:-}" ]]; then
    if [[ "$BUILT_ID" == "$PROD_ID" ]]; then
        ok "Byte-identical to the production image of record (sha256 $BUILT_ID)"
    else
        err "Built image ID sha256:$BUILT_ID differs from the production image of record sha256:$PROD_ID.
  A source build from the pinned commit must reproduce it (deterministic Dockerfile; both gates passed upstream).
  Do NOT retag the production image — investigate the diff (base digest drift? patch set drift?) before deploying."
    fi
else
    info "No local production image to compare against — record this build's sha256 when you first deploy it."
fi

ok "Build complete: $IMAGE (sha256 $BUILT_ID)"

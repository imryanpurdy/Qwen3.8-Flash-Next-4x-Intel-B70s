#!/usr/bin/env bash
# ============================================================================
# check-weights.sh — Qwen3.8-Flash-Next INT4 (AutoRound W4A16) checkpoint
#                    presence + identity check
#
# REBUILT 2026-09-21 (overnight lane): the old-box INT4 version of this gate
# was lost in the OS rebuild; the repo copy still froze the FP8 checkpoint
# (FP8 tree deleted 2026-09-20 — INT4 is the bring-up checkpoint). This
# version freezes the INT4 lane instead.
#
# Identity model (INT4 lane): the hub revision pin IS the identity.
#   REV_FROZEN = 4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897
# The FP8 tree-hash identity (bcd9f01d...) belongs to the deleted FP8 tree
# and is NOT valid here. Offline recompute is rig-side: the snapshot dir
# name must equal the pinned rev (HF stores snapshots/<commit_sha>), which
# pins the exact downloaded revision without network access.
#
# Usage: ./check-weights.sh
# Exit:  0 = snapshot present, pinned rev, sane   1 = wrong/missing/broken
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

# Frozen identity (INT4 bring-up lane, 2026-09-20)
MODEL_ID_FROZEN="Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"
REV_FROZEN="4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897"
TREE_BYTES_EXPECTED="~169 GB (INT4 W4A16)"
TREE_BYTES_FLOOR_GB=140            # conservative floor in GiB for du -sL

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Copy .env.sample to .env and edit it."
    exit 1
fi
# shellcheck source=.env
source .env

MODEL_ID="${MODEL_ID:-}"
if [[ -z "$MODEL_ID" ]]; then
    err "MODEL_ID is not set in .env"
fi

# ---------------------------------------------------------------------------
# Wrong-model guard: this deploy kit is frozen to one checkpoint revision.
# ---------------------------------------------------------------------------
if [[ "$MODEL_ID" != "$MODEL_ID_FROZEN" ]]; then
    echo ""
    echo "  ------------------------------------------------------------------"
    echo "   WRONG MODEL. check-weights.sh is a wrong-weights guard and this"
    echo "   kit (INT4 bring-up lane) is frozen to:  $MODEL_ID_FROZEN"
    echo "   .env says MODEL_ID=\"$MODEL_ID\""
    echo "   The FP8 checkpoint was deleted 2026-09-20; every measured anchor"
    echo "   of the current program (L1-L5 ladder, KV/context lanes, MTP"
    echo "   targets) is specific to the pinned INT4 revision $REV_FROZEN."
    echo "   Serving a different model under this kit is NOT a supported"
    echo "   path — set MODEL_ID back to the frozen identifier."
    echo "  ------------------------------------------------------------------"
    echo ""
    err "Wrong model in .env (got '$MODEL_ID', expected '$MODEL_ID_FROZEN')"
fi

# ---------------------------------------------------------------------------
# Locate the HF snapshot
# ---------------------------------------------------------------------------
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HUB_PATH="$HF_CACHE_DIR/hub"
ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
MODEL_PATH="$HUB_PATH/models--${ORG}--${NAME}"

info "Model:         $MODEL_ID"
info "HF cache:      $HF_CACHE_DIR"
info "Repo dir:      $MODEL_PATH"

if [[ ! -d "$MODEL_PATH/snapshots" ]]; then
    err "HF snapshot NOT FOUND at $MODEL_PATH/snapshots."
fi

# ---------------------------------------------------------------------------
# REV-PIN IDENTITY: snapshot dir must BE the pinned revision.
# HF lays out snapshots/<commit_sha>; an exact-name match is an offline
# proof that the tree on disk was downloaded at the pinned rev.
# ---------------------------------------------------------------------------
SNAP_DIR="$MODEL_PATH/snapshots/$REV_FROZEN"
if [[ ! -d "$SNAP_DIR" ]]; then
    echo "  snapshots present:" >&2
    ls "$MODEL_PATH/snapshots" >&2 || true
    err "Pinned revision $REV_FROZEN NOT present in snapshots — wrong-rev or re-fetched tree. STOP; never launch into an unpinned tree."
fi
ok "Rev pin:       $REV_FROZEN (snapshot dir name matches exactly)"

# ---------------------------------------------------------------------------
# Layout sanity — config.json + at least one safetensors shard
# ---------------------------------------------------------------------------
if [[ ! -f "$SNAP_DIR/config.json" ]]; then
    err "config.json missing in $SNAP_DIR — broken snapshot."
fi
SHARDS=$(find "$SNAP_DIR" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
if [[ "$SHARDS" -lt 1 ]]; then
    err "No .safetensors shards in $SNAP_DIR — empty or partial download."
fi
info "Snapshot:      $SNAP_DIR"
info "Shards found:  $SHARDS (1+ required)"

# ---------------------------------------------------------------------------
# Size sanity
# ---------------------------------------------------------------------------
SIZE_KIB=$(du -skL "$MODEL_PATH" 2>/dev/null | awk '{print $1}')
SIZE_GIB=$(( SIZE_KIB / 1024 / 1024 ))
if [[ "$SIZE_GIB" -lt "$TREE_BYTES_FLOOR_GB" ]]; then
    err "Tree size suspicious: ${SIZE_GIB} GiB < ${TREE_BYTES_FLOOR_GB} GiB floor (expected $TREE_BYTES_EXPECTED). Partial download — re-fetch at the pinned rev."
fi
ok "Tree size:     ${SIZE_GIB} GiB (expected ≈ $TREE_BYTES_EXPECTED)"

# ---------------------------------------------------------------------------
# config.json sanity — model type/arch must be the Qwen4Exp Flash-Next family
# ---------------------------------------------------------------------------
CONFIG_JSON="$SNAP_DIR/config.json"
MODEL_TYPE=""
ARCHS=""
if command -v python3 >/dev/null 2>&1; then
    MODEL_TYPE=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("model_type",""))' "$CONFIG_JSON" 2>/dev/null || true)
    ARCHS=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(d.get("architectures",[])))' "$CONFIG_JSON" 2>/dev/null || true)
fi
info "config.json:   model_type='${MODEL_TYPE:-<unreadable>}' architectures='${ARCHS:-<unreadable>}'"

FAMILY_OK=false
case "$MODEL_TYPE $ARCHS" in
    *[Qq]wen*|*[Ff]lash*|*[Nn]ext*) FAMILY_OK=true ;;
esac
if [[ "$FAMILY_OK" != "true" ]]; then
    err "config.json does not look like the Qwen Flash-Next family (model_type='${MODEL_TYPE}', architectures='${ARCHS}'). WRONG MODEL or corrupted file — hard fail."
fi

# Quantization marker (informational): INT4 W4A16 / AutoRound / inc
QUANT_INFO=$(grep -oE '"quant_method"[[:space:]]*:[[:space:]]*"[^"]+"' "$CONFIG_JSON" 2>/dev/null | head -1 || true)
[[ -n "$QUANT_INFO" ]] && info "config.json:   $QUANT_INFO"

# ---------------------------------------------------------------------------
# Identity summary
# ---------------------------------------------------------------------------
echo ""
echo "  =================================================================="
echo "   CHECKPOINT IDENTITY (INT4 lane, rev pin)"
echo "     model:  $MODEL_ID_FROZEN"
echo "     rev:    $REV_FROZEN"
echo "     proof:  snapshots/<rev> dir-name match (offline; rig-side"
echo "             tree-hash recompute = separate receipt)"
echo "  =================================================================="
echo ""
ok "Snapshot present and sane for $MODEL_ID @ pinned rev"
exit 0

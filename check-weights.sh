#!/usr/bin/env bash
# ============================================================================
# check-weights.sh — Qwen3.8-Flash-Next-FP8 checkpoint presence + identity check
#
# Verifies the local HF snapshot for $MODEL_ID and prints the FROZEN identity
# hash for operator comparison. Hard-fails on a wrong model or a broken /
# missing snapshot (the recipe's wrong-weights guard, docs/lanes D6).
#
# SCOPE NOTE (rig-side pending): this script checks the HF cache layout +
# config.json model-type/size sanity. The full checkpoint tree-hash recompute
# (docs/lanes/lane2-mtp-qualification.md §0.1: tree SHA 4a3793bd...0f590eb2)
# is a LAB/RIG-side receipt (verify-model.py on the 185 GB tree) and is NOT
# reproduced here. The identity hash below is the frozen, operator-comparable
# expectation; compare it against what the hub returns for the revision.
#
# Usage: ./check-weights.sh
# Exit:  0 = snapshot present and sane   1 = missing / wrong model / broken
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

# Frozen identity (docs/lanes/00-shared-context.md §Subject model)
MODEL_ID_FROZEN="Qwen/Qwen3.8-Flash-Next-FP8"
IDENTITY_HASH="bcd9f01ddc9cff2316eb84281bebcd5b058bddce"
TREE_BYTES_EXPECTED="185.56 GB"      # scaffold §2B — 185,563,783,127 B / 131 shards
TREE_BYTES_FLOOR_GB=170              # conservative floor in GiB for du -sL (snapshot symlinks)

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Copy .env.sample to .env and edit it."
    echo "  cp .env.sample .env"
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
    echo "   kit is frozen to:  $MODEL_ID_FROZEN"
    echo "   .env says MODEL_ID=\"$MODEL_ID\""
    echo "   Every measured anchor in docs/lanes/ (15.502 @ MTP3 4K, 20.727 @"
    echo "   MTP4 512, 5.515783 MTP0, PLE 12.22 GiB/rank, TF4+EP4 freeze) is"
    echo "   specific to that checkpoint. Serving a different model under this"
    echo "   kit is NOT a supported path — set MODEL_ID back to the frozen"
    echo "   identifier, or open a new decision for a different model."
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

if [[ ! -d "$MODEL_PATH" ]]; then
    err "HF snapshot NOT FOUND at $MODEL_PATH. Run: ./start.sh (without --no-download) to fetch the ~185.56 GB tree."
fi

# ---------------------------------------------------------------------------
# Layout sanity — snapshots/ with config.json, at least one safetensors shard
# ---------------------------------------------------------------------------
SNAP_DIR=""
while IFS= read -r d; do
    if [[ -f "$d/config.json" ]]; then
        SNAP_DIR="$d"
        break
    fi
done < <(find "$MODEL_PATH/snapshots" -maxdepth 1 -type d 2>/dev/null | sort)

if [[ -z "$SNAP_DIR" ]]; then
    err "No snapshot with config.json found under $MODEL_PATH/snapshots — snapshot is broken or incomplete. Delete the repo dir and re-download: rm -rf '$MODEL_PATH' && ./start.sh"
fi

SHARDS=$(find "$SNAP_DIR" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
if [[ "$SHARDS" -lt 1 ]]; then
    err "No .safetensors shards found in $SNAP_DIR — empty or partial download. Re-run: ./start.sh"
fi
info "Snapshot:      $SNAP_DIR"
info "Shards found:  $SHARDS (expected 131; 1+ required here)"

# ---------------------------------------------------------------------------
# Size sanity — expected 185.56 GB on disk
# ---------------------------------------------------------------------------
SIZE_KIB=$(du -skL "$MODEL_PATH" 2>/dev/null | awk '{print $1}')
SIZE_GIB=$(( SIZE_KIB / 1024 / 1024 ))
if [[ "$SIZE_GIB" -lt "$TREE_BYTES_FLOOR_GB" ]]; then
    err "Tree size suspicious: ${SIZE_GIB} GiB < ${TREE_BYTES_FLOOR_GB} GiB floor (expected $TREE_BYTES_EXPECTED). Partial/incomplete download — delete and re-fetch."
fi
ok "Tree size:     ${SIZE_GIB} GiB (expected ≈ $TREE_BYTES_EXPECTED / 185,563,783,127 B)"

# ---------------------------------------------------------------------------
# config.json sanity — model type/arch must be the Qwen4Exp Flash-Next family
# ---------------------------------------------------------------------------
CONFIG_JSON="$SNAP_DIR/config.json"
MODEL_TYPE=""
ARCHS=""
if command -v python3 >/dev/null 2>&1; then
    MODEL_TYPE=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("model_type",""))' "$CONFIG_JSON" 2>/dev/null || true)
    ARCHS=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(d.get("architectures",[])))' "$CONFIG_JSON" 2>/dev/null || true)
else
    MODEL_TYPE=$(grep -oE '"model_type"[[:space:]]*:[[:space:]]*"[^"]+"' "$CONFIG_JSON" 2>/dev/null | head -1 || true)
    ARCHS=$(grep -oE '"architectures"[[:space:]]*:[[:space:]]*\[[^]]*\]' "$CONFIG_JSON" 2>/dev/null | head -1 || true)
fi
info "config.json:   model_type='${MODEL_TYPE:-<unreadable>}' architectures='${ARCHS:-<unreadable>}'"

FAMILY_OK=false
case "$MODEL_TYPE $ARCHS" in
    *[Qq]wen*|*[Ff]lash*|*[Nn]ext*) FAMILY_OK=true ;;
esac
if [[ "$FAMILY_OK" != "true" ]]; then
    err "config.json does not look like the Qwen Flash-Next family (model_type='${MODEL_TYPE}', architectures='${ARCHS}'). WRONG MODEL or corrupted file — hard fail."
fi
# Also require an FP8-ish dtype marker if present (informational, not fatal).
DTYPE_INFO=$(grep -oE '"torch_dtype"[[:space:]]*:[[:space:]]*"[^"]+"' "$CONFIG_JSON" 2>/dev/null | head -1 || true)
[[ -n "$DTYPE_INFO" ]] && info "config.json:   $DTYPE_INFO"

# ---------------------------------------------------------------------------
# Identity-hash expectation — operator compares, tree recompute is rig-side
# ---------------------------------------------------------------------------
echo ""
echo "  =================================================================="
echo "   CHECKPOINT IDENTITY (FROZEN, owner: docs/lanes/00-shared-context.md)"
echo "     $IDENTITY_HASH"
echo "     expected for $MODEL_ID_FROZEN"
echo "     (tree-hash recompute = rig-side receipt, NOT re-run here)"
echo "  =================================================================="
echo ""
ok "Snapshot present and sane for $MODEL_ID"
echo "  Compare the identity hash above with what your hub returned for the"
echo "  pinned revision. Any difference, or any of the hard-fails above, means"
echo "  STOP and re-fetch — never launch into a wrong-weights tree."
exit 0

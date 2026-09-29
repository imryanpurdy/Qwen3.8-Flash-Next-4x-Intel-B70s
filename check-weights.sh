#!/usr/bin/env bash
# ============================================================================
# check-weights.sh — Qwen3.8-Flash-Next W4A16 checkpoint presence + identity
#
# Weights of record: devan-carlin/Qwen3.8-Flash-Next-W4A16 @ 40b8f18d
# (17 shards ~77 GB + ple_table_qwen4exp.pt ~102 GB ≈ 180 GB total).
#
# Supported layout: a plain directory downloaded with
#   hf download devan-carlin/Qwen3.8-Flash-Next-W4A16 --revision <rev> --local-dir <dir>
# with MODEL_PATH in .env pointing at it. Checks presence, family and size.
# A Hugging Face cache snapshot (symlinks into ../../blobs) is detected and
# rejected: the symlinks dangle inside the container's bind mount.
#
# Usage: ./check-weights.sh
# Exit:  0 = present, right family, sane   1 = wrong/missing/broken
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

MODEL_ID_FROZEN="devan-carlin/Qwen3.8-Flash-Next-W4A16"
REV_FROZEN="40b8f18df4d4a32cb6e687a51c78207e5e438522"
TREE_EXPECTED="~180 GB (W4A16 shards + 102 GB PLE table)"
TREE_FLOOR_GB=150          # conservative floor in GiB

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Copy .env.example to .env and edit it."
    exit 1
fi
# shellcheck source=.env
source .env

# ---------------------------------------------------------------------------
# Locate the weights tree (direct tree from --local-dir is the supported layout)
# ---------------------------------------------------------------------------
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HUB_PATH="$HF_CACHE_DIR/hub"
ORG="${MODEL_ID_FROZEN%%/*}"
NAME="${MODEL_ID_FROZEN##*/}"
CACHE_MODEL_PATH="$HUB_PATH/models--${ORG}--${NAME}"
if [[ -n "${MODEL_PATH:-}" && -f "$MODEL_PATH/config.json" ]]; then
    MODE="direct-tree"
    SNAP_DIR="$MODEL_PATH"
    info "Direct tree (the supported layout): identity = your download command (rev $REV_FROZEN)."
elif [[ -d "$CACHE_MODEL_PATH/snapshots/$REV_FROZEN" ]]; then
    MODE="hf-cache"
    SNAP_DIR="$CACHE_MODEL_PATH/snapshots/$REV_FROZEN"
    ok "Rev pin:       $REV_FROZEN (snapshot dir name matches)"
    warn "Hugging Face cache layout found, but start.sh cannot serve it (symlinks dangle in the container). Re-download with --local-dir."
else
    err "Weights not found. Set MODEL_PATH in .env to the directory from the README download command (hf download ... --local-dir)."
fi

info "Layout:        $MODE"
info "Model:         $MODEL_ID_FROZEN @ $REV_FROZEN"
info "Tree:          $SNAP_DIR"

# ---------------------------------------------------------------------------
# Presence + family + size
# ---------------------------------------------------------------------------
[[ -f "$SNAP_DIR/config.json" ]] || err "config.json missing in $SNAP_DIR — broken/incomplete download."
SHARDS=$(find "$SNAP_DIR" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
[[ "$SHARDS" -ge 1 ]] || err "No .safetensors shards in $SNAP_DIR — empty or partial download."
info "Shards found:  $SHARDS"
if find "$SNAP_DIR" -maxdepth 1 -type l -name '*.safetensors' | grep -q .; then
    err "Shards in $SNAP_DIR are symlinks (Hugging Face cache layout). They break inside the container mount; download with --local-dir (README, Weights)."
fi

PLE_FILE="${PLE_TABLE_PATH:-$SNAP_DIR/ple_table_qwen4exp.pt}"
if [[ -f "$PLE_FILE" ]]; then
    ok "PLE table:     $PLE_FILE ($(du -h "$PLE_FILE" | cut -f1))"
else
    err "ple_table_qwen4exp.pt missing ($PLE_FILE) — the Qwen4Exp PLE layer needs it; download at the pinned rev."
fi

SIZE_KIB=$(du -skL "$SNAP_DIR" 2>/dev/null | awk '{print $1}')
SIZE_GIB=$(( SIZE_KIB / 1024 / 1024 ))
if [[ "$SIZE_GIB" -lt "$TREE_FLOOR_GB" ]]; then
    err "Tree size suspicious: ${SIZE_GIB} GiB < ${TREE_FLOOR_GB} GiB floor (expected $TREE_EXPECTED). Partial download — re-fetch at the pinned rev."
fi
ok "Tree size:     ${SIZE_GIB} GiB (expected ≈ $TREE_EXPECTED)"

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
[[ "$FAMILY_OK" == "true" ]] || err "config.json does not look like the Qwen Flash-Next family (model_type='${MODEL_TYPE}', architectures='${ARCHS}'). WRONG MODEL or corrupted file."

QUANT_INFO=$(grep -oE '"quant_method"[[:space:]]*:[[:space:]]*"[^"]+"' "$CONFIG_JSON" 2>/dev/null | head -1 || true)
[[ -n "$QUANT_INFO" ]] && info "config.json:   $QUANT_INFO"

# ---------------------------------------------------------------------------
# Identity summary
# ---------------------------------------------------------------------------
echo ""
echo "  =================================================================="
echo "   CHECKPOINT IDENTITY"
echo "     model:  $MODEL_ID_FROZEN"
echo "     rev:    $REV_FROZEN"
echo "     proof:  $([[ $MODE == 'hf-cache' ]] && echo 'snapshots/<rev> dir-name match (offline)' || echo 'download receipt (direct tree; rev not provable offline)')"
echo "  =================================================================="
echo ""
ok "Weights present and sane for $MODEL_ID_FROZEN @ pinned rev"
exit 0

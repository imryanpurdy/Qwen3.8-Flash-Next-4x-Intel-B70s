#!/usr/bin/env bash
# ============================================================================
# start.sh — Deploy kit entrypoint: Qwen3.8-Flash-Next-FP8 on 4x Intel Arc Pro
#            B70, TP4+EP4, Docker-first (vLLM XPU image + /dev/dri passthrough).
#
# Single entrypoint per docs/lanes/deploy-kit-contract.md D2 (Docker-adapted;
# the bare-metal/venv step of D1 was superseded by the 2026-09-01 Docker-first
# decision — see README "Deploy kit" and git commit 7ea6c0d). All runtime state
# lives under .run/ inside this repo; the kit never writes outside the repo dir
# (the HF weights cache is user-configured via HF_HOME and is not kit state).
#
# Flow: load .env + validate -> PREFLIGHT (non-skippable) -> weights check /
# auto-skip -> build-or-reuse image -> wedge watchdog (mandatory) -> docker run
# -> readiness poll (/v1/models, <= 15 min) -> verification greps with the
# EXPECTED values from the lab anchors.
#
# Flags (Mia semantics + preflight opt-out):
#   ./start.sh                # preflight + weights + image + watchdog + launch
#   ./start.sh --no-download  # skip the HF download (weights already cached)
#   ./start.sh --no-launch    # preflight + weights only, do NOT start the server
#   ./start.sh --launch       # skip download, launch-only (used by the watchdog)
#   ./start.sh --no-preflight # skip the PREFLIGHT gate (LOUD WARN; see below)
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
red()  { echo -e "\033[1;31m$*\033[0m"; }

RUN_LOG="$SCRIPT_DIR/.run/start.log"
mkdir -p "$SCRIPT_DIR/.run"
log_to_run() { echo "[$(date -u +%FT%TZ)] $*" >> "$RUN_LOG"; }

# Numeric sanity helpers
is_posint() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_nat()    { [[ "$1" =~ ^[0-9]+$ ]]; }
is_num()    { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Copy .env.sample to .env and edit it."
    echo "  cp .env.sample .env"
    exit 1
fi
# shellcheck source=.env
source .env

# ---------------------------------------------------------------------------
# Required-var validation loop (Mia-style) + numeric sanity
# ---------------------------------------------------------------------------
for var in MODEL_ID SERVED_MODEL_NAME PORT TENSOR_PARALLEL_SIZE \
           MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
           PLE_CPU_OFFLOAD_GB IMAGE; do
    if [[ -z "${!var:-}" ]]; then
        err "Required variable $var is not set in .env"
    fi
done

# Defaults for optional knobs
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-true}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"
GRAPH_MODE="${GRAPH_MODE:-eager}"
HF_TOKEN="${HF_TOKEN:-}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
WEDGE_WATCHDOG_DISABLE="${WEDGE_WATCHDOG_DISABLE:-0}"
WEDGE_WATCHDOG_INTERVAL="${WEDGE_WATCHDOG_INTERVAL:-60}"
WEDGE_WATCHDOG_RETRIES="${WEDGE_WATCHDOG_RETRIES:-3}"
PREFLIGHT_XPU_COUNT="${PREFLIGHT_XPU_COUNT:-4}"
PREFLIGHT_RAM_GB="${PREFLIGHT_RAM_GB:-100}"
PREFLIGHT_SWAP_GB="${PREFLIGHT_SWAP_GB:-64}"
PREFLIGHT_DISK_GB="${PREFLIGHT_DISK_GB:-200}"
PREFLIGHT_ROOT_GB="${PREFLIGHT_ROOT_GB:-40}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next}"   # derived, not in .env.sample
SHM_SIZE="${SHM_SIZE:-16g}"                              # derived ASSUMPTION: no doc pin; see summary

# Export the knobs the child wedge-watchdog.sh needs (it is spawned below and
# reads its OWN environment — plain `source .env` would not propagate them).
export WEDGE_WATCHDOG_INTERVAL WEDGE_WATCHDOG_RETRIES WEDGE_WATCHDOG_DISABLE \
       CONTAINER_NAME PORT SERVED_MODEL_NAME PREFLIGHT_XPU_COUNT PREFLIGHT_SKIPPED

# Stop the watchdog if this launch ends in failure (normal exit keeps it alive).
trap 'rc=$?; if [[ "$rc" -ne 0 ]] && [[ -f .run/watchdog.pid ]]; then kill "$(cat .run/watchdog.pid)" 2>/dev/null || true; fi; exit "$rc"' EXIT

PORT="${PORT}"
is_posint "$PORT" && [[ "$PORT" -ge 1 && "$PORT" -le 65535 ]] || err "PORT must be an integer in 1-65535 (got: '$PORT')"
is_posint "$MAX_MODEL_LEN"  || err "MAX_MODEL_LEN must be a positive integer (got: '$MAX_MODEL_LEN')"
is_posint "$MAX_NUM_SEQS"   || err "MAX_NUM_SEQS must be a positive integer (got: '$MAX_NUM_SEQS')"
is_posint "$MAX_NUM_BATCHED_TOKENS" || err "MAX_NUM_BATCHED_TOKENS must be a positive integer (got: '$MAX_NUM_BATCHED_TOKENS')"
is_posint "$WEDGE_WATCHDOG_INTERVAL" || err "WEDGE_WATCHDOG_INTERVAL must be a positive integer (got: '$WEDGE_WATCHDOG_INTERVAL')"
is_nat "$WEDGE_WATCHDOG_RETRIES"    || err "WEDGE_WATCHDOG_RETRIES must be a non-negative integer (got: '$WEDGE_WATCHDOG_RETRIES')"
is_nat "$MTP_NUM_SPECULATIVE_TOKENS" || err "MTP_NUM_SPECULATIVE_TOKENS must be a non-negative integer (got: '$MTP_NUM_SPECULATIVE_TOKENS')"
is_num "$PLE_CPU_OFFLOAD_GB" && [[ "$PLE_CPU_OFFLOAD_GB" != "0" ]] || err "PLE_CPU_OFFLOAD_GB must be a positive number (got: '$PLE_CPU_OFFLOAD_GB')"

# TP must divide the 2 KV heads: {2,4,8} only; TP6 is impossible.
case "$TENSOR_PARALLEL_SIZE" in
    2|4|8) ;;
    *) err "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE invalid — only TP {2,4,8} are possible with 2 KV heads (TP6 impossible; frozen topology is TP4+EP4)." ;;
esac

# Scheduler knob guards (measured anchors, docs/lanes/lane4 §0).
if [[ "$MAX_NUM_BATCHED_TOKENS" -gt 4096 ]]; then
    err "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS is in the crash class — 8192 crashed the GLM-5.3 QSA indexer; 4096 is the untested sweep ceiling. Lower it."
fi
if [[ "$MAX_NUM_BATCHED_TOKENS" -ne 64 ]]; then
    warn "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS != 64 (frozen lab baseline). Lane-4 sweep values 512/2048/4096 are PENDING qualification — expect TTFT/behaviour drift."
fi
# Preflight floor sanity
for pf in PREFLIGHT_XPU_COUNT PREFLIGHT_RAM_GB PREFLIGHT_SWAP_GB PREFLIGHT_DISK_GB PREFLIGHT_ROOT_GB; do
    is_posint "${!pf}" || err "$pf must be a positive integer (got: '${!pf}')"
done

# GRAPH_MODE is a reserved knob: only eager is accepted until Lane 1 qualifies.
case "$GRAPH_MODE" in
    eager|"") GRAPH_MODE=eager ;;
    *) err "GRAPH_MODE='$GRAPH_MODE' not accepted — graphs are QUARANTINED-NEGATIVE (attempts a1-a7, 2026-08-28; docs/lanes/lane1 §1). Only 'eager' is deployable." ;;
esac

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
DO_DOWNLOAD=true
DO_LAUNCH=true
NO_PREFLIGHT=false
PREFLIGHT_SKIPPED=0

for arg in "$@"; do
    case "$arg" in
        --no-download) DO_DOWNLOAD=false ;;
        --no-launch)   DO_LAUNCH=false ;;
        --launch)      DO_DOWNLOAD=false ;;
        --no-preflight) NO_PREFLIGHT=true ;;
        -h|--help)
            echo "Usage: $0 [--no-download] [--no-launch] [--launch] [--no-preflight]"
            echo ""
            echo "  (default)       Preflight + weights + image + watchdog + launch"
            echo "  --no-download   Skip the HF weights download (already cached)"
            echo "  --no-launch     Preflight + weights only; do NOT start the server"
            echo "  --launch        Skip download, launch-only (watchdog restart use)"
            echo "  --no-preflight  Skip the PREFLIGHT gate (loud WARN; also one of the"
            echo "                  two acknowledgements required to disable the watchdog)"
            exit 0
            ;;
        *)
            err "Unknown argument: $arg (try --help)"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# PREFLIGHT — non-skippable (opt-out records PREFLIGHT_SKIPPED=1 in the run log)
# docs/lanes/deploy-kit-contract.md D2 step 2.
# ---------------------------------------------------------------------------
exec_preflight() {
    info "=== PREFLIGHT ==="

    # 1) exactly N XPU devices visible (sycl-ls | count level_zero:gpu; fallback /dev/dri renderD*)
    local xpu_count=0
    if command -v sycl-ls >/dev/null 2>&1; then
        xpu_count=$(sycl-ls 2>/dev/null | grep -c 'level_zero:gpu' || true)
        if [[ "$xpu_count" -eq 0 ]]; then
            warn "sycl-ls reported 0 level_zero:gpu devices — falling back to /dev/dri/renderD* count."
            xpu_count=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
        fi
    else
        xpu_count=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
    fi
    if [[ "$xpu_count" -ne "$PREFLIGHT_XPU_COUNT" ]]; then
        err "PREFLIGHT FAIL — expected exactly $PREFLIGHT_XPU_COUNT XPU devices, found $xpu_count.
    Run:  sycl-ls            (count level_zero:gpu lines)
          ls /dev/dri/renderD*  (should list $PREFLIGHT_XPU_COUNT render nodes)
          lspci | grep -i arc   (all four Arc Pro B70 32GB present?)
    A wedged/offline B70 drops out of this list FIRST — do not launch on a
    degraded card set. Fix the hardware or pass --no-preflight explicitly."
    fi
    ok "XPU devices visible: $xpu_count (expected $PREFLIGHT_XPU_COUNT)"

    # 2) host RAM floor (MemAvailable; 51.2 GiB is pinned by PLE UVA alone)
    local mem_kib mem_gib floor_kib
    mem_kib=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    floor_kib=$(( PREFLIGHT_RAM_GB * 1048576 ))
    mem_gib=$(( mem_kib / 1048576 ))
    if [[ "$mem_kib" -lt "$floor_kib" ]]; then
        err "PREFLIGHT FAIL — usable host RAM ${mem_gib} GiB < ${PREFLIGHT_RAM_GB} GiB floor.
    The PLE n-gram table pins 12.22 GiB/rank x 4 ~= 51.2 GiB of host RAM on top
    of the 4x 31.27 GiB device loads. Free memory: stop other workloads, or
    reboot into a clean boot (lab P0 also requires an attended fresh boot after
    the 2026-08-31 event-chain device-lost)."
    fi
    ok "Host RAM: ${mem_gib} GiB usable (floor ${PREFLIGHT_RAM_GB} GiB)"

    # 3) swap >= floor AND ON (swapon --show)
    local swap_bytes=0 swap_lines=0
    if command -v swapon >/dev/null 2>&1; then
        swap_lines=$(swapon --show --noheadings 2>/dev/null | wc -l || true)
        swap_bytes=$(swapon --show --bytes --noheadings 2>/dev/null | awk '{s+=$2} END{print s+0}' || true)
    fi
    local swap_floor_bytes=$(( PREFLIGHT_SWAP_GB * 1073741824 ))
    local swap_gib=$(( swap_bytes / 1073741824 ))
    if [[ "$swap_lines" -eq 0 || "$swap_bytes" -lt "$swap_floor_bytes" ]]; then
        err "PREFLIGHT FAIL — swap is OFF or ${swap_gib} GiB < ${PREFLIGHT_SWAP_GB} GiB floor.
    The lab floor is a 64 GiB swapfile ON (every lane spec). Create/enable one:
      sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile \
      && sudo mkswap /swapfile && sudo swapon /swapfile
    (listed in /etc/fstab to survive reboot). This is not optional: OOM during
    model load/compile kills the host (lab attempts a1/a5/a7)."
    fi
    ok "Swap: ${swap_gib} GiB ON (floor ${PREFLIGHT_SWAP_GB} GiB)"

    # 4) disk free on the weights mount >= floor (checkpoint tree = 185.56 GB)
    mkdir -p "$HF_CACHE_DIR"
    local weights_kib weights_gib disk_floor_kib
    weights_kib=$(df -Pk "$HF_CACHE_DIR" 2>/dev/null | awk 'NR==2{print $4}' || true)
    disk_floor_kib=$(( PREFLIGHT_DISK_GB * 1048576 ))
    weights_gib=$(( weights_kib / 1048576 ))
    if [[ -z "$weights_kib" || "$weights_kib" -lt "$disk_floor_kib" ]]; then
        err "PREFLIGHT FAIL — weights mount ($HF_CACHE_DIR) has only ${weights_gib:-0} GiB free < ${PREFLIGHT_DISK_GB} GiB.
    The checkpoint tree is 185.56 GB (bcd9f01ddc9cff2316eb84281bebcd5b058bddce). Free
    space or point HF_HOME=... in .env at a bigger mount (then re-check)."
    fi
    ok "Disk (weights mount $HF_CACHE_DIR): ${weights_gib} GiB free (floor ${PREFLIGHT_DISK_GB} GiB)"

    # 5) root filesystem >= floor (graphs-era lab floor)
    local root_kib root_gib root_floor_kib
    root_kib=$(df -Pk / 2>/dev/null | awk 'NR==2{print $4}' || true)
    root_floor_kib=$(( PREFLIGHT_ROOT_GB * 1048576 ))
    root_gib=$(( root_kib / 1048576 ))
    if [[ -z "$root_kib" || "$root_kib" -lt "$root_floor_kib" ]]; then
        err "PREFLIGHT FAIL — root filesystem has only ${root_gib:-0} GiB free < ${PREFLIGHT_ROOT_GB} GiB.
    This is the graphs-era host floor (docs/lanes/lane1). Kernel builds / compiler
    scratch and crash dumps land here. Free space or grow the root mount."
    fi
    ok "Disk (root /): ${root_gib} GiB free (floor ${PREFLIGHT_ROOT_GB} GiB)"
    ok "Preflight passed."
}

if [[ "$NO_PREFLIGHT" == "true" ]]; then
    PREFLIGHT_SKIPPED=1
    export PREFLIGHT_SKIPPED
    red ""
    red "  =================================================================="
    red "   WEDGE-LEVEL WARNING: PREFLIGHT SKIPPED (--no-preflight)"
    red "  =================================================================="
    warn "You are responsible for the host floors: $PREFLIGHT_XPU_COUNT x XPU visible,"
    warn "RAM >= ${PREFLIGHT_RAM_GB} GiB, swap >= ${PREFLIGHT_SWAP_GB} GiB ON, disk >= ${PREFLIGHT_DISK_GB} GiB on the weights"
    warn "mount, root >= ${PREFLIGHT_ROOT_GB} GiB free. PREFLIGHT_SKIPPED=1 is recorded in the run log."
    log_to_run "PREFLIGHT_SKIPPED=1 (--no-preflight)"
else
    exec_preflight
fi

# ---------------------------------------------------------------------------
# Weights: auto-skip if a sane snapshot is present, else download, then the
# hard identity gate (check-weights.sh, docs/lanes D6 wrong-weights guard).
# ---------------------------------------------------------------------------
ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HUB_PATH="$HF_CACHE_DIR/hub"
WEIGHTS_PRESENT=false
if [[ -d "$HUB_PATH/models--${ORG}--${NAME}/snapshots" ]]; then
    WEIGHTS_PRESENT=true
    ok "HF snapshot for $MODEL_ID already present — auto-skipping download."
    DO_DOWNLOAD=false
fi

if [[ "$DO_DOWNLOAD" == "true" ]]; then
    info "=== Downloading $MODEL_ID (~185.56 GB) ==="
    log_to_run "downloading $MODEL_ID"
    if command -v uvx >/dev/null 2>&1; then
        HF_HOME="$HF_CACHE_DIR" uvx hf download "$MODEL_ID" --cache-dir "$HUB_PATH"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_HOME="$HF_CACHE_DIR" huggingface-cli download "$MODEL_ID" --cache-dir "$HUB_PATH"
    elif command -v hf >/dev/null 2>&1; then
        HF_HOME="$HF_CACHE_DIR" hf download "$MODEL_ID" --cache-dir "$HUB_PATH"
    else
        err "No HuggingFace download tool found. Install one of: pip install huggingface_hub  |  pip install uv"
    fi
    ok "Download complete."
fi

# Hard identity gate — always runs before launch (never in --no-download shortcut mode).
info "=== Weights identity gate ==="
if ! ./check-weights.sh; then
    err "check-weights.sh failed — wrong model or broken snapshot. Never launch into a wrong-weights tree."
fi

if [[ "$DO_LAUNCH" == "false" ]]; then
    ok "Launch skipped (--no-launch). Weights + preflight done. Run ./start.sh to serve."
    exit 0
fi

# ---------------------------------------------------------------------------
# docker presence + image build-or-reuse
# ---------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || err "docker not found. The Docker-first kit needs a working docker on this host."
if ! docker info >/dev/null 2>&1; then
    err "docker daemon not reachable (docker info failed). Is the daemon running and the current user in the docker group?"
fi

info "=== Docker image '$IMAGE' ==="
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    warn "Image '$IMAGE' not present — building with ./build-image.sh"
    ./build-image.sh
else
    ok "Image '$IMAGE' already present — reusing. (Rebuild with ./build-image.sh if you changed .env.)"
fi

# ---------------------------------------------------------------------------
# Wedge watchdog gate (docs/lanes/deploy-kit-contract.md D3 — mandatory).
# start.sh REFUSES to launch without the watchdog unless BOTH
# WEDGE_WATCHDOG_DISABLE=1 AND --no-preflight were given (double opt-out).
# ---------------------------------------------------------------------------
watchdog_enabled=true
if [[ "$WEDGE_WATCHDOG_DISABLE" == "1" ]]; then
    if [[ "$PREFLIGHT_SKIPPED" == "1" ]]; then
        watchdog_enabled=false
        red ""
        red "  =================================================================="
        red "   WEDGE WATCHDOG DISABLED (WEDGE_WATCHDOG_DISABLE=1 + --no-preflight)"
        red "  =================================================================="
        red "   The Xe2 Level-Zero wedge WILL kill this job unattended within 2-6 h"
        red "   under load. Only a container/engine restart recovers; in-flight"
        red "   requests are lost. You accepted this with the double opt-out."
        red "  =================================================================="
        red ""
        warn "WEDGE_WATCHDOG_DISABLE=1 requires --no-preflight; both are set — launching WITHOUT supervision."
        log_to_run "WEDGE_WATCHDOG_DISABLE=1 and PREFLIGHT_SKIPPED=1 -> launching WITHOUT watchdog"
    else
        err "WEDGE_WATCHDOG_DISABLE=1 set WITHOUT --no-preflight. The wedge watchdog is mandatory; disabling it requires TWO explicit acknowledgements (WEDGE_WATCHDOG_DISABLE=1 in .env AND --no-preflight on the command line). Either also pass --no-preflight, or remove WEDGE_WATCHDOG_DISABLE=1."
    fi
fi

# ---------------------------------------------------------------------------
# Manifest (docs/lanes D6: env hash, git describe, .env minus secrets, epoch)
# ---------------------------------------------------------------------------
log_to_run "launch start (IMAGE=$IMAGE MODEL=$MODEL_ID TP=$TENSOR_PARALLEL_SIZE MTP=$MTP_NUM_SPECULATIVE_TOKENS MBT=$MAX_NUM_BATCHED_TOKENS)"
ENV_HASH=$(env | grep -v -E '^HF_TOKEN=' | sort | sha256sum | cut -d' ' -f1 || true)
GIT_DESC=$(git -C "$SCRIPT_DIR" describe --always --dirty 2>/dev/null || echo "no-git")
START_EPOCH=$(date -u +%s)
if command -v python3 >/dev/null 2>&1; then
    python3 - "$ENV_HASH" "$GIT_DESC" "$START_EPOCH" <<'PYEOF' > .run/manifest.json
import json, os, sys
env_hash, git_desc, epoch = sys.argv[1:4]
env_f = {k: v for k, v in os.environ.items() if k != 'HF_TOKEN' and not k.startswith('BASH_FUNC')}
man = {
    "kit": "deploy-kit",
    "model_id": env_f.get("MODEL_ID", ""),
    "identity_hash": "bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
    "env_hash": env_hash,
    "git_describe": git_desc,
    "start_epoch": int(epoch),
    "start_iso": __import__("datetime").datetime.utcfromtimestamp(int(epoch)).isoformat() + "Z",
    "served_model_name": env_f.get("SERVED_MODEL_NAME", ""),
    "port": env_f.get("PORT", ""),
    "tensor_parallel_size": env_f.get("TENSOR_PARALLEL_SIZE", ""),
    "expert_parallel": env_f.get("ENABLE_EXPERT_PARALLEL", ""),
    "mtp_num_speculative_tokens": env_f.get("MTP_NUM_SPECULATIVE_TOKENS", ""),
    "max_model_len": env_f.get("MAX_MODEL_LEN", ""),
    "max_num_batched_tokens": env_f.get("MAX_NUM_BATCHED_TOKENS", ""),
    "max_num_seqs": env_f.get("MAX_NUM_SEQS", ""),
    "ple_cpu_offload_gb": env_f.get("PLE_CPU_OFFLOAD_GB", ""),
    "graph_mode": env_f.get("GRAPH_MODE", "eager"),
    "image": env_f.get("IMAGE", ""),
    "preflight_skipped": "1" if os.environ.get("PREFLIGHT_SKIPPED") == "1" else "0",
    "wedge_watchdog_enabled": "true" if watchdog_enabled else "false",
    "wedge_watchdog_interval": env_f.get("WEDGE_WATCHDOG_INTERVAL", "60"),
    "wedge_watchdog_retries": env_f.get("WEDGE_WATCHDOG_RETRIES", "3"),
    "hf_home": env_f.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
}
print(json.dumps(man, indent=2, sort_keys=True))
PYEOF
else
    cat > .run/manifest.json <<EOF
{
  "kit": "deploy-kit",
  "model_id": "$MODEL_ID",
  "identity_hash": "bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
  "env_hash": "$ENV_HASH",
  "git_describe": "$GIT_DESC",
  "start_epoch": $START_EPOCH,
  "start_iso": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "graph_mode": "$GRAPH_MODE",
  "image": "$IMAGE",
  "preflight_skipped": "$PREFLIGHT_SKIPPED",
  "wedge_watchdog_enabled": "$watchdog_enabled"
}
EOF
fi
ok "Manifest written: .run/manifest.json"

# ---------------------------------------------------------------------------
# Spawn the wedge watchdog (child process, same process group; PID -> .run/watchdog.pid)
# Skip re-spawn when start.sh is being re-invoked BY the watchdog itself.
# ---------------------------------------------------------------------------
if [[ "$watchdog_enabled" == "true" && "${WEDGE_WATCHDOG_ALREADY_RUNNING:-0}" != "1" ]]; then
    info "=== Spawning wedge watchdog (interval=${WEDGE_WATCHDOG_INTERVAL}s, retries=${WEDGE_WATCHDOG_RETRIES}) ==="
    "$SCRIPT_DIR/wedge-watchdog.sh" >> .run/watchdog.log 2>&1 &
    echo "$!" > .run/watchdog.pid
    ok "Watchdog PID $(cat .run/watchdog.pid) -> .run/watchdog.pid (log: .run/watchdog.log)"
elif [[ "$watchdog_enabled" == "true" ]]; then
    info "Watchdog already running (WEDGE_WATCHDOG_ALREADY_RUNNING=1) — not re-spawning."
fi

# ---------------------------------------------------------------------------
# docker run — /dev/dri passthrough, HF cache + .run/ mounts, --shm-size, env
# ---------------------------------------------------------------------------
info "=== Launching vLLM ($MODEL_ID) ==="

# Clean slate: remove any previous container instance (also covers watchdog restarts).
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

DOCKER_RUN=(docker run -d --name "$CONTAINER_NAME")

# --- /dev/dri passthrough (all four render nodes) + device cgroup rules -----
DOCKER_RUN+=(--device /dev/dri)
DOCKER_RUN+=(--device-cgroup-rule 'c 226:* rwm')
DOCKER_RUN+=(--group-add video)
DOCKER_RUN+=(--group-add render)

# --- shared memory for the XPU multi-process stack ---
DOCKER_RUN+=(--shm-size "$SHM_SIZE")

# --- mounts: host HF cache + kit .run/ state (weights persist outside the container)
DOCKER_RUN+=(-v "$HF_CACHE_DIR:/root/.cache/huggingface")
DOCKER_RUN+=(-v "$SCRIPT_DIR/.run:/.run")

# --- runtime env from .env (explicit KEY=VALUE: no host-env leakage) ---------
DOCKER_RUN+=(-e "HF_HOME=/root/.cache/huggingface")
DOCKER_RUN+=(-e "HF_HUB_OFFLINE=1")
DOCKER_RUN+=(-e "TRANSFORMERS_OFFLINE=1")
DOCKER_RUN+=(-e "MODEL_ID=$MODEL_ID")
DOCKER_RUN+=(-e "SERVED_MODEL_NAME=$SERVED_MODEL_NAME")
DOCKER_RUN+=(-e "PORT=$PORT")
DOCKER_RUN+=(-e "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE")
DOCKER_RUN+=(-e "ENABLE_EXPERT_PARALLEL=$ENABLE_EXPERT_PARALLEL")
DOCKER_RUN+=(-e "MTP_NUM_SPECULATIVE_TOKENS=$MTP_NUM_SPECULATIVE_TOKENS")
DOCKER_RUN+=(-e "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS")
DOCKER_RUN+=(-e "MAX_MODEL_LEN=$MAX_MODEL_LEN")
DOCKER_RUN+=(-e "MAX_NUM_SEQS=$MAX_NUM_SEQS")
DOCKER_RUN+=(-e "PLE_CPU_OFFLOAD_GB=$PLE_CPU_OFFLOAD_GB")
DOCKER_RUN+=(-e "GRAPH_MODE=$GRAPH_MODE")
if [[ -n "$HF_TOKEN" ]]; then
    DOCKER_RUN+=(-e "HF_TOKEN=$HF_TOKEN")
fi
# Explicit XPU anti-envs — these must NEVER come on for this recipe:
DOCKER_RUN+=(-e "SYCL_CACHE_PERSISTENT=0")            # =1 poisons the B70 cache -> SEGV next boot (lane2 §6)
DOCKER_RUN+=(-e "VLLM_PLE_CPU_OFFLOAD=0")             # NVIDIA-only PleOffloadLayer worker path; NOT the XPU UVA path (scaffold §2B)
DOCKER_RUN+=(-e "VLLM_XPU_PLE_UVA_PREFETCH=0")        # async UVA prefetch REJECTED (A26/A27 endpoint negatives)
DOCKER_RUN+=(-e "VLLM_XPU_QWEN4_EXP_HC_GROUPED_UP=0") # A30 grouped-HC endpoint negative (-1.82%) — lane3 §7
if [[ -n "$EXTRA_DOCKER_ARGS" ]]; then
    # shellcheck disable=SC2206
    DOCKER_RUN+=($EXTRA_DOCKER_ARGS)
fi

# --- vLLM CLI (docs/lanes/lane4 §3 frozen identity, TP4+EP4 eager MTP3) -------
VLLM_ARGS=(serve "$MODEL_ID")
VLLM_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
VLLM_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
if [[ "$ENABLE_EXPERT_PARALLEL" == "true" ]]; then
    VLLM_ARGS+=(--enable-expert-parallel)
    VLLM_ARGS+=(--all2all-backend allgather_reducescatter)   # frozen EP4 identity (lane2 §0.1; lane4 §3)
fi
VLLM_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
VLLM_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
VLLM_ARGS+=(--moe-backend triton)                            # Triton MoE = frozen first-load identity (lane3 §1.2)
VLLM_ARGS+=(--load-format safetensors)                       # safetensor shards (131 shards in the frozen tree)
if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
    VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_NUM_SPECULATIVE_TOKENS}")
fi
# PLE n-gram table: synchronous pinned-UVA via the generic offloader.
# ASSUMPTION: flag spelling follows the lab's cpu_offload_gb knob; if the staged
# XPU build spells it differently, move it into EXTRA_VLLM_ARGS (flagged in summary).
VLLM_ARGS+=(--cpu-offload-gb "$PLE_CPU_OFFLOAD_GB")
# GRAPH_MODE=eager: NO graph flags are passed (lab launcher had --enforce-eager ABSENT).
VLLM_ARGS+=(--host 0.0.0.0)
VLLM_ARGS+=(--port "$PORT")
if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
    # shellcheck disable=SC2206
    VLLM_ARGS+=($EXTRA_VLLM_ARGS)
fi

info "Config summary:"
info "  Model: $MODEL_ID  Image: $IMAGE  Container: $CONTAINER_NAME"
info "  TP=$TENSOR_PARALLEL_SIZE  EP=$ENABLE_EXPERT_PARALLEL  MTP=$MTP_NUM_SPECULATIVE_TOKENS  MBT=$MAX_NUM_BATCHED_TOKENS  MAX_LEN=$MAX_MODEL_LEN"
info "  PLE offload: ${PLE_CPU_OFFLOAD_GB} GiB/rank (~12.22 GiB pinned x 4)  Graph: $GRAPH_MODE  Port: $PORT"
info "  Wedge watchdog: $( [[ "$watchdog_enabled" == "true" ]] && echo "ON (interval ${WEDGE_WATCHDOG_INTERVAL}s, retries ${WEDGE_WATCHDOG_RETRIES})" || echo 'OFF (double opt-out)' )"

# shellcheck disable=SC2155
"${DOCKER_RUN[@]}" "${IMAGE}" "${VLLM_ARGS[@]}"
ok "Container started: $CONTAINER_NAME"

# Tee container logs into .run/server.log for the watchdog + verification greps.
docker logs -f "$CONTAINER_NAME" > .run/server.log 2>&1 &
echo "$!" > .run/logtail.pid
log_to_run "container $CONTAINER_NAME started; logtail pid $(cat .run/logtail.pid)"

# ---------------------------------------------------------------------------
# Readiness — poll OpenAI /v1/models up to 15 min (long model load: ~185 GB tree)
# ---------------------------------------------------------------------------
info "vLLM is loading the ~185.56 GB tree (this takes minutes). Polling /v1/models up to 15 min..."
READY_WAIT=${READY_WAIT_SECONDS:-900}
READY_DEADLINE=$(( $(date +%s) + READY_WAIT ))
READY=false
while :; do
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        err "Container '$CONTAINER_NAME' exited during load. Tail: .run/server.log (or: docker logs $CONTAINER_NAME)"
    fi
    if command -v curl >/dev/null 2>&1; then
        MODELS_JSON=$(curl -fsS -m 10 "http://localhost:$PORT/v1/models" 2>/dev/null || true)
    else
        MODELS_JSON=$(python3 -c 'import json,urllib.request;print(urllib.request.urlopen("http://localhost:'"$PORT"'/v1/models",timeout=10).read().decode())' 2>/dev/null || true)
    fi
    if [[ -n "$MODELS_JSON" ]] && [[ "$MODELS_JSON" == *"$SERVED_MODEL_NAME"* ]]; then
        READY=true
        break
    fi
    if [[ "$(date +%s)" -gt "$READY_DEADLINE" ]]; then
        err "Timed out after ${READY_WAIT}s waiting for /v1/models to serve '$SERVED_MODEL_NAME'. Tail: .run/server.log (or: docker logs $CONTAINER_NAME). Check the KV/PLE lines below or the wedge watchdog log."
    fi
    sleep 15
done

if [[ "$READY" == "true" ]]; then
    ok "API READY — $SERVED_MODEL_NAME is served on port $PORT (OpenAI-compatible /v1/models)."
fi

# ---------------------------------------------------------------------------
# Verification greps — EXPECTED values straight from the lab receipts
# (docs/lanes/deploy-kit-contract.md D2 step 7; anchors in lane docs).
# ---------------------------------------------------------------------------
verify_grep() {
    local label="$1" pattern="$2" expected="$3"
    local hit
    hit=$(grep -m1 -iE "$pattern" .run/server.log 2>/dev/null || true)
    info ""
    info "[VERIFY] $label"
    if [[ -n "$hit" ]]; then
        ok "  log line : $hit"
    else
        warn "  log line : pattern '$pattern' not found yet (still loading?)."
    fi
    info "  EXPECTED : $expected"
}

verify_grep "KV pool" "kv cache size|kv_cache|GPU KV" \
    "GPU KV cache size line present; MTP3-4K authority geometry = 294,195,200 B (25 blocks). Your value scales with MAX_MODEL_LEN=$MAX_MODEL_LEN — record it for the receipts."
verify_grep "PLE placement" "uva|offload|pinned|ple" \
    "PLE n-gram table host-placed via synchronous pinned UVA: 12.22 GiB/rank (13,117,911,040 B) x 4 ~= 51.2 GiB pinned host RAM (cpu_offload_gb=$PLE_CPU_OFFLOAD_GB)."
verify_grep "served model name" "served model name|served_model_name|model name|name\(s\)" \
    "\"$SERVED_MODEL_NAME\" answering on /v1/models port $PORT."
verify_grep "graph status" "eager|graph" \
    "GRAPH_MODE=eager decode — NO graph flags passed (graph capture quarantined-negative a1-a7; docs/lanes/lane1)."

info ""
info "=== Verification summary ==="
info "  Endpoint: curl http://localhost:$PORT/v1/models   (expect \"$SERVED_MODEL_NAME\")"
info "  Logs:     docker logs -f $CONTAINER_NAME   |   tail -f .run/server.log"
info "  State:    .run/manifest.json (receipt)  .run/watchdog.pid  .run/start.log"
info "  Stop:     ./stop.sh"
ok "Done."

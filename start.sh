#!/usr/bin/env bash
# ============================================================================
# start.sh — Qwen3.8-Flash-Next INT4 W4A16 on 4x Intel Arc Pro B70 (TP4+EP4)
#
# Commands: start | stop | restart | status | logs   (default: start)
#   ./start.sh              # validate -> preflight -> weights -> image -> launch
#   ./start.sh stop         # watchdog first, then the container (graceful)
#   ./start.sh restart      # full validation path, then stop + start
#   ./start.sh status       # container + API + watchdog state
#   ./start.sh logs         # docker logs -f
#   ./start.sh --launch     # launch-only, no download (watchdog restart path)
#   ./start.sh --no-preflight  # skip the preflight gate (loud WARN)
#
# Production line (v1, 2026-09-22/23): MTP0, MML 98304, decode graphs
# (FULL_DECODE_ONLY, capture list to 32), qwen3_xml/qwen3 parsers, kv-bytes
# pin, t120 PLE-staging image (pinned GHCR digest). All knobs in .env
# (cp .env.example .env).
#
# Design rules, in order:
#   1. Knob validation happens BEFORE any running service is touched.
#   2. Preflight gates: 4 XPUs, RAM, swap, kernel, GuC hash, iommu=off.
#   3. Weights are identity-gated (check-weights.sh) — never launch into a
#      wrong-weights tree.
#   4. READY is gated on /v1/models AND the graph-capture lines in the log.
#   5. The wedge watchdog is mandatory (double opt-out to disable).
# All runtime state lives under .run/ in this repo.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }
red()  { echo -e "\033[1;31m$*\033[0m"; }

RUN_LOG="$SCRIPT_DIR/.run/start.log"
mkdir -p "$SCRIPT_DIR/.run"
log_to_run() { echo "[$(date -u +%FT%TZ)] $*" >> "$RUN_LOG"; }

is_posint() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_nat()    { [[ "$1" =~ ^[0-9]+$ ]]; }
is_num()    { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

CMD="start"
NO_PREFLIGHT=false
DO_DOWNLOAD=true
for arg in "$@"; do
    case "$arg" in
        start|stop|restart|status|logs) CMD="$arg" ;;
        --no-preflight) NO_PREFLIGHT=true ;;
        --launch) DO_DOWNLOAD=false ;;          # watchdog restart path
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) err "Unknown argument: $arg (try --help)" ;;
    esac
done

# ---------------------------------------------------------------------------
# Load + validate .env  (ALL validation happens before we stop anything)
# ---------------------------------------------------------------------------
[[ -f .env ]] || err ".env not found. Run:  cp .env.example .env"
# shellcheck source=.env
source .env

for var in MODEL_ID SERVED_MODEL_NAME PORT TENSOR_PARALLEL_SIZE \
           MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
           PLE_CPU_OFFLOAD_GB CAP_SIZES_LIST IMAGE; do
    [[ -n "${!var:-}" ]] || err "Required variable $var is not set in .env"
done

is_posint "$PORT" && [[ "$PORT" -ge 1 && "$PORT" -le 65535 ]] \
    || err "PORT must be an integer 1-65535 (got: '$PORT')"
is_posint "$MAX_MODEL_LEN" || err "MAX_MODEL_LEN must be a positive integer (got: '$MAX_MODEL_LEN')"
is_posint "$MAX_NUM_SEQS"  || err "MAX_NUM_SEQS must be a positive integer (got: '$MAX_NUM_SEQS')"
is_posint "$MAX_NUM_BATCHED_TOKENS" || err "MAX_NUM_BATCHED_TOKENS must be a positive integer (got: '$MAX_NUM_BATCHED_TOKENS')"
is_nat  "${MTP_NUM_SPECULATIVE_TOKENS:-0}" || err "MTP_NUM_SPECULATIVE_TOKENS must be a non-negative integer"
is_num  "$PLE_CPU_OFFLOAD_GB" && [[ "$PLE_CPU_OFFLOAD_GB" != "0" ]] \
    || err "PLE_CPU_OFFLOAD_GB must be a positive number (got: '$PLE_CPU_OFFLOAD_GB')"
case "$TENSOR_PARALLEL_SIZE" in 2|4|8) ;; *) err "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE invalid — only {2,4,8} (2 KV heads; frozen topology TP4+EP4)." ;; esac

# MTP1 is unusable on this image (temp-0 A/B: MTP1-miss / MTP0-clean; ledger
# B5-DISCRIMINATOR; docs/rebuild/2026-09-22-mtp1-corruption-temp0-diff.md).
if [[ "${MTP_NUM_SPECULATIVE_TOKENS:-0}" -ne 0 ]]; then
    err "MTP_NUM_SPECULATIVE_TOKENS=${MTP_NUM_SPECULATIVE_TOKENS}: MTP1 produces corrupted output on this image (temp-0 diff, ledger B5-DISCRIMINATOR). The validated line is MTP0."
fi

# 98K ceiling: 98304 passes; 130K dies mid-prefill; 170K DEVICE_LOST (error-20).
if [[ "$MAX_MODEL_LEN" -gt 98304 ]]; then
    err "MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds the measured 98304 ceiling — 130K dies in prefill, 170K DEVICE_LOST (error-20; docs/rebuild/2026-09-22-device-lost-130k-170k.md). Set MAX_MODEL_LEN=98304."
fi

# MBT crash class (8192 killed the QSA indexer; 4096 = untested sweep ceiling)
[[ "$MAX_NUM_BATCHED_TOKENS" -le 4096 ]] || err "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS is in the crash class (QSA indexer). Use <= 4096."

# Capture list: digits+commas, must cover MAX_NUM_SEQS (an aligned decode batch
# at an uncaptured size falls back to eager — the v24h 5x cliff).
echo "$CAP_SIZES_LIST" | grep -qE "^[0-9]+(,[0-9]+)*$" || err "CAP_SIZES_LIST must be digits+commas (got: '$CAP_SIZES_LIST')"
REQ_CAP="$MAX_NUM_SEQS"   # MTP0: required captured size == MAX_NUM_SEQS
case ",$CAP_SIZES_LIST," in
    *",$REQ_CAP,"*) : ;;
    *) err "CAP_SIZES_LIST does not cover MAX_NUM_SEQS=$REQ_CAP — aligned batches at that width would fall back to eager (v24h cliff)." ;;
esac

CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next}"
SHM_SIZE="${SHM_SIZE:-16g}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
WEDGE_WATCHDOG_DISABLE="${WEDGE_WATCHDOG_DISABLE:-0}"
WEDGE_WATCHDOG_INTERVAL="${WEDGE_WATCHDOG_INTERVAL:-60}"
WEDGE_WATCHDOG_RETRIES="${WEDGE_WATCHDOG_RETRIES:-3}"
PREFLIGHT_XPU_COUNT="${PREFLIGHT_XPU_COUNT:-4}"
PREFLIGHT_RAM_GB="${PREFLIGHT_RAM_GB:-100}"
PREFLIGHT_SWAP_GB="${PREFLIGHT_SWAP_GB:-64}"
PREFLIGHT_DISK_GB="${PREFLIGHT_DISK_GB:-30}"
PREFLIGHT_ROOT_GB="${PREFLIGHT_ROOT_GB:-40}"
READY_WAIT="${READY_WAIT_SECONDS:-1800}"
GATE_WAIT="${GRAPH_GATE_WAIT_SECONDS:-180}"
export WEDGE_WATCHDOG_INTERVAL WEDGE_WATCHDOG_RETRIES WEDGE_WATCHDOG_DISABLE \
       CONTAINER_NAME PORT SERVED_MODEL_NAME PREFLIGHT_XPU_COUNT

trap 'rc=$?; if [[ "$rc" -ne 0 && "$CMD" != "stop" && -f .run/watchdog.pid ]]; then kill "$(cat .run/watchdog.pid)" 2>/dev/null || true; fi; exit "$rc"' EXIT

running() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$1"; }

# ---------------------------------------------------------------------------
# status / logs / stop  (no validation needed)
# ---------------------------------------------------------------------------
if [[ "$CMD" == "status" ]]; then
    if running "$CONTAINER_NAME"; then
        echo "container : $(docker inspect -f '{{.State.Status}} (up since {{.State.StartedAt}})' "$CONTAINER_NAME")"
    else
        echo "container : NOT RUNNING"
    fi
    api=$(curl -fsS -m 5 "http://localhost:$PORT/v1/models" 2>/dev/null || true)
    if [[ -n "$api" && "$api" == *"$SERVED_MODEL_NAME"* ]]; then
        echo "api       : READY on :$PORT ($SERVED_MODEL_NAME)"
    else
        echo "api       : not answering on :$PORT"
    fi
    if [[ -f .run/watchdog.pid ]] && kill -0 "$(cat .run/watchdog.pid)" 2>/dev/null; then
        echo "watchdog  : running (pid $(cat .run/watchdog.pid), interval ${WEDGE_WATCHDOG_INTERVAL}s)"
    else
        echo "watchdog  : NOT running"
    fi
    echo "log       : .run/server.log ($(wc -l < .run/server.log 2>/dev/null || echo 0) lines)"
    exit 0
fi

if [[ "$CMD" == "logs" ]]; then
    exec docker logs -f "$CONTAINER_NAME"
fi

if [[ "$CMD" == "stop" ]]; then
    info "=== stop: watchdog first, then the container ==="
    [[ -f .run/watchdog.pid ]] && { kill "$(cat .run/watchdog.pid)" 2>/dev/null || true; rm -f .run/watchdog.pid; ok "watchdog stopped"; }
    [[ -f .run/logtail.pid ]] && { kill "$(cat .run/logtail.pid)" 2>/dev/null || true; rm -f .run/logtail.pid; }
    if running "$CONTAINER_NAME"; then
        docker rm -f "$CONTAINER_NAME" >/dev/null
        ok "container $CONTAINER_NAME removed"
    else
        info "container not running"
    fi
    ok "stopped."
    exit 0
fi

# CMD == start / restart ------------------------------------------------------

# ---------------------------------------------------------------------------
# PREFLIGHT — XPU count, RAM, swap, kernel, GuC hash, iommu=off
# ---------------------------------------------------------------------------
GUC_SHA_EXPECT="70d74627e395347ea04c37168d92f01c9e940f4b32e0743b6350ca808fdb67bb"
GUC_FW="/lib/firmware/xe/bmg_guc_70.bin"

exec_preflight() {
    info "=== PREFLIGHT ==="
    local xpu_count=0
    xpu_count=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
    if command -v sycl-ls >/dev/null 2>&1; then
        local c; c=$(sycl-ls 2>/dev/null | grep -c 'level_zero:gpu' || true)
        [[ -n "$c" && "$c" -gt 0 ]] && xpu_count="$c"
    fi
    [[ "$xpu_count" -eq "$PREFLIGHT_XPU_COUNT" ]] \
        || err "PREFLIGHT FAIL — expected $PREFLIGHT_XPU_COUNT XPUs, found $xpu_count. (sycl-ls | grep -c level_zero:gpu; ls /dev/dri/renderD*; lspci | grep -i arc)"
    ok "XPUs: $xpu_count (expected $PREFLIGHT_XPU_COUNT)"

    local mem_kib mem_gib
    mem_kib=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    mem_gib=$(( mem_kib / 1048576 ))
    [[ "$mem_kib" -ge $(( PREFLIGHT_RAM_GB * 1048576 )) ]] \
        || err "PREFLIGHT FAIL — usable RAM ${mem_gib} GiB < ${PREFLIGHT_RAM_GB} GiB floor (the PLE table pins ~51 GiB on top of the 4x device loads)."
    ok "RAM: ${mem_gib} GiB available (floor ${PREFLIGHT_RAM_GB} GiB)"

    local swap_lines swap_bytes swap_gib
    swap_lines=$(swapon --show --noheadings 2>/dev/null | wc -l || echo 0)
    swap_bytes=$(swapon --show --bytes --noheadings 2>/dev/null | awk '{s+=$3} END{print s+0}' || echo 0)
    swap_gib=$(( swap_bytes / 1073741824 ))
    [[ "$swap_lines" -gt 0 && "$swap_bytes" -ge $(( PREFLIGHT_SWAP_GB * 1073741824 )) ]] \
        || err "PREFLIGHT FAIL — swap OFF or ${swap_gib} GiB < ${PREFLIGHT_SWAP_GB} GiB. Create: sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile (and add to /etc/fstab)."
    ok "Swap: ${swap_gib} GiB ON (floor ${PREFLIGHT_SWAP_GB} GiB)"

    local kern
    kern=$(uname -r)
    case "$kern" in
        6.17.0-1010-intel) ok "Kernel: $kern (platform of record)" ;;
        7.0.0-31-generic)  warn "Kernel: $kern — v1 served here in testing, but 6.17.0-1010-intel is the platform of record (scripts/host-setup.sh)."
                           warn "  On 7.0.0-31 the 16-way sustained number is ~300 (soakfix), matching the platform of record; 98K KV is NOT reachable (v1 OOM x3, NEO mirror — docs/rebuild/)." ;;
        *) err "PREFLIGHT FAIL — kernel $kern is not a known-good v1 platform (6.17.0-1010-intel expected; scripts/host-setup.sh installs it)." ;;
    esac

    local guc_sha=""
    [[ -f "$GUC_FW" ]] && guc_sha=$(sha256sum "$GUC_FW" | cut -d' ' -f1)
    if [[ "$guc_sha" == "$GUC_SHA_EXPECT" ]]; then
        ok "GuC: bmg_guc_70.bin = 70.65 (${guc_sha:0:8}…, linux-firmware fb0889c0)"
    else
        err "PREFLIGHT FAIL — GuC firmware hash mismatch (found ${guc_sha:-absent}; expected 70.65 $GUC_SHA_EXPECT). Run scripts/host-setup.sh."
    fi

    if grep -q 'iommu=off' /proc/cmdline; then
        ok "IOMMU: off"
    else
        err "PREFLIGHT FAIL — iommu=off not in cmdline. Set GRUB_CMDLINE_LINUX_DEFAULT=\"iommu=off\" in /etc/default/grub + update-grub, or run scripts/host-setup.sh."
    fi

    local weights_kib weights_gib
    mkdir -p "$HF_CACHE_DIR"
    weights_kib=$(df -Pk "$HF_CACHE_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
    weights_gib=$(( weights_kib / 1048576 ))
    [[ "$weights_kib" -ge $(( PREFLIGHT_DISK_GB * 1048576 )) ]] \
        || err "PREFLIGHT FAIL — weights mount ($HF_CACHE_DIR) has ${weights_gib:-0} GiB free < ${PREFLIGHT_DISK_GB} GiB (tree ~= 186 GB)."
    ok "Disk (weights mount): ${weights_gib} GiB free (floor ${PREFLIGHT_DISK_GB} GiB)"

    local root_kib root_gib
    root_kib=$(df -Pk / 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
    root_gib=$(( root_kib / 1048576 ))
    [[ "$root_kib" -ge $(( PREFLIGHT_ROOT_GB * 1048576 )) ]] \
        || err "PREFLIGHT FAIL — root fs has ${root_gib:-0} GiB free < ${PREFLIGHT_ROOT_GB} GiB."
    ok "Disk (root /): ${root_gib} GiB free (floor ${PREFLIGHT_ROOT_GB} GiB)"
    ok "Preflight passed."
}

if [[ "$NO_PREFLIGHT" == "true" ]]; then
    red "  === PREFLIGHT SKIPPED (--no-preflight) — you own every gate ==="
    log_to_run "PREFLIGHT_SKIPPED=1 (--no-preflight)"
else
    exec_preflight
fi

# ---------------------------------------------------------------------------
# Weights: download if missing, then the hard identity gate
# ---------------------------------------------------------------------------
ORG="${MODEL_ID%%/*}"; NAME="${MODEL_ID##*/}"
HUB_PATH="$HF_CACHE_DIR/hub"
if [[ -d "$HUB_PATH/models--${ORG}--${NAME}/snapshots" ]]; then
    ok "HF snapshot for $MODEL_ID present — download skipped."
    DO_DOWNLOAD=false
fi
if [[ "$DO_DOWNLOAD" == "true" ]]; then
    info "=== Downloading $MODEL_ID (~186 GB) ==="
    log_to_run "downloading $MODEL_ID"
    if command -v uvx >/dev/null 2>&1; then
        HF_HOME="$HF_CACHE_DIR" uvx hf download "$MODEL_ID" --cache-dir "$HUB_PATH"
    elif command -v hf >/dev/null 2>&1; then
        HF_HOME="$HF_CACHE_DIR" hf download "$MODEL_ID" --cache-dir "$HUB_PATH"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_HOME="$HF_CACHE_DIR" huggingface-cli download "$MODEL_ID" --cache-dir "$HUB_PATH"
    else
        err "No HF download tool found (pip install huggingface_hub, or uv)."
    fi
    ok "Download complete."
fi
info "=== Weights identity gate ==="
./check-weights.sh || err "check-weights.sh failed — wrong model or broken snapshot. Never launch into a wrong-weights tree."

# ---------------------------------------------------------------------------
# Image: pull the pinned reference if absent (no local build step)
# ---------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || err "docker not found."
docker info >/dev/null 2>&1 || err "docker daemon not reachable (is your user in the docker group?)."
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    info "Image $IMAGE not local — pulling (pinned digest)."
    docker pull "$IMAGE"
else
    ok "Image present: $IMAGE"
fi

# ---------------------------------------------------------------------------
# Stop any running instance (validation + preflight already passed)
# ---------------------------------------------------------------------------
[[ -f .run/watchdog.pid ]] && { kill "$(cat .run/watchdog.pid)" 2>/dev/null || true; rm -f .run/watchdog.pid; }
[[ -f .run/logtail.pid ]]  && { kill "$(cat .run/logtail.pid)" 2>/dev/null || true; rm -f .run/logtail.pid; }
if running "$CONTAINER_NAME"; then
    info "Stopping existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
ENV_HASH=$(grep -v -E '^HF_TOKEN=' .env | sort | sha256sum | cut -d' ' -f1)
GIT_DESC=$(git -C "$SCRIPT_DIR" describe --always --dirty 2>/dev/null || echo "no-git")
log_to_run "launch start (IMAGE=$IMAGE MODEL=$MODEL_ID TP=$TENSOR_PARALLEL_SIZE MTP=${MTP_NUM_SPECULATIVE_TOKENS:-0} MBT=$MAX_NUM_BATCHED_TOKENS MML=$MAX_MODEL_LEN git=$GIT_DESC envhash=$ENV_HASH)"
cat > .run/manifest.json <<EOF
{
  "kit": "qwen38-flash-next-4xb70",
  "model_id": "$MODEL_ID",
  "image": "$IMAGE",
  "git_describe": "$GIT_DESC",
  "env_hash": "$ENV_HASH",
  "start_iso": "$(date -u +%FT%TZ)",
  "mtp_num_speculative_tokens": "${MTP_NUM_SPECULATIVE_TOKENS:-0}",
  "max_model_len": "$MAX_MODEL_LEN",
  "max_num_seqs": "$MAX_NUM_SEQS",
  "max_num_batched_tokens": "$MAX_NUM_BATCHED_TOKENS",
  "cap_sizes_list": "$CAP_SIZES_LIST",
  "preflight_skipped": "$( [[ "$NO_PREFLIGHT" == "true" ]] && echo 1 || echo 0 )"
}
EOF
ok "Manifest: .run/manifest.json"

# ---------------------------------------------------------------------------
# Wedge watchdog (mandatory; double opt-out to disable)
# ---------------------------------------------------------------------------
if [[ "$WEDGE_WATCHDOG_DISABLE" != "1" || "$NO_PREFLIGHT" != "true" ]]; then
    info "Spawning wedge watchdog (interval=${WEDGE_WATCHDOG_INTERVAL}s, retries=${WEDGE_WATCHDOG_RETRIES})"
    nohup "$SCRIPT_DIR/wedge-watchdog.sh" >> .run/watchdog.log 2>&1 &
    echo "$!" > .run/watchdog.pid
    ok "Watchdog pid $(cat .run/watchdog.pid) (log: .run/watchdog.log)"
else
    red "  === WEDGE WATCHDOG DISABLED (double opt-out) — the rig WILL wedge unattended within 2-6 h under load ==="
    log_to_run "watchdog disabled (double opt-out)"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
info "=== Launching vLLM ($MODEL_ID) ==="
DOCKER_RUN=(docker run -d --name "$CONTAINER_NAME")
# /dev/dri passthrough: device + by-path bind mount (oneCCL drmfd fallback opens
# the device DIRECTORY; --device alone does not carry by-path symlinks).
DOCKER_RUN+=(--device /dev/dri)
DOCKER_RUN+=(-v /dev/dri/by-path:/dev/dri/by-path)
DOCKER_RUN+=(--device-cgroup-rule 'c 226:* rwm')
DOCKER_RUN+=(--group-add video --group-add 991)
# Image ships ENTRYPOINT=api_server — override or the vLLM args become positional (argparse death).
DOCKER_RUN+=(--entrypoint /opt/venv/bin/vllm)
DOCKER_RUN+=(--shm-size "$SHM_SIZE")
DOCKER_RUN+=(--cap-add IPC_LOCK --ulimit memlock=-1:-1)   # PLE worker mlocks its table
DOCKER_RUN+=(-v "$HF_CACHE_DIR:/root/.cache/huggingface")
DOCKER_RUN+=(-v "$SCRIPT_DIR/.run:/.run")
DOCKER_RUN+=(-e "HF_HOME=/root/.cache/huggingface")
DOCKER_RUN+=(-e "HF_HUB_OFFLINE=1" -e "TRANSFORMERS_OFFLINE=1")
DOCKER_RUN+=(-e "VLLM_TARGET_DEVICE=xpu")
DOCKER_RUN+=(-e "MODEL_ID=$MODEL_ID")
DOCKER_RUN+=(-e "SERVED_MODEL_NAME=$SERVED_MODEL_NAME")
DOCKER_RUN+=(-e "PORT=$PORT")
DOCKER_RUN+=(-e "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE")
DOCKER_RUN+=(-e "ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-true}")
DOCKER_RUN+=(-e "MAX_MODEL_LEN=$MAX_MODEL_LEN")
DOCKER_RUN+=(-e "MAX_NUM_SEQS=$MAX_NUM_SEQS")
DOCKER_RUN+=(-e "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS")
DOCKER_RUN+=(-e "MTP_NUM_SPECULATIVE_TOKENS=${MTP_NUM_SPECULATIVE_TOKENS:-0}")
DOCKER_RUN+=(-e "PLE_CPU_OFFLOAD_GB=$PLE_CPU_OFFLOAD_GB")
DOCKER_RUN+=(-e "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=${VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY:-0}")
# Anti-envs — these must NEVER come on for this recipe:
DOCKER_RUN+=(-e "SYCL_CACHE_PERSISTENT=0")            # =1 poisons the B70 cache -> SEGV next boot
DOCKER_RUN+=(-e "VLLM_PLE_CPU_OFFLOAD=0")             # NVIDIA-only worker path; XPU uses UVA
DOCKER_RUN+=(-e "VLLM_XPU_PLE_UVA_PREFETCH=0")        # async UVA prefetch rejected (A26/A27)
DOCKER_RUN+=(-e "VLLM_XPU_QWEN4_EXP_HC_GROUPED_UP=0") # grouped-HC negative (A30)
if [[ -n "${HF_TOKEN:-}" ]]; then DOCKER_RUN+=(-e "HF_TOKEN=$HF_TOKEN"); fi
if [[ -n "${EXTRA_DOCKER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    DOCKER_RUN+=($EXTRA_DOCKER_ARGS)
fi

VLLM_ARGS=(serve "$MODEL_ID")
VLLM_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
VLLM_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
if [[ "${ENABLE_EXPERT_PARALLEL:-true}" == "true" ]]; then
    VLLM_ARGS+=(--enable-expert-parallel)
    VLLM_ARGS+=(--all2all-backend allgather_reducescatter)   # frozen EP4 identity
fi
VLLM_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
VLLM_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
VLLM_ARGS+=(--load-format safetensors)
VLLM_ARGS+=(--cpu-offload-gb "$PLE_CPU_OFFLOAD_GB")
VLLM_ARGS+=(--host 0.0.0.0 --port "$PORT")
# Decode graphs: FULL_DECODE_ONLY with the explicit capture list (v24h2).
VLLM_ARGS+=(--compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$CAP_SIZES_LIST]}")
if [[ "${MTP_NUM_SPECULATIVE_TOKENS:-0}" -gt 0 ]]; then
    VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_NUM_SPECULATIVE_TOKENS}")
fi
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    VLLM_ARGS+=($EXTRA_VLLM_ARGS)
fi

log_to_run "docker run argv: ${DOCKER_RUN[*]} $IMAGE ${VLLM_ARGS[*]}"
"${DOCKER_RUN[@]}" "$IMAGE" "${VLLM_ARGS[@]}" >/dev/null
ok "Container started: $CONTAINER_NAME"
docker logs -f "$CONTAINER_NAME" > .run/server.log 2>&1 &
echo "$!" > .run/logtail.pid
log_to_run "logtail pid $(cat .run/logtail.pid)"

# ---------------------------------------------------------------------------
# Readiness: /v1/models (up to READY_WAIT) AND graph-capture lines in the log
# ---------------------------------------------------------------------------
info "Loading ~186 GB tree + graph capture — polling /v1/models up to ${READY_WAIT}s..."
DEADLINE=$(( $(date +%s) + READY_WAIT ))
while :; do
    running "$CONTAINER_NAME" || err "Container exited during load. Tail: $(tail -20 .run/server.log)"
    MODELS_JSON=$(curl -fsS -m 10 "http://localhost:$PORT/v1/models" 2>/dev/null || true)
    [[ -n "$MODELS_JSON" && "$MODELS_JSON" == *"$SERVED_MODEL_NAME"* ]] && break
    [[ "$(date +%s)" -gt "$DEADLINE" ]] && err "Timed out after ${READY_WAIT}s waiting for /v1/models. Tail: $(tail -20 .run/server.log)"
    sleep 15
done

info "API up — gating READY on graph-capture log lines (${GATE_WAIT}s grace)..."
GATE_DEADLINE=$(( $(date +%s) + GATE_WAIT ))
CAP_N=0
while :; do
    CAP_N=$(grep -c "Graph capturing finished" .run/server.log 2>/dev/null || echo 0)
    [[ "$CAP_N" -ge 1 ]] && break
    running "$CONTAINER_NAME" || err "Container exited while waiting for graph capture. Tail: $(tail -20 .run/server.log)"
    [[ "$(date +%s)" -gt "$GATE_DEADLINE" ]] \
        && err "READY GATE FAIL — /v1/models answers but no 'Graph capturing finished' lines in .run/server.log. Decode graphs did not capture; serving would fall back to eager."
    sleep 10
done
ok "Graph capture confirmed (${CAP_N} worker lines)."

# ---------------------------------------------------------------------------
# PLE warm-up (production hotfix 2026-09-22): a cold PLE table page-in blows
# the connector staging window (now 120 s) -> torn embeddings. Fire one
# tool-call request at boot. Logged, not fatal.
# ---------------------------------------------------------------------------
python3 - "$PORT" <<'PYW' || warn "PLE warm-up request failed (non-fatal; engine stays up)"
import json, sys, urllib.request
port = sys.argv[1]
body = {"model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
        "tools": [{"type": "function", "function": {"name": "get_weather",
            "description": "Get the current weather conditions for a city.",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
            "required": ["location"]}}}],
        "tool_choice": "auto", "max_tokens": 96, "temperature": 0}
req = urllib.request.Request("http://localhost:%s/v1/chat/completions" % port,
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=180) as r:
    obj = json.loads(r.read())
tc = obj.get("choices", [{}])[0].get("message", {}).get("tool_calls")
print("[WARMUP] HTTP 200 tool_calls=%s" % ("YES" if tc else "NO"))
PYW

log_to_run "READY (api + graph-capture gate passed)"
ok "READY — $SERVED_MODEL_NAME on :$PORT (graphs captured, watchdog armed)."
info "  status: ./start.sh status   logs: ./start.sh logs   stop: ./start.sh stop"

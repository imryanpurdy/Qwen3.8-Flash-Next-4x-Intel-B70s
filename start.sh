#!/usr/bin/env bash
# ============================================================================
# start.sh — Qwen3.8-Flash-Next W4A16 on 4x Intel Arc Pro B70 (TP4+EP)
#            SIDE-LANE STACK (deploy of record, 2026-09-23)
#
# Commands: start | stop | restart | status | logs   (default: start)
#   ./start.sh              # validate -> preflight -> weights -> XPU gate -> image -> launch
#   ./start.sh stop         # watchdog first, then the container (graceful)
#   ./start.sh restart      # full validation path, then stop + start
#   ./start.sh status       # container + API + watchdog state
#   ./start.sh logs         # docker logs -f
#   ./start.sh --launch     # launch-only, skips the XPU gate (watchdog restart path)
#   ./start.sh --no-preflight  # skip the preflight gate (loud WARN)
#
# The verified line (their stack, byte-identical to boots of record):
#   vLLM fork devan-carlin/vllm@xpu-qwen4exp (a69fba21), image from
#   intel/omix:0.4.0-devel-ubuntu24.04; TP4+EP, MML 262144, MNS 16, kv fp8,
#   parsers qwen3/qwen3_xml, sampler pin temp 0.7/top_p 0.80/top_k 20/presence 1.5.
#   All knobs in .env (cp .env.example .env).
#
# Design rules, in order:
#   1. Knob validation happens BEFORE any running service is touched.
#   2. Preflight gates: 4 XPUs, RAM, swap, kernel, GuC hash, iommu=off.
#   3. Weights are identity-gated in place — never launch into a wrong tree.
#   4. PRE-BOOT XPU GATE (mandatory): trivial triton vector-add on one card
#      must compile AND produce the exact result before any model boot.
#   5. READY is gated on /v1/models AND "Application startup complete".
#   6. The wedge watchdog is mandatory (double opt-out to disable).
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
mkdir -p "$SCRIPT_DIR/.run/eslane"
log_to_run() { echo "[$(date -u +%FT%TZ)] $*" >> "$RUN_LOG"; }

is_posint() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_num()    { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

CMD="start"
NO_PREFLIGHT=false
SKIP_XPU_GATE=false
for arg in "$@"; do
    case "$arg" in
        start|stop|restart|status|logs) CMD="$arg" ;;
        --no-preflight) NO_PREFLIGHT=true ;;
        --launch) SKIP_XPU_GATE=true ;;        # watchdog restart path
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) err "Unknown argument: $arg (try --help)" ;;
    esac
done

# ---------------------------------------------------------------------------
# Load + validate .env  (ALL validation happens before we stop anything)
# ---------------------------------------------------------------------------
[[ -f .env ]] || err ".env not found. Run:  cp .env.example .env"
# shellcheck source=.env
source .env

for var in MODEL_PATH PLE_TABLE_PATH SERVED_MODEL_NAME PORT TENSOR_PARALLEL_SIZE \
           MAX_MODEL_LEN MAX_NUM_SEQS DTYPE KV_CACHE_DTYPE GPU_MEMORY_UTILIZATION \
           OVERRIDE_GENERATION_CONFIG IMAGE; do
    [[ -n "${!var:-}" ]] || err "Required variable $var is not set in .env"
done

is_posint "$PORT" && [[ "$PORT" -ge 1 && "$PORT" -le 65535 ]] \
    || err "PORT must be an integer 1-65535 (got: '$PORT')"
is_posint "$MAX_MODEL_LEN" || err "MAX_MODEL_LEN must be a positive integer (got: '$MAX_MODEL_LEN')"
is_posint "$MAX_NUM_SEQS"  || err "MAX_NUM_SEQS must be a positive integer (got: '$MAX_NUM_SEQS')"
case "$TENSOR_PARALLEL_SIZE" in 4) ;; *) err "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE invalid — the verified line is TP4 (2 KV heads; TP6 impossible)." ;; esac

# Their-line ceilings (validated, one variable per boot law):
#   MML 262144 = their line; 250,700-ptok needle CORRECT on it. Above = untested.
if [[ "$MAX_MODEL_LEN" -gt 262144 ]]; then
    err "MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds the validated 262144 (their MML; 250K needle CORRECT at 262144). Above is untested — gate it first."
fi
#   MNS 16 = soak-validated operating point (626.3 tok/s sustained, 0 restarts).
#   MNS ladder 2026-09-24 (quiet engine, no spec decode): 2->95, 4->180, 8->335, 16->629 tok/s agg.
#   32 = KV knee at MML 262144 (809,600/886,567 tokens at 25% long-mix).
#   17-31 untested interim values -> WARN; 32 = gated test target; >32 hard-fails.
if [[ "$MAX_NUM_SEQS" -gt 32 ]]; then
    err "MAX_NUM_SEQS=$MAX_NUM_SEQS exceeds the KV-math knee of 32 at MML 262144 (gate it first)."
fi
if [[ "$MAX_NUM_SEQS" -gt 16 && "$MAX_NUM_SEQS" -ne 32 ]]; then
    echo "WARN: MAX_NUM_SEQS=$MAX_NUM_SEQS is in the untested 17-31 band (validated: 16; 32 under test)." >&2
fi

WEDGE_WATCHDOG_DISABLE="${WEDGE_WATCHDOG_DISABLE:-0}"
WEDGE_WATCHDOG_INTERVAL="${WEDGE_WATCHDOG_INTERVAL:-60}"
WEDGE_WATCHDOG_RETRIES="${WEDGE_WATCHDOG_RETRIES:-3}"
XPU_GATE_DISABLE="${XPU_GATE_DISABLE:-0}"
PREFLIGHT_XPU_COUNT="${PREFLIGHT_XPU_COUNT:-4}"
PREFLIGHT_RAM_GB="${PREFLIGHT_RAM_GB:-100}"
PREFLIGHT_SWAP_GB="${PREFLIGHT_SWAP_GB:-64}"
PREFLIGHT_ROOT_GB="${PREFLIGHT_ROOT_GB:-40}"
READY_WAIT="${READY_WAIT_SECONDS:-900}"
export WEDGE_WATCHDOG_INTERVAL WEDGE_WATCHDOG_RETRIES WEDGE_WATCHDOG_DISABLE \
       CONTAINER_NAME PORT SERVED_MODEL_NAME PREFLIGHT_XPU_COUNT

trap 'rc=$?; if [[ "$rc" -ne 0 && "$CMD" != "stop" && -f .run/eslane/watchdog.pid ]]; then kill "$(cat .run/eslane/watchdog.pid)" 2>/dev/null || true; fi; exit "$rc"' EXIT

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
    if [[ -f .run/eslane/watchdog.pid ]] && kill -0 "$(cat .run/eslane/watchdog.pid)" 2>/dev/null; then
        echo "watchdog  : running (pid $(cat .run/eslane/watchdog.pid), interval ${WEDGE_WATCHDOG_INTERVAL}s)"
    else
        echo "watchdog  : NOT running"
    fi
    echo "log       : docker logs $CONTAINER_NAME"
    exit 0
fi

if [[ "$CMD" == "logs" ]]; then
    exec docker logs -f "$CONTAINER_NAME"
fi

if [[ "$CMD" == "stop" ]]; then
    info "=== stop: watchdog first, then the container ==="
    [[ -f .run/eslane/watchdog.pid ]] && { kill "$(cat .run/eslane/watchdog.pid)" 2>/dev/null || true; rm -f .run/eslane/watchdog.pid; ok "watchdog stopped"; }
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
# (platform of record — identical floors to the v1 kit; scripts/host-setup.sh)
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
        || err "PREFLIGHT FAIL — usable RAM ${mem_gib} GiB < ${PREFLIGHT_RAM_GB} GiB floor."
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
        *) err "PREFLIGHT FAIL — kernel $kern is not the platform of record (6.17.0-1010-intel expected; scripts/host-setup.sh installs it)." ;;
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

    # Weights are a LOCAL tree (bind-mounted read-only): serving needs no
    # download headroom. Free space on the weights mount is REPORT-ONLY
    # (the rig's /data runs at 92% and that is fine for serving); the HARD
    # gate is that the weights exist, done after preflight. 2 GiB floor is
    # a serving-operations floor (logs/caches), not a download floor.
    local wroot_mnt wroot_kib wroot_gib
    wroot_mnt=$(df -Pk "$MODEL_PATH" 2>/dev/null | awk 'NR==2{print $6}')
    wroot_kib=$(df -Pk "$MODEL_PATH" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
    wroot_gib=$(( wroot_kib / 1048576 ))
    if [[ "$wroot_kib" -lt $(( 2 * 1048576 )) ]]; then
        err "PREFLIGHT FAIL — weights mount has ${wroot_gib:-0} GiB free < 2 GiB serving floor."
    fi
    [[ "$wroot_gib" -lt 30 ]] \
        && warn "Disk (weights mount $wroot_mnt): ${wroot_gib} GiB free — tight but sufficient for a local-weights serve (report-only standing state; no download occurs)."
    [[ "$wroot_gib" -ge 30 ]] \
        && ok "Disk (weights mount $wroot_mnt): ${wroot_gib} GiB free (serving floor 2 GiB)"

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
# Weights: local-tree identity gate (no HF download — tree lives on /data)
# ---------------------------------------------------------------------------
[[ -d "$MODEL_PATH" ]] || err "MODEL_PATH $MODEL_PATH does not exist."
[[ -f "$MODEL_PATH/config.json" ]] || err "config.json missing in $MODEL_PATH — wrong or torn weights tree."
local_shards=$(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l)
[[ "$local_shards" -ge 1 ]] || err "no *.safetensors shards in $MODEL_PATH — wrong or torn weights tree."
[[ -f "$PLE_TABLE_PATH" ]] || err "PLE table missing: $PLE_TABLE_PATH (Qwen4Exp MTP layer needs it)."
ok "Weights tree: $MODEL_PATH (${local_shards} shards)"
ok "PLE table: $PLE_TABLE_PATH ($(du -h "$PLE_TABLE_PATH" | cut -f1))"

# ---------------------------------------------------------------------------
# PRE-BOOT XPU GATE — trivial triton vector-add must compile AND be exact
# (mandatory standing rule; catches every remaining JIT gap in seconds)
# ---------------------------------------------------------------------------
if [[ "$SKIP_XPU_GATE" == "true" ]]; then
    info "XPU gate skipped (--launch restart path)."
elif [[ "$XPU_GATE_DISABLE" == "1" ]]; then
    red "  === XPU GATE DISABLED (XPU_GATE_DISABLE=1) — you own every JIT gap ==="
    log_to_run "XPU_GATE_SKIPPED=1"
else
    info "=== Pre-boot XPU gate (triton vector-add on one card) ==="
    command -v docker >/dev/null 2>&1 || err "docker not found."
    GATE_OUT=$(docker run --rm --name es-xpu-gate --device /dev/dri \
        -v "$SCRIPT_DIR/docker/sidelane/gate.py:/gate.py:ro" \
        --entrypoint python3 "$IMAGE" /gate.py 2>&1) \
        || { echo "$GATE_OUT" | tail -5; err "XPU gate failed to run (see above)."; }
    echo "$GATE_OUT" | grep -q "TRITON_XPU_GATE=PASS" \
        || { echo "$GATE_OUT" | tail -5; err "XPU GATE FAIL — vector-add did not compile/verify (see above). Fix before any model boot."; }
    ok "TRITON_XPU_GATE=PASS (compile + exact result)"
    log_to_run "XPU_GATE=PASS"
fi

# ---------------------------------------------------------------------------
# Image: the pinned digest must already exist locally (no pull: local build
# of record until GHCR push — README Open Items)
# ---------------------------------------------------------------------------
docker info >/dev/null 2>&1 || err "docker daemon not reachable (is your user in the docker group?)."
IMAGE_REF="$IMAGE"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    LOCAL_TAG="es-lane:qwen4exp-a69fba21"
    if docker image inspect "$LOCAL_TAG" >/dev/null 2>&1; then
        warn "Pinned digest $IMAGE not found locally; using local build tag $LOCAL_TAG."
        warn "A fresh host builds it: docker build -t $LOCAL_TAG docker/sidelane/ (README — GHCR push is OPEN)."
        IMAGE_REF="$LOCAL_TAG"
    else
        err "Neither $IMAGE nor $LOCAL_TAG present locally. Build it: docker build -t $LOCAL_TAG docker/sidelane/ (see README Open Items — GHCR push pending)."
    fi
fi
ok "Image present: $IMAGE_REF"

# ---------------------------------------------------------------------------
# Stop any running instance (validation + preflight already passed)
# ---------------------------------------------------------------------------
[[ -f .run/eslane/watchdog.pid ]] && { kill "$(cat .run/eslane/watchdog.pid)" 2>/dev/null || true; rm -f .run/eslane/watchdog.pid; }
if running "$CONTAINER_NAME"; then
    info "Stopping existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
ENV_HASH=$(grep -v -E '^HF_TOKEN=' .env | sort | sha256sum | cut -d' ' -f1)
GIT_DESC=$(git -C "$SCRIPT_DIR" describe --always --dirty 2>/dev/null || echo "no-git")
log_to_run "launch start (IMAGE=$IMAGE MODEL=$MODEL_PATH TP=$TENSOR_PARALLEL_SIZE MML=$MAX_MODEL_LEN MNS=$MAX_NUM_SEQS git=$GIT_DESC envhash=$ENV_HASH)"
cat > .run/manifest.json <<EOF
{
  "kit": "qwen38-flash-next-4xb70-sidelane",
  "model_path": "$MODEL_PATH",
  "image": "$IMAGE",
  "git_describe": "$GIT_DESC",
  "env_hash": "$ENV_HASH",
  "start_iso": "$(date -u +%FT%TZ)",
  "max_model_len": "$MAX_MODEL_LEN",
  "max_num_seqs": "$MAX_NUM_SEQS",
  "served_model_name": "$SERVED_MODEL_NAME"
}
EOF
ok "Manifest: .run/manifest.json"

# ---------------------------------------------------------------------------
# Wedge watchdog (mandatory; double opt-out to disable)
# ---------------------------------------------------------------------------
if [[ "$WEDGE_WATCHDOG_DISABLE" != "1" || "$NO_PREFLIGHT" != "true" ]]; then
    info "Spawning wedge watchdog (interval=${WEDGE_WATCHDOG_INTERVAL}s, retries=${WEDGE_WATCHDOG_RETRIES})"
    nohup "$SCRIPT_DIR/scripts/wedge-watchdog-eslane.sh" >> .run/eslane/watchdog-eslane.log 2>&1 &
    echo "$!" > .run/eslane/watchdog.pid
    ok "Watchdog pid $(cat .run/eslane/watchdog.pid) (log: .run/eslane/watchdog-eslane.log)"
else
    red "  === WEDGE WATCHDOG DISABLED (double opt-out) — the rig WILL wedge unattended within 2-6 h under load ==="
    log_to_run "watchdog disabled (double opt-out)"
fi

# ---------------------------------------------------------------------------
# Launch — the verified line, byte-identical to boots of record
# (idempotent: rm -f first; a stale container must not block re-creation)
# ---------------------------------------------------------------------------
info "=== Launching vLLM side-lane stack ($SERVED_MODEL_NAME on :$PORT) ==="
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER_NAME" \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v "$MODEL_PATH:/data/hf-devan/Qwen3.8-Flash-Next-W4A16:ro" \
  --shm-size "$SHM_SIZE" \
  -p "$PORT:$PORT" \
  -e ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1,2,3}" \
  -e ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0,1,2,3}" \
  -e MASTER_ADDR=127.0.0.1 \
  -e PLE_TABLE_PATH=/data/hf-devan/Qwen3.8-Flash-Next-W4A16/ple_table_qwen4exp.pt \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e MASTER_PORT=29530 \
  -e VLLM_CACHE_ROOT=/root/.cache/vllm \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TRANSFORMERS_OFFLINE=1 \
  -e UR_L0_SYNC_MODE="${UR_L0_SYNC_MODE:-BLOCKING}" \
  -e VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}" \
  -e CCL_TOPO_P2P_ACCESS="${CCL_TOPO_P2P_ACCESS:-0}" \
  -e VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_XPU_ENABLE_XPU_GRAPH:-1}" \
  -e MAX_JOBS="${MAX_JOBS:-16}" \
  $IMAGE \
  python3 -m vllm.entrypoints.openai.api_server \
  --model /data/hf-devan/Qwen3.8-Flash-Next-W4A16 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --enable-expert-parallel \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --reasoning-parser "$REASONING_PARSER" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --generation-config "$GENERATION_CONFIG" \
  --override-generation-config "$OVERRIDE_GENERATION_CONFIG"
ok "Container started: $CONTAINER_NAME"
log_to_run "container up (boot $(date -u +%FT%TZ))"

# ---------------------------------------------------------------------------
# Readiness: /v1/models (up to READY_WAIT) AND "Application startup complete"
# (their stack's READY receipt; graphs capture inside torch.compile before it)
# ---------------------------------------------------------------------------
info "Loading weights + torch.compile (~4-5 min) — polling /v1/models up to ${READY_WAIT}s..."
DEADLINE=$(( $(date +%s) + READY_WAIT ))
while :; do
    running "$CONTAINER_NAME" || err "Container exited during load. Tail: $(docker logs --tail 20 "$CONTAINER_NAME" 2>&1)"
    MODELS_JSON=$(curl -fsS -m 10 "http://localhost:$PORT/v1/models" 2>/dev/null || true)
    [[ -n "$MODELS_JSON" && "$MODELS_JSON" == *"$SERVED_MODEL_NAME"* ]] && break
    [[ "$(date +%s)" -gt "$DEADLINE" ]] && err "Timed out after ${READY_WAIT}s waiting for /v1/models. Tail: $(docker logs --tail 20 "$CONTAINER_NAME" 2>&1)"
    sleep 15
done

info "API up — gating READY on the engine startup receipt..."
GATE_DEADLINE=$(( $(date +%s) + 180 ))
while :; do
    if docker logs "$CONTAINER_NAME" 2>&1 | grep -q "Application startup complete"; then
        break
    fi
    running "$CONTAINER_NAME" || err "Container exited while waiting for startup receipt. Tail: $(docker logs --tail 20 "$CONTAINER_NAME" 2>&1)"
    [[ "$(date +%s)" -gt "$GATE_DEADLINE" ]] \
        && err "READY GATE FAIL — /v1/models answers but no 'Application startup complete' in container logs."
    sleep 10
done
ok "Engine startup receipt confirmed."

log_to_run "READY (api + startup-receipt gate passed)"
ok "READY — $SERVED_MODEL_NAME on :$PORT (startup receipt confirmed, watchdog armed)."
info "  status: ./start.sh status   logs: ./start.sh logs   stop: ./start.sh stop"

#!/usr/bin/env bash
# ============================================================================
# start.sh — PRODUCTION LANE: Lumnus b70-flash-next engine (vLLM v0.30.0 +
#            patch series 0001-0019), wtdcode AWQ checkpoint, INT8 PLE NVMe.
#            4x Intel Arc Pro B70 (TP4+EP), qwen-256k.
#            Repo-generalized from the deployed production launcher: every
#            path and knob rides .env (cp .env.example .env).
#
# Commands: start | stop | restart | status | logs   (default: start)
#   ./scripts/start.sh               # validate -> preflight -> weights -> XPU gate -> image -> launch
#   ./scripts/start.sh stop          # watchdog first, then the container (graceful; delegates to stop.sh)
#   ./scripts/start.sh restart       # full validation path, then stop + start
#   ./scripts/start.sh status        # container + API + watchdog state (delegates to status.sh)
#   ./scripts/start.sh logs          # docker logs -f
#   ./scripts/start.sh --launch      # launch-only, skips the XPU gate (watchdog restart path)
#   ./scripts/start.sh --no-preflight  # skip the preflight gate (loud WARN)
#   ./scripts/start.sh --dry-run     # validate + preflight + print the exact
#                                    # docker run line, launch NOTHING (clean-room diff)
#
# The verified line (trial-proven, promoted 2026-10-04):
#   Lumnus b70-flash-next engine (vLLM v0.30.0 + series 0001-0019), wtdcode
#   AWQ checkpoint, INT8 PLE table from NVMe (native reader), TP4+EP,
#   MML 262144, MNS 32, KV offload tier, copy-offload flag, sampler pin.
#   Serve flags in $SERVE_ARGS; engine env in $LUMNUS_ENV (docker --env-file).
#
# Design rules, in order:
#   1. Knob validation happens BEFORE any running service is touched.
#   2. Preflight gates: 4 XPUs, RAM, swap, kernel, GuC hash, iommu=off,
#      docker+buildx, disk floors (weights mount report-only, PLE NVMe and
#      root hard), sudo -n dmesg readability (the watchdog's xe
#      engine-reset monitor needs it).
#   3. Weights are identity-gated in place — never launch into a wrong tree.
#   4. PRE-BOOT XPU GATE (mandatory): trivial triton vector-add on one card
#      must compile AND produce the exact result before any model boot.
#   5. READY is gated on /v1/models AND "Application startup complete".
#   6. The wedge watchdog is mandatory (double opt-out to disable), with a
#      single-instance guard: two watchdogs double-probe and race the pidfile.
# All runtime state lives under $LANE_DIR/.run/ (default: repo root .run/).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }
red()  { echo -e "\033[1;31m$*\033[0m"; }

CMD="start"
NO_PREFLIGHT=false
SKIP_XPU_GATE=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        start|stop|restart|status|logs) CMD="$arg" ;;
        --no-preflight) NO_PREFLIGHT=true ;;
        --launch) SKIP_XPU_GATE=true ;;        # watchdog restart path
        --dry-run) DRY_RUN=true ;;             # print the docker line, launch nothing
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) err "Unknown argument: $arg (try --help)" ;;
    esac
done

# ---------------------------------------------------------------------------
# Load + validate .env  (ALL validation happens before we stop anything)
# ---------------------------------------------------------------------------
[[ -f .env ]] || err ".env not found. Run:  cp .env.example .env"
# shellcheck source=.env
source .env

# LANE_DIR: where runtime state lives (.run/ logs, pidfiles, wedge captures).
# Default: the repo root. Point it elsewhere to run several lanes from one tree.
LANE_DIR="${LANE_DIR:-$REPO_DIR}"
RUN_DIR="$LANE_DIR/.run"
mkdir -p "$RUN_DIR"
RUN_LOG="$RUN_DIR/start.log"
log_to_run() { echo "[$(date -u +%FT%TZ)] $*" >> "$RUN_LOG"; }

is_posint() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

for var in MODEL_PATH PLE_TABLE_PATH SERVED_MODEL_NAME PORT IMAGE CONTAINER_NAME \
           LUMNUS_ENV SERVE_ARGS KV_OFFLOADING_SIZE LANE_DIR \
           PLE_BF16_DIR PLE_INT8_DIR AWQ_ORIG_DIR LANE_CACHE; do
    [[ -n "${!var:-}" ]] || err "Required variable $var is not set in .env"
done

is_posint "$PORT" && [[ "$PORT" -ge 1 && "$PORT" -le 65535 ]] \
    || err "PORT must be an integer 1-65535 (got: '$PORT')"
is_posint "$KV_OFFLOADING_SIZE" \
    || err "KV_OFFLOADING_SIZE must be a positive integer (GiB total over the TP ranks; power of two — the awq env header)."

# Serve flags live in $SERVE_ARGS (single source of truth for MML/MNS/TP).
# Parse and gate them here so a bad args edit fails BEFORE any service is touched.
# Relative paths resolve against the repo root (production convention).
[[ "$SERVE_ARGS" != /* ]] && SERVE_ARGS="$REPO_DIR/$SERVE_ARGS"
[[ "$LUMNUS_ENV" != /* ]] && LUMNUS_ENV="$REPO_DIR/$LUMNUS_ENV"
[[ -f "$SERVE_ARGS" ]] || err "SERVE_ARGS file not found: $SERVE_ARGS. If you just cloned: cp serve-args <name> and point SERVE_ARGS at it (see .env.example)."
[[ -f "$LUMNUS_ENV" ]] || err "LUMNUS_ENV file not found: $LUMNUS_ENV. If you just cloned: cp lumnus.env.example lumnus.env (see .env.example)."
SERVE_MML=$(grep -oE '^--max-model-len [0-9]+' "$SERVE_ARGS" | awk '{print $2}')
SERVE_MNS=$(grep -oE '^--max-num-seqs [0-9]+'  "$SERVE_ARGS" | awk '{print $2}')
SERVE_TP=$(grep -oE '^--tensor-parallel-size [0-9]+' "$SERVE_ARGS" | awk '{print $2}')
is_posint "$SERVE_MML" || err "serve args: --max-model-len missing/invalid"
is_posint "$SERVE_MNS" || err "serve args: --max-num-seqs missing/invalid"
[[ "$SERVE_TP" == "4" ]] || err "serve args: --tensor-parallel-size must be 4 (TP4; 2 KV heads — TP6 impossible)"
if [[ "$SERVE_MML" -gt 262144 ]]; then
    err "serve args: MAX_MODEL_LEN=$SERVE_MML exceeds the validated 262144 (250K needle CORRECT at 262144). Above is untested — gate it first."
fi
#   MNS 32 = soak-validated operating point on the Lumnus engine (3112/3112,
#   0 err, 60 min; n32 fan-out 1118.4 tok/s). >32 hard-fails (KV knee).
if [[ "$SERVE_MNS" -gt 32 ]]; then
    err "serve args: MAX_NUM_SEQS=$SERVE_MNS exceeds the KV-math knee of 32 at MML 262144 (gate it first)."
fi
MAX_MODEL_LEN="$SERVE_MML"; MAX_NUM_SEQS="$SERVE_MNS"; TENSOR_PARALLEL_SIZE="$SERVE_TP"

WEDGE_WATCHDOG_DISABLE="${WEDGE_WATCHDOG_DISABLE:-0}"
WEDGE_WATCHDOG_INTERVAL="${WEDGE_WATCHDOG_INTERVAL:-60}"
WEDGE_WATCHDOG_RETRIES="${WEDGE_WATCHDOG_RETRIES:-3}"
XPU_GATE_DISABLE="${XPU_GATE_DISABLE:-0}"
PREFLIGHT_XPU_COUNT="${PREFLIGHT_XPU_COUNT:-4}"
PREFLIGHT_RAM_GB="${PREFLIGHT_RAM_GB:-100}"
PREFLIGHT_SWAP_GB="${PREFLIGHT_SWAP_GB:-64}"
PREFLIGHT_ROOT_GB="${PREFLIGHT_ROOT_GB:-40}"
PREFLIGHT_PLE_NVME_GB="${PREFLIGHT_PLE_NVME_GB:-100}"
READY_WAIT="${READY_WAIT_SECONDS:-900}"
export WEDGE_WATCHDOG_INTERVAL WEDGE_WATCHDOG_RETRIES WEDGE_WATCHDOG_DISABLE \
       CONTAINER_NAME PORT SERVED_MODEL_NAME PREFLIGHT_XPU_COUNT LANE_DIR

# When the watchdog itself calls us (restart path via PROD_RESTART_CMD), it
# sets WEDGE_WATCHDOG_ALREADY_RUNNING=1: never kill or respawn the calling
# watchdog, or its bounded-retry counter resets.
WD_CALLER="${WEDGE_WATCHDOG_ALREADY_RUNNING:-0}"
trap 'rc=$?; if [[ "$rc" -ne 0 && "$CMD" != "stop" && "$WD_CALLER" != "1" && -f "$RUN_DIR/watchdog.pid" ]]; then kill "$(cat "$RUN_DIR/watchdog.pid")" 2>/dev/null || true; fi; exit "$rc"' EXIT

running() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$1"; }

# ---------------------------------------------------------------------------
# status / logs / stop  (no validation needed; thin delegates)
# ---------------------------------------------------------------------------
if [[ "$CMD" == "status" ]]; then
    exec "$SCRIPT_DIR/status.sh"
fi
if [[ "$CMD" == "logs" ]]; then
    [[ -f .env ]] || err ".env not found. Run:  cp .env.example .env"
    exec docker logs -f "${CONTAINER_NAME:-b70-lumnus-prod}"
fi
if [[ "$CMD" == "stop" ]]; then
    exec "$SCRIPT_DIR/stop.sh"
fi

# CMD == start / restart ------------------------------------------------------

# ---------------------------------------------------------------------------
# PREFLIGHT — XPU count, RAM, swap, kernel, GuC hash, iommu=off, docker+buildx,
# disk floors, sudo -n dmesg readability (platform of record; host-setup.sh
# installs the kernel/firmware/grub pieces).
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

    # docker daemon + buildx (buildx needed to rebuild the image of record;
    # the launch itself only needs the daemon).
    docker info >/dev/null 2>&1 || err "PREFLIGHT FAIL — docker daemon not reachable (is your user in the docker group?)."
    ok "Docker daemon reachable"
    docker buildx version >/dev/null 2>&1 \
        || err "PREFLIGHT FAIL — docker buildx missing (the image of record is built with buildx; install docker-buildx-plugin)."
    ok "Docker buildx present"

    # Weights are a LOCAL tree (bind-mounted read-only): serving needs no
    # download headroom. Free space on the weights mount is REPORT-ONLY
    # (the weights NVMe runs tight and that is fine for serving); the HARD
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

    # PLE INT8 NVMe: the INT8 rowscale table is SERVED FROM THIS NVMe through
    # a small pinned row cache (native reader). Unlike the weights mount this
    # is a hot path — a full or slow mount is a hard fail.
    if [[ -n "${PLE_INT8_DIR:-}" && -d "${PLE_INT8_DIR}" ]]; then
        local ple_mnt ple_kib ple_gib
        ple_mnt=$(df -Pk "$PLE_INT8_DIR" 2>/dev/null | awk 'NR==2{print $6}')
        ple_kib=$(df -Pk "$PLE_INT8_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
        ple_gib=$(( ple_kib / 1048576 ))
        [[ "$ple_kib" -ge $(( PREFLIGHT_PLE_NVME_GB * 1048576 )) ]] \
            || err "PREFLIGHT FAIL — PLE NVMe mount ($ple_mnt) has ${ple_gib:-0} GiB free < ${PREFLIGHT_PLE_NVME_GB} GiB floor (the INT8 table is served from NVMe — keep headroom)."
        ok "Disk (PLE NVMe mount $ple_mnt): ${ple_gib} GiB free (floor ${PREFLIGHT_PLE_NVME_GB} GiB)"
    fi

    local root_kib root_gib
    root_kib=$(df -Pk / 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
    root_gib=$(( root_kib / 1048576 ))
    [[ "$root_kib" -ge $(( PREFLIGHT_ROOT_GB * 1048576 )) ]] \
        || err "PREFLIGHT FAIL — root fs has ${root_gib:-0} GiB free < ${PREFLIGHT_ROOT_GB} GiB."
    ok "Disk (root /): ${root_gib} GiB free (floor ${PREFLIGHT_ROOT_GB} GiB)"

    # sudo -n dmesg: the wedge watchdog's xe engine-reset monitor reads dmesg
    # non-interactively every cycle (and py-spy captures use sudo -n too).
    # WARN-only, not a hard gate — the watchdog degrades gracefully (it just
    # loses the engine-reset log line), so this must never block a boot.
    if sudo -n dmesg >/dev/null 2>&1; then
        ok "sudo -n dmesg readable (watchdog xe engine-reset monitor armed)"
    else
        warn "sudo -n dmesg NOT readable without a password — the watchdog's xe engine-reset monitor will be blind. Add a sudoers NOPASSWD rule: '<user> ALL=(ALL) NOPASSWD: /usr/bin/dmesg, /usr/local/bin/py-spy' (visudo -f /etc/sudoers.d/b70-watchdog)."
    fi

    ok "Preflight passed."
}

if [[ "$NO_PREFLIGHT" == "true" ]]; then
    red "  === PREFLIGHT SKIPPED (--no-preflight) — you own every gate ==="
    export PREFLIGHT_SKIPPED=1   # the watchdog reads this for the double opt-out
    log_to_run "PREFLIGHT_SKIPPED=1 (--no-preflight)"
else
    exec_preflight
fi

# ---------------------------------------------------------------------------
# Weights: local-tree identity gate (no HF download — tree lives on the NVMe)
# ---------------------------------------------------------------------------
# Missing artifacts -> OFFER the one-time bootstrap. Never silent: interactive
# TTY gets a y/N prompt; --dry-run and non-interactive just print the command
# (fetch-weights.py has its own sha gates and refuses torn trees anyway).
if [[ ! -d "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" || ! -f "$PLE_TABLE_PATH" ]] \
   || ! ls "$MODEL_PATH"/*.safetensors >/dev/null 2>&1; then
    info "Weights bootstrap missing (MODEL_PATH=$MODEL_PATH, PLE_TABLE_PATH=$PLE_TABLE_PATH)."
    info "One-time bootstrap: python3 $SCRIPT_DIR/fetch-weights.py — downloads the pinned AWQ checkpoint, builds the snapshot and the INT8 PLE table (~390 GB, hours; its own env knobs: AWQ_DIR / BF16_DIR / SNAPSHOT_DIR / INT8_PLE_DIR, see the script header)."
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN — would offer to run: python3 $SCRIPT_DIR/fetch-weights.py"
    elif [[ -t 0 ]] && read -r -p "Run it now? [y/N] " ans && [[ "${ans,,}" == "y" ]]; then
        python3 "$SCRIPT_DIR/fetch-weights.py" || err "fetch-weights.py failed — see its output above."
    else
        err "Weights missing. Run when ready: python3 scripts/fetch-weights.py, then re-run start.sh."
    fi
fi
[[ -d "$MODEL_PATH" ]] || err "MODEL_PATH $MODEL_PATH does not exist (fetch-weights.py did not produce it — check AWQ_DIR/SNAPSHOT_DIR env knobs)."
[[ -f "$MODEL_PATH/config.json" ]] || err "config.json missing in $MODEL_PATH — wrong or torn weights tree."
local_shards=$(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l)
[[ "$local_shards" -ge 1 ]] || err "no *.safetensors shards in $MODEL_PATH — wrong or torn weights tree."
[[ -f "$PLE_TABLE_PATH" ]] || err "PLE table missing: $PLE_TABLE_PATH (Qwen4Exp MTP layer needs it)."
# The AWQ snapshot's PLE entry is a symlink into the BF16 tree (cross-check
# reference); it must resolve on the HOST, where the snapshot was built — a
# dangling symlink inside the container mount is a boot-time crash.
if [[ -L "$PLE_TABLE_PATH" ]] && [[ ! -e "$PLE_TABLE_PATH" ]]; then
    err "PLE_TABLE_PATH is a dangling symlink ($PLE_TABLE_PATH -> $(readlink "$PLE_TABLE_PATH")). The snapshot's PLE symlink must resolve on the host; check the BF16 tree mount (PLE_BF16_DIR / HF_DEVAN_MIRROR in .env)."
fi
# HF cache snapshots are symlink forests — they dangle inside the container mount.
# The awq_snapshot.py snapshot itself is ALSO a symlink forest, but its links are
# relative and anchored one level up (../../data-awq/...), which resolve on the
# host AND inside the container when the parent of both trees is mounted
# (HF_DEVAN_MIRROR / the /data parent in production). Distinguish the two: a
# symlinked shard is broken ONLY if its target does not resolve on this host.
if find "$MODEL_PATH" -maxdepth 1 -type l -name '*.safetensors' | grep -q .; then
    broken=0
    while IFS= read -r -d '' link; do
        [[ -e "$link" ]] || { err "Broken shard symlink: $link -> $(readlink "$link") (HF cache snapshots dangle inside the container mount; download with --local-dir — README — Weights)."; broken=1; }
    done < <(find "$MODEL_PATH" -maxdepth 1 -type l -name '*.safetensors' -print0)
    [[ "$broken" == "0" ]] || exit 1
    info "MODEL_PATH is a snapshot symlink forest (relative links resolve on the host — awq_snapshot.py layout)."
fi
ok "Weights tree: $MODEL_PATH (${local_shards} shards)"
ok "PLE table: $PLE_TABLE_PATH ($(du -h "$PLE_TABLE_PATH" | cut -f1))"

# ---------------------------------------------------------------------------
# Image: build if missing (pinned Lumnus commit; verify-overlay gates run at
# build time; the built image is compared against the digest of record).
# MUST come before the XPU gate: the gate docker-runs $IMAGE, so on a fresh
# host without the image the gate would dead-end before the build ever ran.
# ---------------------------------------------------------------------------
docker info >/dev/null 2>&1 || err "docker daemon not reachable (is your user in the docker group?)."
IMAGE_REF="$IMAGE"
if [[ "$DRY_RUN" == "true" ]]; then
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        ok "Image present: $IMAGE_REF"
    else
        info "[DRY-RUN] image $IMAGE not present — a real start would build it (scripts/build-image.sh)"
    fi
else
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        info "Image $IMAGE not present — building from the pinned Lumnus commit (scripts/build-image.sh)..."
        [[ -x "$SCRIPT_DIR/build-image.sh" ]] || err "scripts/build-image.sh missing — cannot build the image of record."
        "$SCRIPT_DIR/build-image.sh" --tag "$IMAGE"
    fi
    docker image inspect "$IMAGE" >/dev/null 2>&1 || err "Image $IMAGE still not present after the build attempt."
    ok "Image present: $IMAGE_REF"
fi

# ---------------------------------------------------------------------------
# PRE-BOOT XPU GATE — trivial triton vector-add must compile AND be exact
# (mandatory standing rule; catches every remaining JIT gap in seconds)
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
    info "[DRY-RUN] would run XPU gate: PASS expected (triton vector-add on one card; not executed)"
elif [[ "$SKIP_XPU_GATE" == "true" ]]; then
    info "XPU gate skipped (--launch restart path)."
elif [[ "$XPU_GATE_DISABLE" == "1" ]]; then
    red "  === XPU GATE DISABLED (XPU_GATE_DISABLE=1) — you own every JIT gap ==="
    log_to_run "XPU_GATE_SKIPPED=1"
else
    info "=== Pre-boot XPU gate (triton vector-add on one card) ==="
    command -v docker >/dev/null 2>&1 || err "docker not found."
    GATE_OUT=$(docker run --rm --name b70-xpu-gate --device /dev/dri \
        -v "$REPO_DIR/docker/gate.py:/gate.py:ro" \
        --entrypoint python3 "$IMAGE" /gate.py 2>&1) \
        || { echo "$GATE_OUT" | tail -5; err "XPU gate failed to run (see above)."; }
    echo "$GATE_OUT" | grep -q "TRITON_XPU_GATE=PASS" \
        || { echo "$GATE_OUT" | tail -5; err "XPU GATE FAIL — vector-add did not compile/verify (see above). Fix before any model boot."; }
    ok "TRITON_XPU_GATE=PASS (compile + exact result)"
    log_to_run "XPU_GATE=PASS"
fi

# ---------------------------------------------------------------------------
# Stop any running instance (validation + preflight already passed).
# When called BY the watchdog (WD_CALLER=1) the container is already dead —
# never kill the calling watchdog, or its bounded-retry counter resets.
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" != "true" && "$WD_CALLER" != "1" && -f "$RUN_DIR/watchdog.pid" ]]; then
    kill "$(cat "$RUN_DIR/watchdog.pid")" 2>/dev/null || true; rm -f "$RUN_DIR/watchdog.pid"
fi
if [[ "$DRY_RUN" != "true" ]] && running "$CONTAINER_NAME"; then
    info "Stopping existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
ENV_HASH=$(cat "$LUMNUS_ENV" "$SERVE_ARGS" | grep -v -E '^HF_TOKEN=' | sort | sha256sum | cut -d' ' -f1)
GIT_DESC=$(git -C "$REPO_DIR" describe --always --dirty 2>/dev/null || echo "no-git")
# Dry-run writes NOTHING under .run/ (manifest + start.log included) — the
# docker line printed at the dry-run exit carries the same identity fields.
if [[ "$DRY_RUN" != "true" ]]; then
log_to_run "launch start (IMAGE=$IMAGE MODEL=$MODEL_PATH TP=$TENSOR_PARALLEL_SIZE MML=$MAX_MODEL_LEN MNS=$MAX_NUM_SEQS KV_OFFLOAD=$KV_OFFLOADING_SIZE git=$GIT_DESC envhash=$ENV_HASH)"
cat > "$RUN_DIR/manifest.json" <<EOF
{
  "kit": "qwen38-flash-next-4xb70-lumnus-prod",
  "model_path": "$MODEL_PATH",
  "image": "$IMAGE",
  "git_describe": "$GIT_DESC",
  "env_hash": "$ENV_HASH",
  "start_iso": "$(date -u +%FT%TZ)",
  "max_model_len": "$MAX_MODEL_LEN",
  "max_num_seqs": "$MAX_NUM_SEQS",
  "kv_offloading_size": "$KV_OFFLOADING_SIZE",
  "served_model_name": "$SERVED_MODEL_NAME"
}
EOF
ok "Manifest: $RUN_DIR/manifest.json"
fi

# ---------------------------------------------------------------------------
# Serve-args splitter: one flag per line, "flag value" split into two array
# entries (values may contain spaces — JSON blobs — and stay intact).
# ---------------------------------------------------------------------------
mapfile -t args < <(grep -v '^#' "$SERVE_ARGS" | sed '/^$/d' | sed 's/^\(--[^ ]*\) \(.*\)$/\1\n\2/')

# ---------------------------------------------------------------------------
# The docker line, assembled as an array so --dry-run prints EXACTLY what a
# real launch runs.
#
# Env pass-through law: boolean/enum engine flags ride the
# ${VAR:+-e VAR=$VAR} pattern — an EMPTY value reaching the container is not
# "unset": oneCCL parses it and dies at worker init. Unset vars are simply
# OMITTED from the docker line. The bulk of the engine env rides --env-file
# ($LUMNUS_ENV); EXTRA_ENGINE_ENV_VARS lists optional overrides passed here.
# ---------------------------------------------------------------------------
DOCKER_RUN=(docker run -d --name "$CONTAINER_NAME"
    --device /dev/dri --group-add render
    -v /dev/dri/by-path:/dev/dri/by-path
    --ipc host --network host --shm-size "${SHM_SIZE:-16g}"
    --env-file "$LUMNUS_ENV")
for v in ${EXTRA_ENGINE_ENV_VARS:-}; do
    # Indirect expansion, then word-split ONLY the var name; the VALUE is
    # appended as one element via "$v=${!v}" quoting — a space-containing
    # value (e.g. OVERRIDE_GENERATION_CONFIG's JSON) stays a single argv item.
    val="${!v:-}"
    if [[ -n "$val" ]]; then
        DOCKER_RUN+=(-e "$v=$val")
    fi
done
# The engine env (lumnus.env) puts EVERY cache path under /cache — HF_HOME,
# TMPDIR, TRITON_CACHE_DIR, VLLM_CACHE_ROOT, XDG_CACHE_HOME,
# B70_PLE_INT8_NVME_NATIVE_DIR. The host cache dir therefore mounts AT /cache
# (production mount of record: <lane>/cache -> /cache), NOT at its own host
# path — mounting $LANE_CACHE:$LANE_CACHE leaves /cache empty inside and the
# offline caches (HF_HUB_OFFLINE=1) miss.
LANE_CACHE_CONTAINER="${LANE_CACHE_CONTAINER:-/cache}"
mkdir -p "$LANE_CACHE"
DOCKER_RUN+=(-v "$LANE_CACHE:$LANE_CACHE_CONTAINER"
    -v "$MODEL_PATH:/data/model:ro"
    -v "$PLE_BF16_DIR:/ple/bf16:ro"
    -v "$PLE_INT8_DIR:/ple/int8:ro"
    -v "$AWQ_ORIG_DIR:/data-awq:ro")
# Optional: the BF16 tree's parent dir (snapshot PLE symlink resolution +
# cross-check tools). Production mounts the parent; set HF_DEVAN_MIRROR to it.
if [[ -n "${HF_DEVAN_MIRROR:-}" ]]; then
    DOCKER_RUN+=(-v "$HF_DEVAN_MIRROR:$HF_DEVAN_MIRROR:ro")
fi
SERVE_CONFIG="${SERVE_CONFIG:-$REPO_DIR/serve-config.json}"
[[ "$SERVE_CONFIG" != /* ]] && SERVE_CONFIG="$REPO_DIR/$SERVE_CONFIG"
# -f gate: without it docker silently creates an EMPTY DIRECTORY at the host
# path and mounts it — the container boot-loops on the missing config.
[[ -f "$SERVE_CONFIG" ]] || err "serve-config.json not found at $SERVE_CONFIG. It ships in the repo root (sha of record c7a2b345927976d911cfd57d1083b71d1a75fee245f61a17b8f126b6717342c8); if genuinely absent, regenerate with: python3 scripts/fetch-weights.py (step 'serve-config' writes + sha-verifies it), or point SERVE_CONFIG in .env at an existing file."
DOCKER_RUN+=(-v "$SERVE_CONFIG:/opt/b70-flashnext/serve-config.json:ro"
    "$IMAGE"
    vllm serve /data/model "${args[@]}" --kv-offloading-size "$KV_OFFLOADING_SIZE")

if [[ "$DRY_RUN" == "true" ]]; then
    info "=== DRY RUN — validation + preflight done (XPU gate not run); printing the docker run line, launching NOTHING ==="
    printf '%q ' "${DOCKER_RUN[@]}" | fold -s -w 100 | sed 's/ $//' | sed 's/$/ \\/' | sed '$ s/ \\$//'
    echo
    ok "Dry run complete — nothing was stopped, spawned, or launched."
    exit 0
fi

# ---------------------------------------------------------------------------
# Wedge watchdog (mandatory; double opt-out to disable). Single-instance
# guard is HOST-WIDE BY DESIGN (pgrep by script name): one rig, one serving
# lane — two watchdogs double-probe, double-restart, and race the pidfile.
# LANE_DIR separates state dirs, not concurrency.
# ---------------------------------------------------------------------------
if [[ "$WD_CALLER" == "1" ]]; then
    info "Restart requested by the running watchdog — leaving it in place (no respawn)."
elif [[ "$WEDGE_WATCHDOG_DISABLE" != "1" || "$NO_PREFLIGHT" != "true" ]]; then
    info "Spawning wedge watchdog (interval=${WEDGE_WATCHDOG_INTERVAL}s, retries=${WEDGE_WATCHDOG_RETRIES})"
    WD_PID=""
    if _existing_wd=$(pgrep -f 'wedge-watchdog\.sh$' 2>/dev/null | head -1) && [[ -n "$_existing_wd" ]]; then
        info "Wedge watchdog already running (pid $_existing_wd) - not spawning a second one"
        WD_PID="$_existing_wd"
    else
        [[ -x "$SCRIPT_DIR/wedge-watchdog.sh" ]] || err "scripts/wedge-watchdog.sh missing or not executable (chmod +x scripts/*.sh)."
        # The watchdog's restart interface: start.sh --launch (gate-skipping
        # fast path). Export it so the watchdog uses THIS start.sh, not its
        # own relative default — the deploy-dir path trap (a restore once
        # launched a nonexistent on-box watchdog path silently).
        export PROD_RESTART_CMD="${PROD_RESTART_CMD:-$SCRIPT_DIR/start.sh --launch}"
        nohup "$SCRIPT_DIR/wedge-watchdog.sh" >> "$RUN_DIR/watchdog.log" 2>&1 &
        WD_PID=$!
        echo "$WD_PID" > "$RUN_DIR/watchdog.pid"
    fi
    ok "Watchdog pid ${WD_PID:-unknown} (log: $RUN_DIR/watchdog.log)"
else
    red "  === WEDGE WATCHDOG DISABLED (double opt-out) — the rig WILL wedge unattended within 2-6 h under load ==="
    log_to_run "watchdog disabled (double opt-out)"
fi

# ---------------------------------------------------------------------------
# Launch — the production line (trial-proven; promoted 2026-10-04)
# (idempotent: rm -f first; a stale container must not block re-creation)
# ---------------------------------------------------------------------------
info "=== Launching Lumnus production stack ($SERVED_MODEL_NAME on :$PORT) ==="
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
"${DOCKER_RUN[@]}"
ok "Container started: $CONTAINER_NAME"
log_to_run "container up (boot $(date -u +%FT%TZ))"

# ---------------------------------------------------------------------------
# Readiness: /v1/models (up to READY_WAIT) AND "Application startup complete"
# (the engine's READY receipt; graphs capture inside torch.compile before it)
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
info "  status: ./scripts/status.sh   logs: ./scripts/start.sh logs   stop: ./scripts/stop.sh"

# ---------------------------------------------------------------------------
# Deviations from the deployed production launcher (clean-room diff notes):
#   1. status/logs/stop delegate to scripts/status.sh / scripts/stop.sh
#      (production keeps them inline in start.sh).
#   2. Single pidfile $LANE_DIR/.run/watchdog.pid (production split its
#      pidfile between start.sh's .run/prod/watchdog.pid and stop.sh's
#      .run/watchdog.pid — a real 2026-10-03 bug where stop.sh left a live
#      watchdog behind; one path removes the class of bug).
#   3. WD_CALLER guard on the EXIT trap and the watchdog block: production
#      start.sh kills .run/prod/watchdog.pid on ANY non-zero exit, including
#      when the watchdog itself invoked `start.sh --launch` — that kills the
#      watchdog mid bounded-retry. The guard preserves the retry counter.
#   4. Preflight adds docker buildx (hard gate) and sudo -n dmesg readability
#      (WARN-only) — the production start.sh gates neither.
#   5. Preflight adds the PLE NVMe disk floor (PREFLIGHT_PLE_NVME_GB, hard) —
#      the INT8 table is served from that NVMe, so it is a hot path, not
#      report-only like the weights mount.
#   6. --dry-run flag: prints the assembled docker run line, launches nothing.
#   7. All host paths parameterized via .env (production hardcodes
#      /dev/dri/by-path, /data/model, /ple/*, /data-awq as CONTAINER-side
#      paths only — those stay fixed; the HOST side is what .env owns).
#   8. Image build-if-missing via scripts/build-image.sh (pinned Lumnus commit
#      + digest comparison); weights bootstrap pointer to scripts/fetch-weights.py.
# ============================================================================

# 

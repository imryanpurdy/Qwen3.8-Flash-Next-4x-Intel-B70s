#!/usr/bin/env bash
# ============================================================================
# status.sh — one-shot production status: container, watchdog, restarts,
#             xe engine resets, KV pool, /metrics finish-reason counters.
# Read-only: touches nothing, restarts nothing, safe at any time.
# Exit code: 0 = READY, 1 = degraded/down (usable from cron/monitors).
# ============================================================================
set -uo pipefail   # NOTE: no -e — every section is best-effort by design

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; }

[[ -f .env ]] || { err ".env not found. Run:  cp .env.example .env"; exit 1; }
# shellcheck source=.env
source .env
CONTAINER_NAME="${CONTAINER_NAME:-b70-lumnus-prod}"
PORT="${PORT:-8022}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-256k}"
LANE_DIR="${LANE_DIR:-$PWD}"
RUN_DIR="$LANE_DIR/.run"

DEGRADED=0

# ---------------------------------------------------------------------------
# 1. Container state + RestartCount
# ---------------------------------------------------------------------------
CSTATE=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || true)
if [[ -z "$CSTATE" ]]; then
    err "Container $CONTAINER_NAME does not exist. Start: ./scripts/start.sh"
    DEGRADED=1
elif [[ "$CSTATE" == "running" ]]; then
    RESTARTS=$(docker inspect -f '{{.RestartCount}}' "$CONTAINER_NAME" 2>/dev/null || echo "?")
    UPTIME=$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER_NAME" 2>/dev/null || echo "?")
    IMAGE_ID=$(docker inspect -f '{{.Config.Image}}' "$CONTAINER_NAME" 2>/dev/null || echo "?")
    ok "Container: running since $UPTIME (image: $IMAGE_ID)"
    if [[ "$RESTARTS" != "0" ]]; then
        warn "RestartCount=$RESTARTS — the container has restarted since creation (docker-level crash loop?)."
        DEGRADED=1
    else
        ok "RestartCount: 0"
    fi
else
    err "Container: $CSTATE (expected running). Last logs:"
    docker logs --tail 10 "$CONTAINER_NAME" 2>&1 | sed 's/^/        /' || true
    DEGRADED=1
fi

# ---------------------------------------------------------------------------
# 2. Watchdog alive?
# ---------------------------------------------------------------------------
if [[ -f "$RUN_DIR/watchdog.pid" ]] && kill -0 "$(cat "$RUN_DIR/watchdog.pid")" 2>/dev/null; then
    ok "Wedge watchdog: alive (pid $(cat "$RUN_DIR/watchdog.pid"), log: $RUN_DIR/watchdog.log)"
else
    warn "Wedge watchdog: NOT running (pidfile absent or stale) — the rig wedges unattended within 2-6 h under load. Restart via ./scripts/start.sh."
    DEGRADED=1
fi
# Watchdog's own recent verdict, if any (last 3 sample lines).
if [[ -f "$RUN_DIR/watchdog.log" ]]; then
    info "Watchdog log (last 3 lines):"
    tail -n 3 "$RUN_DIR/watchdog.log" 2>/dev/null | sed 's/^/        /' || true
fi

# ---------------------------------------------------------------------------
# 3. API: /v1/models + startup receipt
# ---------------------------------------------------------------------------
MODELS_JSON=$(curl -fsS -m 10 "http://localhost:$PORT/v1/models" 2>/dev/null || true)
if [[ -n "$MODELS_JSON" && "$MODELS_JSON" == *"$SERVED_MODEL_NAME"* ]]; then
    ok "API: /v1/models answers with '$SERVED_MODEL_NAME' on :$PORT"
else
    err "API: /v1/models not answering with '$SERVED_MODEL_NAME' on :$PORT (still loading, or down)."
    DEGRADED=1
fi
if [[ -n "$CSTATE" ]] && docker logs "$CONTAINER_NAME" 2>&1 | grep -q "Application startup complete"; then
    ok "Startup receipt: 'Application startup complete' present."
elif [[ "$CSTATE" == "running" ]]; then
    warn "Startup receipt: 'Application startup complete' NOT yet in logs (torch.compile/graphs still capturing?)."
fi

# ---------------------------------------------------------------------------
# 4. xe engine resets (dmesg) — the wedge fingerprint the watchdog watches.
#    sudo -n dmesg (NOPASSWD rule from host-setup); degrade to plain dmesg.
# ---------------------------------------------------------------------------
DMESG=$(sudo -n dmesg 2>/dev/null || dmesg 2>/dev/null || true)
if [[ -z "$DMESG" ]]; then
    warn "xe resets: dmesg not readable (add sudoers NOPASSWD: /usr/bin/dmesg — see start.sh preflight note)."
else
    RESETS=$(printf '%s\n' "$DMESG" | grep -ciE 'xe .*engine reset|GT .*reset' || true)
    if [[ "${RESETS:-0}" -gt 0 ]]; then
        warn "xe engine resets in dmesg: $RESETS occurrence(s) — wedge fingerprint. Recent:"
        printf '%s\n' "$DMESG" | grep -iE 'xe .*engine reset|GT .*reset' | tail -n 3 | sed 's/^/        /'
        DEGRADED=1
    else
        ok "xe engine resets: none in dmesg."
    fi
fi

# ---------------------------------------------------------------------------
# 5. KV pool size (native offload tier) — from the container's startup logs.
#    The engine logs the GPU KV cache + offloaded KV cache sizes at boot;
#    KV_OFFLOADING_SIZE (GiB total over TP ranks) must show up as the offload
#    pool, not be silently dropped.
# ---------------------------------------------------------------------------
if [[ -n "$CSTATE" ]]; then
    KVLINE=$(docker logs "$CONTAINER_NAME" 2>&1 | grep -iE 'KV cache size|GPU KV cache size|CPU KV cache|Offloading.*KV|kv_offloading' | tail -n 4 || true)
    if [[ -n "$KVLINE" ]]; then
        info "KV pool (from container logs; KV_OFFLOADING_SIZE=${KV_OFFLOADING_SIZE:-?} GiB total over TP ranks):"
        printf '%s\n' "$KVLINE" | sed 's/^/        /'
    else
        warn "KV pool: no KV-cache size lines found in container logs yet."
    fi
fi

# ---------------------------------------------------------------------------
# 6. /metrics — the load-aware watchdog's signal source (it samples these
#    twice LOAD_SAMPLE_S apart; here we print one snapshot): token counters,
#    in-flight requests, and finish-reason counters.
# ---------------------------------------------------------------------------
METRICS=$(curl -fsS -m 10 "http://localhost:$PORT/metrics" 2>/dev/null || true)
if [[ -n "$METRICS" ]]; then
    info "Token counters + in-flight requests (/metrics):"
    printf '%s\n' "$METRICS" | awk '
        /^vllm:num_requests_running/  {print "        " $0}
        /^vllm:num_requests_waiting/  {print "        " $0}
        /^vllm:generation_tokens_total/ {print "        " $0}
        /^vllm:prompt_tokens_total/   {print "        " $0}'
    info "Finish-reason counters (/metrics):"
    printf '%s\n' "$METRICS" | grep -E '^vllm:(e2e_request_latency|request_success)_seconds_count' | sed 's/^/        /' || true
    printf '%s\n' "$METRICS" | grep -E '^vllm:.*finish_reason' | sed 's/^/        /' || \
        warn "  (no finish_reason-labelled series on this engine build — the counters above are the watchdog's progress signal)"
else
    warn "Metrics: /metrics not answering (API down or still loading)."
fi

echo
if [[ "$DEGRADED" -eq 0 ]]; then
    ok "STATUS: READY — container, watchdog, API, dmesg all clean."
    exit 0
else
    err "STATUS: DEGRADED — see [ERR ]/[WARN ] lines above."
    exit 1
fi

# 

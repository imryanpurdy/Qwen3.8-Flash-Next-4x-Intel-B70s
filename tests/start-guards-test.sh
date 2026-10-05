#!/usr/bin/env bash
# ============================================================================
# start-guards-test.sh — GPU-free proof of the 2026-10-05 guard law:
#
#   G1  DRY_RUN as an environment variable is a HARD ERROR (the dry-run
#       switch is the --dry-run flag ONLY — a silently-ignored env var once
#       launched production).
#   G2  start.sh refuses to stop a RUNNING container without --replace.
#   G3  start.sh --replace (and the watchdog/systemd restart path) is allowed
#       to stop a running container — exercised via a stub docker.
#
# Needs no GPU, no real docker: a fake `docker` on PATH stubs `ps`/`inspect`.
# The real start.sh is invoked with a stub .env so validation passes far
# enough to reach the guards. Runs in ~5 s. Safe on a live rig: it never
# touches the real container name (stub .env uses a different CONTAINER_NAME)
# and never runs docker for real.
#
# Runnable right after a fresh clone: bash tests/start-guards-test.sh
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

FAIL=0
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; FAIL=1; }

SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/startguards.XXXXXX")"
trap 'rm -rf "$SANDBOX"' EXIT

# --- Stub docker: reports a container named by $STUB_CONTAINER as running ---
mkdir -p "$SANDBOX/bin"
cat > "$SANDBOX/bin/docker" <<'EOF'
#!/usr/bin/env bash
# stub docker: answers the exact calls start.sh makes on the guard path
# (ps / info / image inspect); every MUTATING call (rm, run, build...) is
# recorded and refused so a bug can never touch the host.
case "$1 $2" in
  "ps --format")  if [[ -n "${STUB_CONTAINER:-}" ]]; then echo "$STUB_CONTAINER"; fi; exit 0 ;;
  "info >/dev/null"|"info ") case "$1" in info) exit 0;; esac ;;
esac
case "$1" in
  info) exit 0 ;;
  buildx) exit 0 ;;
  image)
    # docker image inspect <ref> -> "present"
    if [[ "$2" == "inspect" ]]; then exit 0; fi
    exit 0 ;;
  ps)
    if [[ -n "${STUB_CONTAINER:-}" ]]; then echo "$STUB_CONTAINER"; fi
    exit 0 ;;
  rm|run|build)
    echo "STUB-DOCKER-MUTATE: $*" >> "${SANDBOX:?}/docker-calls.log"
    exit 99 ;;
  *)
    echo "STUB-DOCKER-ILLEGAL: $*" >> "${SANDBOX:?}/docker-calls.log"
    exit 99 ;;
esac
EOF
chmod +x "$SANDBOX/bin/docker"
export PATH="$SANDBOX/bin:$PATH"

# --- Stub lane .env (name does NOT match any real container) -----------------
STUB_CONTAINER="startguards-stub-container"
export STUB_CONTAINER
cat > "$SANDBOX/.env" <<EOF
MODEL_PATH=$SANDBOX/model
PLE_TABLE_PATH=$SANDBOX/ple.bin
SERVED_MODEL_NAME=startguards-stub
PORT=18999
IMAGE=startguards-stub-image
CONTAINER_NAME=$STUB_CONTAINER
LUMNUS_ENV=$SANDBOX/lumnus.env
SERVE_ARGS=$SANDBOX/serve-args
KV_OFFLOADING_SIZE=8
LANE_DIR=$SANDBOX
PLE_BF16_DIR=$SANDBOX/bf16
PLE_INT8_DIR=$SANDBOX/int8
AWQ_ORIG_DIR=$SANDBOX/awq
LANE_CACHE=$SANDBOX/cache
PREFLIGHT_XPU_COUNT=1
EOF
cat > "$SANDBOX/lumnus.env" <<'EOF'
STUB=1
EOF
cat > "$SANDBOX/serve-args" <<'EOF'
--tensor-parallel-size 4
--max-model-len 1024
--max-num-seqs 8
EOF
mkdir -p "$SANDBOX/model" "$SANDBOX/cache"
touch "$SANDBOX/ple.bin" "$SANDBOX/model/config.json" "$SANDBOX/model/model-00001.safetensors"
# serve-config.json + XPU-gate skip: the gate needs a real GPU image; stub the
# gate off via the double opt-out is WRONG (that also disables the watchdog).
# Instead the stub docker refuses `run` (the gate's docker run) — for --dry-run
# the gate is skipped by design; for G2/G3/G4 use XPU_GATE_DISABLE=1 scoped to
# the call, or rely on... (see run_start: XPU_GATE_DISABLE passed per call).

# start.sh resolves its lane root from its own BASH_SOURCE path, so the stub
# lane is a mirror of the repo's scripts/ tree: scripts/ + serve-config.json
# symlinked into the sandbox; .env and runtime state are sandbox-local.
mkdir -p "$SANDBOX/scripts"
ln -sf "$SCRIPT_DIR"/scripts/*.sh "$SANDBOX/scripts/" 2>/dev/null
ln -sf "$SCRIPT_DIR/serve-config.json" "$SANDBOX/serve-config.json"
run_start() { ( cd "$SANDBOX" && SANDBOX="$SANDBOX" XPU_GATE_DISABLE="${XPU_GATE_DISABLE:-0}" bash "$SANDBOX/scripts/start.sh" "$@" ) 2>&1; }

# ============================================================================
# G1: DRY_RUN env var is a hard error (before ANY validation runs)
# ============================================================================
echo "== G1: DRY_RUN env var must hard-error =="
out="$(DRY_RUN=true run_start --dry-run 2>&1)"; rc=$?
unset DRY_RUN   # env-scope hygiene: a var assigned before a function call can persist
if [[ $rc -ne 0 ]] && grep -q "DRY_RUN environment variable detected" <<<"$out"; then
    pass "G1: DRY_RUN env var refused with the explicit error"
else
    fail "G1: DRY_RUN env var not refused (rc=$rc): $(echo "$out" | tail -2)"
fi
# The guard must fire even with NO args (the incident's exact invocation):
out="$(DRY_RUN=true run_start 2>&1)"; rc=$?
unset DRY_RUN
if [[ $rc -ne 0 ]] && grep -q "DRY_RUN environment variable detected" <<<"$out"; then
    pass "G1: DRY_RUN env var refused on a bare start too"
else
    fail "G1: bare start with DRY_RUN env var did NOT fail (rc=$rc)"
fi

# ============================================================================
# G2: running container + no --replace => REFUSED (never stopped)
# ============================================================================
echo "== G2: plain start refuses to touch a live container =="
: > "$SANDBOX/docker-calls.log"
out="$(WEDGE_WATCHDOG_DISABLE=1 run_start --dry-run --no-preflight 2>&1)"; rc=$?
if [[ $rc -eq 0 ]] && grep -q "DRY RUN" <<<"$out"; then
    pass "G2: --dry-run still works (validation path unaffected)"
else
    fail "G2: --dry-run broke: rc=$rc $(echo "$out" | tail -3)"
fi
: > "$SANDBOX/docker-calls.log"
out="$(WEDGE_WATCHDOG_DISABLE=1 XPU_GATE_DISABLE=1 run_start --no-preflight 2>&1)"; rc=$?
if [[ $rc -ne 0 ]] && grep -q "REFUSING: container" <<<"$out"; then
    pass "G2: plain start refused while container is running"
else
    fail "G2: plain start did not refuse (rc=$rc): $(echo "$out" | tail -3)"
fi
# And the refusal must NOT have stopped anything: no rm in docker calls, and
# the stub container is still reported running by the stub.
if [[ -f "$SANDBOX/docker-calls.log" ]] && grep -q "rm" "$SANDBOX/docker-calls.log" 2>/dev/null; then
    fail "G2: docker rm was called during the refused start"
else
    pass "G2: no docker rm issued during the refusal"
fi

# ============================================================================
# G3: --replace explicitly allows the stop (docker rm recorded via stub)
# ============================================================================
echo "== G3: --replace allows stopping the running container =="
# NOTE: this test proves the GUARD only — it uses a stub image that cannot
# boot, so the run fails later (at launch/ready) BY DESIGN. The assertion is
# that the stop path ran, not that the launch succeeded.
: > "$SANDBOX/docker-calls.log"
out="$(WEDGE_WATCHDOG_DISABLE=1 XPU_GATE_DISABLE=1 run_start --no-preflight --replace 2>&1)"; rc=$?
if grep -q "STUB-DOCKER-MUTATE: rm -f" "$SANDBOX/docker-calls.log" 2>/dev/null; then
    pass "G3: --replace reached docker rm (stub recorded the call; launch stub-refused as designed)"
else
    fail "G3: --replace did not reach the stop path: $(echo "$out" | tail -3)"
fi

# ============================================================================
# G4 (bonus): the same refusals must not kill a live watchdog
# (the EXIT-trap fix: a failed/refused start must never TERM a running
# watchdog that belongs to the live lane)
# ============================================================================
echo "== G4: refused start leaves the live watchdog alive =="
# Simulate a watchdog: a sleeping process recorded in the lane's watchdog.pid
sleep 60 & WD_PID=$!
mkdir -p "$SANDBOX/.run"; echo "$WD_PID" > "$SANDBOX/.run/watchdog.pid"
out="$(WEDGE_WATCHDOG_DISABLE=1 XPU_GATE_DISABLE=1 run_start --no-preflight 2>&1)"; rc=$?
if kill -0 "$WD_PID" 2>/dev/null; then
    pass "G4: watchdog survived the refused start"
else
    fail "G4: watchdog was killed by the refused start"
fi
kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null

# ============================================================================
# G5 (review D1): `restart` carries --replace by definition — the documented
# stop+start path must reach the stop path (not refuse) on a live container.
# The launch itself stub-refuses by design (stub image cannot boot).
# ============================================================================
echo "== G5: restart implies --replace =="
: > "$SANDBOX/docker-calls.log"
out="$(WEDGE_WATCHDOG_DISABLE=1 XPU_GATE_DISABLE=1 run_start --no-preflight restart 2>&1)"; rc=$?
if grep -q "STUB-DOCKER-MUTATE: rm -f" "$SANDBOX/docker-calls.log" 2>/dev/null; then
    pass "G5: restart reached the stop path (docker rm recorded)"
else
    fail "G5: restart did not reach the stop path (rc=$rc): $(echo "$out" | tail -3)"
fi
# Arg-mapping proof: `restart` must map to CMD=start + REPLACE=true (parse-level
# check; the full-gate path needs real hardware for preflight, out of stub scope)
mapline="$(grep -A1 'restart) CMD=' "$SCRIPT_DIR/scripts/start.sh" | head -2 | tr '\n' ' ')"
if [[ "$mapline" == *'restart) CMD="start" REPLACE=true'* ]]; then
    pass "G5: restart maps to start+REPLACE=true in the arg parser"
else
    fail "G5: restart arg mapping wrong: $mapline"
fi
# and restart must NOT print the refusal
if grep -q "REFUSING: container" <<<"$out"; then
    fail "G5: restart wrongly refused"
else
    pass "G5: restart not refused"
fi

# ============================================================================
if [[ "$FAIL" == "0" ]]; then
    echo "START-GUARDS: PASS"
else
    echo "START-GUARDS: FAIL"
    exit 1
fi

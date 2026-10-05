# Rollback runbook — production lane → es-lane

Direction: **prod-lumnus lane → es-lane rollback lane**. The sentinel is
alert-only; every rollback is an operator decision made after reading the
alert trigger and the watchdog evidence.

## The two lanes

| | Production | Rollback |
|---|---|---|
| Deploy dir | `<lane>/` (prod-lumnus) | `rollback/devan-fork/` (in-repo es-lane kit, deploy of record 2026-09-23) |
| Container | `b70-lumnus-prod` | `es-lane` |
| Engine | Lumnus b70-flash-next (vLLM v0.30.0 + patches 0001–0019) | vLLM fork `devan-carlin/vllm@xpu-qwen4exp` (a69fba21) on `intel/omix:0.4.0-devel-ubuntu24.04` |
| Image | `b70-lumnus-trial:v1` (local build of record) | **pinned by digest**: `es-lane@sha256:15a806fc7367a44f6ab66d42e9f1b237fbb7431f4e913eb9ea197505f8d8417a` |
| Checkpoint | wtdcode AWQ W4A16 (snapshot) | devan-carlin W4A16 (`40b8f18d`) — **not** the baseline checkpoint: the pre-Lumnus baseline (README "Results") was measured on this engine running the **wtdcode AWQ** checkpoint (`STATUS-20261004.md` "What's in production", KV pool fingerprint 845,862 = AWQ) |
| Line | TP4+EP, MML 262144, MNS 32, **bf16 KV** (no `--kv-cache-dtype` flag; vLLM default follows `--dtype bfloat16`), 64 GiB CPU KV-offload | TP4+EP, MML 262144, **MNS 4** (deployed es-lane launcher line 117; the kit's `.env.example` default is 32 — the kit's own soak-validated point, not what the deployed rollback lane runs), **fp8 KV** (explicit `--kv-cache-dtype fp8`), **no KV-offload tier**, parsers qwen3/qwen3_xml |
| Sampler | `--override-generation-config` in serve args (see production-lane.md) | `OVERRIDE_GENERATION_CONFIG` in `.env` (same pinned set) |
| Port / served name | 8022 / `qwen-256k` | 8022 / `qwen-256k` — **identical**, so no client/gateway reconfiguration is needed on rollback; only the container swaps |

**PLE on the rollback engine:** it never executes. The devan fork's PLE table
path was absent on the rig and the guard short-circuited (no `FileNotFoundError`
in the boot log; forward never reached `_ensure_table` — only the capture-legal
early-return branch ran). The rollback line is a dense fallback, not a PLE
line; quality/latency deltas vs the Lumnus line include PLE being live.

The rollback lane is a complete, self-contained stack with its own `start.sh`
(same design rules: validate → preflight → weights gate → XPU gate → image
gate → launch → READY gate; mandatory watchdog; `stop`/`restart`/`status`/`logs`
subcommands) and its own `.env` with the image **pinned by digest** — a rollback
never floats to a different image. The kit lives in-repo at
`rollback/devan-fork/` (`start.sh`, `stop.sh`, `.env.example` — `cp
.env.example .env` there; the repo-root `.env` belongs to the production lane).

## Rollback criteria

Roll back when any of these holds (operator judgment, but these are the
standing triggers):

1. **Watchdog exhausted its retries** — 3 bounded restarts failed and the
   watchdog exited loudly (`.run/prod/watchdog.log`); the lane is down.
2. **Repeated wedges** — engine recovers but re-wedges faster than it serves
   (check the sentinel log cadence and `xe engine resets` deltas in
   `.run/prod/watchdog.log`).
3. **Post-restart verification fails** — the smoke tier or full gate
   (`verify.md`) fails twice on the production lane.
4. **Quality regression** — repetition degeneration, wrong tool calls, or
   recall regressions that a sampler/flag flip does not fix within the outage
   budget (~20 min rule: if the fix will take longer, roll back first, debug
   offline).
5. **Sentinel flag present** (`ALERT_NEEDS_ROLLBACK.flag`) **and** the trigger
   (dmesg reset delta / RestartCount bump / API-down-without-boot) is
   confirmed and not transient.

## Procedure

Run everything from the deploy directories themselves — never from a stale
copy of the tree.

```bash
# 0. (planned only) drain: stop routing new work; in-flight children die on
#    the swap; Hermes children retry ~6 min — see restart discipline.

# 1. Stop the production lane (watchdog FIRST, then container — stop.sh
#    does it in this order):
<lane>/stop.sh
#    Confirm nothing is left holding the lane:
docker ps --format '{{.Names}}' | grep -E 'b70-lumnus-prod|es-lane'   # expect: none
pgrep -fa wedge-watchdog.sh                                          # expect: none

# 2. Start the rollback lane (full gate path — preflight, weights, XPU gate,
#    digest-pinned image, watchdog, READY poll):
cd rollback/devan-fork && ./start.sh    # (repo clone; or the deployed lane dir)
#    Boot is ~4–5 min; READY is printed only after /v1/models answers AND
#    "Application startup complete" appears in the container logs.

# 3. Verify (see verify.md smoke tier):
curl -s http://127.0.0.1:8022/v1/models | grep -o '"id":"[^"]*"'   # expect: qwen-256k
# 1-token smoke completion:
curl -s http://127.0.0.1:8022/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-256k","messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":1}' \
  | head -c 400
# then the strict tool-call check (scripts/toolcall.py, small count) — verify.md.

# 4. Confirm the rollback watchdog is up (single instance):
rollback/devan-fork/start.sh status
```

**Do not run both lanes at once** — they share port 8022 and the watchdog
single-instance guard (`pgrep -f wedge-watchdog.sh`) is host-wide by design.

## After rollback

- Record it: date/time, trigger, evidence pointers (wedge log, sentinel log
  line), rollback READY time, verify results.
- Re-point monitoring: the sentinel's container name and dmesg baseline are
  lane-agnostic (it watches `RestartCount` + dmesg + the API), but confirm it
  goes quiet on the next tick.
- The production lane stays parked for post-mortem; its `.run/prod/wedge-*.log`
  py-spy captures are the evidence. Do not delete the prod deploy dir until
  the failure mode is understood.
- Promotion back forward requires the full gate (`verify.md`) on the
  production lane, off the serving port, before traffic moves again.

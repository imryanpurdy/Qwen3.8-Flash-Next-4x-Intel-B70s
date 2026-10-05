# Rollback runbook — production lane → pre-Lumnus devan-fork lane

Direction: **prod-lumnus lane → devan-fork rollback lane**. The sentinel is
alert-only; every rollback is an operator decision made after reading the
alert trigger and the watchdog evidence.

## The rollback of record: last-known-good production (devan fork + AWQ, MNS 32)

The rollback lane is the **last-known-good production stack** — the lane that
served before the Lumnus promotion and measured the production baseline
(32×600 = **1,015 tok/s**; full baseline: 1,015 / 622, 49.3 tok/s
single-stream, 95/100 recall — README "Results"). It is a complete,
self-contained kit: devan-carlin `vllm@xpu-qwen4exp` (a69fba21) on the
digest-pinned es-lane image, running the **wtdcode AWQ** checkpoint at
**MNS 32**.

**Its actual deployed launcher lives on the rig at `~/mns32-lane/`** (kit
git `743ecc5`, "start.sh: gate MNS at KV knee 32…"). The in-repo kit at
`rollback/devan-fork/` is that same launcher family, sanitized for the repo;
its active `.env` of record on the rig pins:

- `MODEL_PATH=/data-awq/Qwen3.8-Flash-Next-AWQ-W4A16` (wtdcode AWQ)
- `MAX_NUM_SEQS=32` (MNS 32 — the lane that measured 1,015 @32)
- `KV_CACHE_DTYPE=fp8` (explicit `--kv-cache-dtype fp8`; **bf16 attention
  dtype**, `DTYPE=bfloat16`)
- Image: `es-lane@sha256:15a806fc7367a44f6ab66d42e9f1b237fbb7431f4e913eb9ea197505f8d8417a`
  (digest-pinned; a rollback never floats)
- `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1`, `ZE_AFFINITY_MASK=0,1,2,3`,
  sampler pin `temp 0.7 / top_p 0.80 / top_k 20 / presence 1.5`,
  `EXTRA_VLLM_ARGS='--cudagraph-capture-sizes 1 2 4 8 16 32 64'`

Historical note (footnote): an earlier rollback lane ran the **devan W4A16**
checkpoint (`/data/hf-devan/...`) at **MNS 4** via
`~/es-lane-launch/start-qwen-256k-vllm.sh` (line 117: `--max-num-seqs 4`,
`--kv-cache-dtype fp8`). That MNS-4 launcher is the older lane — the es-lane
that predates the MNS-32 soak (ladder: 2→95, 4→180, 8→335, 16→629 tok/s; 32
is the KV-math knee at MML 262144). The last-known-good MNS-32 AWQ lane
supersedes it as the rollback of record.

### The two lanes

| | Production | Rollback |
|---|---|---|
| Deploy dir | the lane root (`scripts/…`; e.g. `~/qwen-prod`) | `~/mns32-lane/` on the rig (in-repo kit: `rollback/devan-fork/`) |
| Container | `b70-lumnus-prod` | `es-lane` |
| Engine | Lumnus b70-flash-next (vLLM v0.30.0 + patches 0001–0019) | vLLM fork `devan-carlin/vllm@xpu-qwen4exp` (a69fba21) on `intel/omix:0.4.0-devel-ubuntu24.04` |
| Image | `b70-lumnus-trial:v1` (local build of record) | **pinned by digest**: `es-lane@sha256:15a806fc7367a44f6ab66d42e9f1b237fbb7431f4e913eb9ea197505f8d8417a` |
| Checkpoint | wtdcode AWQ W4A16 (snapshot) | **wtdcode AWQ W4A16** (`/data-awq/Qwen3.8-Flash-Next-AWQ-W4A16`) — the same checkpoint the production baseline was measured on |
| Line | TP4+EP, MML 262144, MNS 32, **bf16 KV** (no `--kv-cache-dtype` flag; vLLM default follows `--dtype bfloat16`), 64 GiB CPU KV-offload | TP4+EP, MML 262144, **MNS 32**, **fp8 KV** (explicit `--kv-cache-dtype fp8`), **no KV-offload tier**, no PLE hot path (below), decode graphs `1 2 4 8 16 32 64`, parsers qwen3/qwen3_xml |
| Throughput of record | 32×600 median 1,118.4 tok/s (README Results) | 32×600 = 1,015 tok/s (the production baseline line); kit-era soak 1,038.3 |
| Sampler | `--override-generation-config` in serve args (see production-lane.md) | `OVERRIDE_GENERATION_CONFIG` in `.env` (same pinned set) |
| Port / served name | 8022 / `qwen-256k` | 8022 / `qwen-256k` — **identical**, so no client/gateway reconfiguration is needed on rollback; only the container swaps |

**PLE on the rollback engine:** it never executes. The AWQ snapshot's PLE
symlink targets the BF16 tree (`/srv/hf-devan/...`), and the devan fork
short-circuits its PLE path when the table is absent from its expected
location — the rollback line is a dense fallback, not a PLE line. The kit's
weights gate requires a *resolvable* PLE table path; point `PLE_TABLE_PATH`
at the BF16 tree's table
(`/srv/hf-devan/Qwen3.8-Flash-Next-W4A16/ple_table_qwen4exp.pt`) exactly as
the deployed `.env` does. Quality/latency deltas vs the Lumnus line include
PLE being live.

### Kit alignment with the deployed launcher

The in-repo kit (`rollback/devan-fork/`) is the deployed `~/mns32-lane/`
launcher, with three deliberate deltas:

1. `REPO_ROOT` resolution: the kit reuses the repo's `docker/gate.py` and
   `scripts/wedge-watchdog.sh` (the deployed kit ships its own copies, e.g.
   `wedge-watchdog-eslane.sh`). The watchdog restart command
   (`ESLANE_RESTART_CMD`, accepted by the shared watchdog as a deprecated
   alias of `PROD_RESTART_CMD`) targets the kit's own `start.sh --launch` —
   a rollback restart never launches the production lane.
2. `.env.example` model path is a placeholder (`/path/to/...`) — the rig's
   real path is `/data-awq/Qwen3.8-Flash-Next-AWQ-W4A16`. On the rig, the
   deployed `~/mns32-lane/.env` is the config of record.
3. The kit carries the watchdog single-instance guard and the
   pass-through-law `${VAR:+-e VAR=$VAR}` pattern from the Oct-4 overhaul
   (the deployed launcher predates parts of it).

Everything else — preflight gates (kernel 6.17.0-1010-intel, GuC hash,
iommu=off, RAM/swap floors), XPU gate, image digest pin, READY-on-receipt
gate, MNS 32 KV-knee gate — matches the deployed launcher.

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
#    digest-pinned image, watchdog, READY poll). On the rig, the deployed
#    launcher IS the config of record:
cd ~/mns32-lane && ./start.sh          # (rig; or rollback/devan-fork from a repo clone)
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
~/mns32-lane/start.sh status
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

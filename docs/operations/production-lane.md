# Production lane — Lumnus b70-flash-next (operational runbook)

The production serving lane for Qwen3.8-Flash-Next W4A16 on 4x Intel Arc Pro B70
(TP4+EP, 262144 context, served name `qwen-256k`, port 8022). Promoted from the
trial lane on **2026-10-04** after the full gate (see STATUS promotion record).

All paths below are written relative to the **deploy directory** (the on-host
lane checkout, referenced here as `<lane>/` — e.g. `prod-lumnus/`). Never invoke
scripts from any other copy of the tree: the deploy directory carries its own
`scripts/wedge-watchdog.sh`, and a restore launched from a stale path has already
happened once (ledger-corrected 2026-10-04).

## Lane layout

| Piece | Path (relative to `<lane>/`) | Role |
|---|---|---|
| Lane entry point | `start.sh` | validate → preflight → weights gate → XPU gate → image gate → launch → ready-poll; also `stop` / `restart` / `status` / `logs` subcommands |
| Graceful stop | `stop.sh` | watchdog first (TERM, then KILL after ~5 s), then `docker rm -f` the container |
| Control vars | `.env` | model paths, served name, port, image tag, watchdog knobs, preflight floors |
| Engine env | `lumnus.env` | passed to the container via `docker run --env-file` (Level-Zero/oneCCL pins, PLE INT8-NVMe switches, chat defaults) |
| Serve flags | `serve-args` (repo root) | one flag per line; `--kv-offloading-size` appended by `start.sh` |
| Model config | `serve-config.json` | bind-mounted architecture config (qwen4_exp) |
| Watchdog | `scripts/wedge-watchdog.sh` | mandatory wedge supervisor (see contract below) |
| Sentinel | alert-only degradation sentinel (operator-scheduled; deployed on the rig, not shipped in-repo) |
| Runtime state | `.run/` | `start.log`, `manifest.json`, `watchdog.pid`, `watchdog.log`, `wedge-<ts>.log` |

Engine of record: Lumnus b70-flash-next (vLLM v0.30.0 + Lumnus patch series
0001–0019 — sub-lettered, 21 files; see `docs/engine/PROVENANCE.md`), image `b70-lumnus-trial:v1` (local build of record), wtdcode AWQ
W4A16 checkpoint, INT8 PLE n-gram table served from NVMe (patch 0013 native
reader), TP4 + expert-parallel, MML 262144, MNS 32, 64 GiB CPU KV-offload tier.
Rollback lane (separate deploy directory, see `rollback.md`): the last-known-good
pre-Lumnus production stack — devan fork + wtdcode AWQ checkpoint at MNS 32
(rig: `~/mns32-lane/`), image pinned by digest.

## One-command start contract

`<lane>/start.sh` (default command `start`) runs this sequence, in order —
**every validation happens before any running service is touched**:

1. **Knob validation** — `.env` required vars present; MML/MNS/TP parsed out of
   `serve-args` and gated: TP must be 4 (2 KV heads — TP6 impossible),
   MML ≤ 262144 (validated ceiling), MNS ≤ 32 (KV-math knee at MML 262144;
   32 is the soak-validated operating point: 3112/3112, 0 errors, 60 min).
2. **Preflight** — exactly 4 XPUs visible; ≥100 GiB available RAM; ≥64 GiB swap
   on; kernel = the platform of record (`6.17.0-1010-intel`); GuC firmware
   `bmg_guc_70.bin` sha256 match (70.65); `iommu=off` on the kernel cmdline;
   weights-mount ≥2 GiB free (hard) / root ≥40 GiB. `--no-preflight` skips with
   a loud banner and a `PREFLIGHT_SKIPPED=1` log line — you own every gate.
   A `DRY_RUN` environment variable hard-errors on EVERY subcommand (the
   dry-run switch is the `--dry-run` flag only) — `unset DRY_RUN` or use
   `scripts/stop.sh` directly if a stray export blocks a graceful stop.
3. **Weights identity gate** — the local tree must exist with `config.json` and
   ≥1 `.safetensors` shard; the PLE table file must exist. Never launch into a
   wrong or torn tree.
4. **Pre-boot XPU gate (mandatory)** — a trivial triton vector-add on one card
   must compile **and** produce the exact result (`TRITON_XPU_GATE=PASS`) before
   any model boot. Catches every remaining JIT gap in seconds. Skipped only via
   `--launch` (the watchdog restart path) or `XPU_GATE_DISABLE=1`.
5. **Image gate** — the pinned image must already exist locally (no pull; local
   build of record).
6. **Manifest** — `.run/manifest.json` records image, model path, git describe,
   env hash (lumnus.env + serve args, secrets excluded), MML/MNS, start time.
7. **Watchdog spawn** — mandatory; double opt-out required to disable (see
   below). Single-instance guard: `pgrep -f wedge-watchdog.sh` — never a second
   watchdog.
8. **Launch** — `docker run -d` with the serve args; idempotent (`rm -f` first).
9. **Verify / READY** — poll `/v1/models` for the served name (up to
   `READY_WAIT_SECONDS`, default 900; boot is ~4–5 min) **and** require
   `Application startup complete` in the container logs (≤180 s after the API
   answers). READY is only announced when both hold.

Other subcommands: `stop`, `restart` (full validation path, then stop+start),
`status`, `logs` (`docker logs -f`). `--launch` = launch-only, skips the XPU
gate — this is the watchdog's restart interface (`PROD_RESTART_CMD`, default
`<lane>/start.sh --launch --replace`).

## Stop / start procedure

- **Stop:** `<lane>/stop.sh` — order matters: TERM the watchdog first (so it
  cannot "detect a wedge" and restart the server mid-teardown), then
  `docker rm -f` the container (graceful: SIGTERM, SIGKILL after stop timeout).
  It reads the single pidfile (`.run/watchdog.pid`) and belt-and-suspenders TERMs any
  lingering `wedge-watchdog.sh` process (end-anchored pattern).
- **Start:** `<lane>/start.sh` — the full contract above.
- **Restart:** `<lane>/start.sh restart` for planned work (re-runs every gate).
  **Drain first** — see restart discipline below.

## status output

`<lane>/start.sh status` prints four lines (no validation, safe anytime):

```
container : running (up since <ISO>) | NOT RUNNING
api       : READY on :8022 (qwen-256k) | not answering on :8022
watchdog  : running (pid <pid>, interval 60s) | NOT running
log       : docker logs <container>
```

Healthy production = all three first lines green. `api` is a live
`/v1/models` probe matched against the served model name.

## Wedge watchdog contract (`scripts/wedge-watchdog.sh`)

The Xe2 Level-Zero wedge kills the serving job every 2–6 h unattended under
load (dmesg signatures: `Engine reset: engine_class=ccs|bcs`,
`Fault response: Unsuccessful`, `guc_exec_queue_timedout_job`). Only a container
restart recovers; in-flight requests are lost. The watchdog is therefore
**mandatory** — disabling requires the double opt-out
(`WEDGE_WATCHDOG_DISABLE=1` **and** `--no-preflight`); a single opt-out is
supervised anyway with a warning.

Loop (interval 60 s, `.env`):

1. **Liveness = gen-probe**: a 1-token chat completion must actually generate
   (a wedged engine can keep `/v1/models` answering). `/v1/models` health is
   checked alongside.
2. **Cold-load guard**: no stall counting until `/v1/models` has answered once
   (boot is ~4–5 min; a booting engine is not a wedged engine).
3. **Log-growth escape**: if the bounded container-log tail (last 2000 lines)
   grew since last cycle, the stall counter resets.
4. **Load-aware gate** (added 2026-10-03 after a false-positive restart during
   8×~120K-token prefills): when the probe fails, take two `/metrics` samples
   `LOAD_SAMPLE_S` (default 30 s) apart and log both (`load-guard samples:
   s1=[running gen prompt] s2=[...]`). Requests running AND either the
   generation **or prompt** token counter advanced = busy, not wedged — a long
   prefill generates zero output tokens for minutes, so the generation counter
   alone misreads a saturated engine as wedged. Fail-open: if `/metrics` is
   unreachable this path never restarts.
5. **Wedge-signature scan**: container-log tail grepped for the engine-reset /
   `EngineDeadError` / sample_tokens-RPC-timeout patterns; device-count check
   (all 4 XPUs visible).
6. **Stall limit**: 3 consecutive failed probes (each passing the load guard) =
   wedge.
7. **Xe engine-reset monitor**: every cycle logs
   `xe engine resets: total=N new_this_cycle=M` from `sudo -n dmesg` — the dmesg
   reset counter is the ground-truth wedge signal, audited continuously.
8. **Capture-first discipline**: on wedge detection, **before any restart**,
   capture last 200 container-log lines + `py-spy` python-frame dumps of every
   TP worker / EngineCore (via `docker top`, `sudo -n py-spy dump`) + top-TID
   CPU table, into `.run/wedge-<ts>.log`. Then kill the container process
   group and `docker rm -f`.
9. **Restart**: via `PROD_RESTART_CMD` (default `<lane>/start.sh --launch --replace`),
   **bounded to 3 retries** (`WEDGE_WATCHDOG_RETRIES`); after that it gives up
   loudly and exits — a human takes over; evidence in `.run/wedge-*.log`.
10. **Single instance (host-wide, by design)**: `start.sh` will not spawn a
    second watchdog while `pgrep -f wedge-watchdog.sh` finds one — one rig,
    one serving lane. `LANE_DIR` separates state dirs, not concurrency.

## Sentinel — alert-only

Operator-scheduled (e.g. cron every 30 min). **Alert-only: no automatic
rollback** — checkpoint/engine switches are operator decisions. Triggers:

- new `Engine reset` lines in `sudo -n dmesg` since the last tick, **or**
- container `RestartCount` increased, **or**
- `/v1/models` non-200 while no boot is in progress (boot log idle > 25 min).

On trigger it appends a line to its log and writes
`ALERT_NEEDS_ROLLBACK.flag` (one alert per 75 minutes — thrash guard). Healthy
ticks are silent, exit 0. When the flag appears: read the trigger, check
`.run/wedge-*.log` and `docker logs`, then decide — restart the lane, or
roll back per `rollback.md`.

## Sampling pins (restored 2026-10-05)

`serve-args` carries the model-card instruct sampling set as a server
override:

```
--override-generation-config {"temperature": 0.7, "top_p": 0.80, "top_k": 20,
  "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
```

**Rationale.** The Qwen model card's instruct-mode sampling recommendation is
exactly this set. The pre-promotion production line ran temperature-only
(`{"temperature": 0.7}`) with no measured reason recorded for dropping the rest
of the pins. The wtdcode checkpoint card warns that temperature > 0.7
degenerates and that greedy decoding loops — the pinned set is the safe
operating envelope, not a tuning experiment. Pre-sampler args are preserved on
the host as `serve-args.bak-pre-sampler`.

**Repetition safety net on top.** `lumnus.env` sets
`B70_DEFAULT_REPETITION_DETECTION=max=1,min=1,count=128` (Lumnus patch 0010):
one token repeated 128× ends generation — this bounds the token-1023 "duct" NaN
loop class if one ever slips through. A repetition stop surfaces as
`finish_reason=repetition`; client-side (Hermes) this is treated as a normal
end-of-turn, not an error.

Loop A/B evidence for the restored set is in the STATUS sampler-restoration
record (0/5 repetition stops with the pins + detection active).

## Restart discipline

- **Drain children first.** In-flight requests die on any restart (wedge or
  planned). Hermes child sessions retry for only ~6 minutes — a restart with
  live children strands them. Before a planned restart: stop routing new work,
  let in-flight children finish (or accept the ~6 min retry window as the
  drain bound), then restart.
- **20-minute outage rule.** Any production outage expected to exceed ~20
  minutes (boot is ~4–5 min; a wedge recovery + verify is longer) must be
  **reported to the operator before the production change is made**, not after.
- **Every production change is a reported change.** Restart, flag flip, args
  edit, rollback — report first, act second. The watchdog's automatic restarts
  are the only unsupervised exception, and they are logged
  (`.run/watchdog.log`, `wedge-*.log`) and alerted (sentinel).
- After any restart: `<lane>/start.sh status` green, then the smoke tier of
  `verify.md` before declaring recovery.

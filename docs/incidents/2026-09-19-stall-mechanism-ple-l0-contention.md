# 2026-09-19 — Stall mechanism CAPTURED: PLE USM memcpy vs graph replay in Level Zero command-list append

## The capture (boot 3174405, stall #3, onset 05:08:19Z)
py-spy 0.4.2 installed in-container; sampler at 5s cadence across all 5 processes; the burst
fired 05:07:04 collapsed to 0.0 tok/s (Running: 8) at 05:08:19 and we sampled straight through
the stall window. Native dumps (--native) of all four ranks taken ~05:11-13, saved before
container death 05:23:01 → `.run/evidence/corpse-tp{0,1,2,3}-native.txt`.

## The four ranks at 05:09:58 (10s-scale into the stall) — RANK DIVERGENCE
- TP0 main (pid 348): **idle 46+s** in `pthread_mutex_lock (libc)` ← `sycl queue submit_impl
  (libsycl.so.8)` ← `sample (model_runner.py:1455)` — trying to SUBMIT the sampler kernel.
- TP1/TP2/TP3 main (374/405/436): **active, sched_yield spin** inside
  `ur_command_list_manager::appendCommandBufferExp (libur_adapter_level_zero_v2)` ←
  `replay (torch/xpu/graphs.py:108)` ← `run_fullgraph` ← `execute_model` — appends never
  completing for ~46s (a replay is ~50ms).
- EngineCore (239): `shm_broadcast.wait` → zmq poll — waiting for results that never come.
- **TP0's PLE connector thread** (`_request_loop` connector.py:296 → `_process_request`
  connector.py:315): stuck inside **`ur_command_list_manager::appendUSMMemcpy`
  (libze_intel_gpu.so.1.15.37833)** — a USM memcpy into the same Level Zero command-list
  machinery the replays are spinning in.

## Mechanism (candidate, now evidence-backed)
The PLE offload connector's background thread issues a USM memcpy per request per step
(the duplicate pile-up GLM filed 2026-09-19: ~30 duplicate PLE requests / 13s during decode).
Those memcpys contend with graph-replay command-buffer appends **inside Level Zero's
command-list manager**. When one append wedges (TP0 connector thread), the shared append
path blocks: TP1-3 spin in replay append, TP0's sampler submit hits the held mutex, the
sample_tokens collective never assembles, EngineCore's RPC wait times out
(900s — `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900` verified active: death at 882s, vs 300s
for the 02:21 crash).

This **unifies the program's failure surface into one site**:
- Boot wedge (post-KV shm_broadcast mute, wedgedAB/ABr/bisect1): same block site
  (workers stop publishing to the broadcast ring) — plausibly the same L0 contention at
  first-graph-capture/first-PLE-transfer.
- Serving stall (02:21, 04:07→05:23 captured, all at 8 concurrent): same site under load.
- PLE duplicate pile-up: the load generator of the contending transfers.
- "Same failure, two exposure windows" (Ryan, confirmed at block site) — now with a
  candidate physical mechanism inside Level Zero.

## Why it looks stochastic
Per-step USM memcpy from a background thread only wedges when it lands while replay appends
are in flight AND hits the bad ordering/lock — a race, consistent with: clean 48K-token soak
then death on a later burst; 3 stalls in 4 bursts today; boot-to-boot variance in serving perf.

## Evidence files
- `.run/evidence/corpse-tp0-native.txt` — mutex block in submit_impl + PLE thread in appendUSMMemcpy
- `.run/evidence/corpse-tp1-native.txt` — sched_yield spin in appendCommandBufferExp during replay
- `.run/evidence/corpse-tp2-native.txt`, `corpse-tp3-native.txt` — same spin
- `.run/wedge-20260919T052419Z.log` — EngineDead at 05:23:01 (TimeoutError sample_tokens)
- `/tmp/soakfix_8way.json` — rounds 0.0 (stall killed the soak); engine counters 171.2→0.0
- stallspy dumps lost with container (content extracted to this doc + transcripts)

## Watchdog v2 field findings (its first live stall)
1. **EngineCore blindspot**: `/v1/models` stays 200 while EngineCore is dead → all ticks
   "healthy" through an 882s stall. Health must be a 1-token generation probe.
2. Decision log worked as designed (49 decisions, full verdicts) — but the mute detection
   never fired because the APIServer kept writing log lines (mtime advanced). D1's mtime
   signal is necessary-not-sufficient: needs the generation probe AND mtime.
3. Restart mechanism failed 3/3 fast (containers died instantly after start.sh --launch;
   wedge-20260919T052520Z.log). Relaunch needs the full start.sh path.
4. wchan capture in capture_and_kill failed (docker exec on exiting container) — capture
   must happen on detection, not after rm.

## Fix candidates (need Ryan's go — config surgery)
A. **PLE offload disable test** (cheapest discriminator): if PLE offload can be disabled via
   config/env, a boot with PLE off → burst test → no stalls = mechanism confirmed. Cost: RAM
   headroom changes; PLE is the model's per-layer offload design — must check what disabling
   does to capacity (12.22 GiB/rank pinned host RAM freed, but device memory layout changes).
B. **Serialize PLE transfers onto the replay queue** (code fix in connector): if the connector
   uses a separate queue, moving its memcpy in-order behind replay appends removes the
   contention (may cost decode bubbles ~transfer time per step).
C. **Gate duplicate PLE requests at the source** (connector.py `_launch`): the per-step
   duplicate send we verified off-path for CORRECTNESS is on-path for L0 contention — 30/13s
   duplicate memcpys are 30/13s append attempts. Dedup BEFORE enqueue, not after recv.
D. **Immediate mitigation for stability**: cap concurrency at the scheduler (MNS below the
   observed trigger... no — stalls happened at 8 running on MNS=12 AND MNS=8; concurrency cap
   is not established as a mitigation).

## Status
- Rig DOWN at write time (watchdog gave up after 3 failed restarts; container gone).
- Next: relaunch to pinned config; watchdog v2 health-check fix (generation probe); then
  burst-test to collect stall-rate baseline; PLE-disable experiment on Ryan's go.

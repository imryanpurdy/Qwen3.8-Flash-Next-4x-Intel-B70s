# 2026-09-19 — Stall #4: live capture mid-wedge, mechanism signature reproduced on a fresh boot

## What happened (boot 3285344, clean relaunch, watchdog v2 OFF for the test window)
Warmup 200, py-spy staged, burst fired 06:29:45. Round-level results (soakfix_8way.json):
- r1: 61.0 agg (recovered from an early partial stall)
- r2: 168.8 agg (healthy full-throughput round — the engine CAN serve)
- r3: 106.7 agg (degraded)
- r4: 0.0 (stalled; round aborted at drain timeout)
- post_models: 200 (APIServer alive; engine wedged)

## Live capture DURING the wedge (06:32:19-06:32:39, stallspy preserved to .run/evidence/stallspy-0632/)
Same signature as stall #3, rank-for-rank:
- EngineCore (220): idle in `shm_broadcast.wait` → zmq poll
- TP0 (329): idle in `sample (model_runner.py:1455)` for 20+ consecutive seconds
- TP1/TP2/TP3 (355/386/417): **active** in `replay (torch/xpu/graphs.py:108)` →
  `run_fullgraph (cudagraph_utils.py:447)`

## Live native dumps mid-wedge (stall #3 window, 05:5x, saved pre-death)
- TP0 main: `pthread_mutex_lock (libc)` ← `sycl queue submit_impl (libsycl.so.8)` ← sample
- **TP0 PLE connector thread: `ur_command_list_manager::appendUSMMemcpy (libze_intel_gpu.so.1.15.37833)` ←
  `_process_request (ple_offload/connector.py:315)` ← `_request_loop (connector.py:296)`**
- TP1/TP2/TP3 main: `sched_yield` spin inside `ur_command_list_manager::appendCommandBufferExp
  (libur_adapter_level_zero_v2)` ← replay
- EngineCore: zmq poll
- Workers pinned at 100% CPU (spinning), GPU-side idle (xpu-smi 0 samples inside container —
  debugfs mount artifact, host-side reads only per skill)

## Mechanical reading (tightened after two independent captures)
The four ranks enter the same step; TP0 reaches the sampler kernel submission while TP1-3 are
still appending their replay command buffers. TP0's PLE connector thread concurrently issues a
USM memcpy through the Level Zero command-list append path. The append path wedges —
TP1-3 spin forever inside appendCommandBufferExp, TP0's submit blocks on the runtime mutex.
Step never completes on any rank; sample_tokens collective never assembles; EngineCore's RPC
wait times out (900s) and the engine dies. APIServer keeps answering /v1/models (health=200)
throughout — models-endpoint health is blind to engine death.

This is now 2 stalls with full native evidence + 1 with python-level evidence, all the same
shape, across 3 different boots and both MNS=8 and MNS=12 configs. The trigger correlates with
concurrent-load bursts (never reproduced idle), but NOT with a specific MNS.

## stall #4 also gives the first partial-round data under a live wedge
r1 61 → r2 169 → r3 107 → r4 0: the engine oscillates between healthy throughput and
degraded rounds before a full stall. This kills the clean "works then dies" binary —
there are WARNING rounds (61, 107) where throughput is depressed but nonzero. Watchdog v2's
generation-probe must treat >2x throughput drop sustained 2 rounds as warning-level telemetry.

## Watchdog v2 status (its second live stall, still OFF for capture purity)
- Health=200 blindspot CONFIRMED again (models endpoint alive through full wedge).
- The old watchdog from start.sh spawned a duplicate on relaunch (pid 3233940 killed at 06:10;
  start.sh's spawn races the manual v2 instance). start.sh line 411 spawns its own watchdog —
  v2 must either replace that line or WEDGE_WATCHDOG_ALREADY_RUNNING must suppress it.

## Program state
- 4 stalls total: 02:21 (MNS=8 boot 2967807), 03:52 (MNS=12 boot 3031727), 05:08 (boot 3174405,
  full native capture), 06:30 (boot 3285344, live dump capture).
- Rig UP and serving at last check (post-stall-#4 engine died at RPC timeout ~06:45; start.sh
  relaunched at 06:10 → that instance was the one that wedged; relaunch pending).
- Evidence: .run/evidence/{corpse-tp0..3-native.txt, live-stall4-tp{0..3}-native.txt,
  stallspy-0632/ (176 dumps)}.
- Fix candidates unchanged from ca76249 doc; add: PLE-disable discriminator is now THE
  cheapest experiment (config-only) — if PLE offload off → no stalls, mechanism confirmed.

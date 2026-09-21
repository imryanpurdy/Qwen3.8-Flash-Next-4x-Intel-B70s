# 2026-09-19 — 12-Way Output Stall: Live Capture on Clean Host

## Status
OPEN — mechanism hypothesis formed, A/B + tracker search in flight.

## The finding
On a clean host (fresh reboot, stock CCL, zero accumulated device state), the
12-way throughput collapse reproduced — so it is NOT driver-state accumulation.
The OS rebuild's expected win shrinks to: MTP1-capture + stability + a clean BOM.
The 200-400 aggregate target depends on solving THIS, a scheduler/output-path lane.

## Measurements (all MTP0, stage-v24g, boot-verified)
| Run | Host state | Rounds (agg tok/s) | Notes |
|---|---|---|---|
| clean12 (pre-reboot baseline... flagless) | fresh boot 19:32 | 45.7, 57.3, 30.3, 28.1 (mean 40.4) | fast rounds first, then slow |
| clean12b (CCL_ZE_CACHE_OPEN_IPC_HANDLES=0) | fresh boot 19:52 | 30.3, 30.4, 30.4, 30.6 (mean 30.4) | flat-slow from r1 |

Shape: **step function, not decay.** Fast rounds (125-157s wall) → slow rounds lock
to ~237s wall (within 1% across rounds and runs). 8-way on the same boot is
unaffected (campaign history: ~163 sustained).

## Live capture (21:07-21:33, mid-slow-phase, py-spy all three processes)
- Frontend (APIServer): 12 requests "running", length-counter frozen at 68 for
  ~20 min, then 12 completions arrived in a wave; counter jumped to 89.
- EngineCore: MainThread **idle at input queue** (`queue.get`,
  `_process_input_queue` core.py:1453). Believes there is no work.
- Worker TP0: idle in `shm_broadcast` dequeue (`worker_busy_loop`
  multiproc_executor.py:1037). Waiting for EngineCore.
- Engine kept serving new small requests in 0.4s throughout the zombie window.

## Interpretation (hypothesis, to be settled)
A 12-way batch's outputs are lost in the EngineCore→APIServer message path.
The frontend holds zombie "running" requests; its periodic retry/re-submission
mechanism eventually recovers (Ryan predicted an 80-100s recovery mechanism —
observed recovery took longer but the shape matches). During the gap, the
request sits below the wave floor. `num_requests_waiting=0` and the idle
EngineCore input queue are consistent with a request the engine never received
(or already consumed an answer for) while the frontend still owns it.

## Log signatures during events
`PleOffloadWorker: Duplicate PLE request for dp_rank=0; skipping duplicate`
at ~4-min intervals, in bursts AND idle. Cannot yet distinguish symptom from
co-morbidity. The dedup-skip is a candidate sink: if a retransmitted request
is skipped as duplicate, it can never complete.

## Instrumentation notes
- `vllm:kv_cache_usage_perc` is STATIC in this build (dead gauge — unusable).
- `num_requests_running/waiting` track frontend truth, not engine truth.
- Worker VmRSS (PLE table working set): 2.85 GB idle → 3.79 GB burst, flat
  within a run — PLE table growth ruled out for the step-change.
- py-spy must be reinstalled per container; wheel cache at rig /tmp/pyspywheel/
  (recreate after host reboot: `pip download py-spy==0.4.2 -d /tmp/pyspywheel`).

## Exonerations recorded
- Driver-state accumulation (clean host reproduces).
- SYCL_UR_USE_LEVEL_ZERO_V2 legacy adapter (faults anyway on MTP1 capture).
- PLE table RSS growth (flat).
- MTP/spec-decode (all measurements MTP0).
- Engine-side compute stall (EngineCore provably idle; serves others in 0.4s).

## Open questions
1. Same-boot A/B: CCL_ZE_CACHE_OPEN_IPC_HANDLES=0 on vs off (4 rounds each) —
   the flag-run was flat-slow from r1 while the flagless run had fast rounds
   first. Confounded by boot order; two boots settle it.
2. Scheduler truth: does EngineCore hold these 12 request IDs? (needs
   instrumented logging or a targeted patch; gauges are untrustworthy).
3. Upstream tracker search (dispatched): EngineCore-idle output-stall family.
4. v3 connector angle: connector thread was silent in all dumps — but the
   queue handoff lives on the runner thread; if a put blocked past the 2s
   bounded poll, the runner would stall AFTER enqueueing... no flag-timeout
   lines in server.log. Weak exclusion; keep on the list.

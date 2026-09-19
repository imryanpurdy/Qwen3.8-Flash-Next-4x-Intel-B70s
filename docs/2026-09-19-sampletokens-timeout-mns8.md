# 2026-09-19 — sample_tokens RPC timeout under 8-way concurrency (FIRST-HAND)

## Event
02:21:42 UTC, boot 2967807 (stage-v24c, sanctioned config: MTP0, MNS=8, MBT=2048, gpu-util 0.75,
native KV 127,078): EngineCore fatal `TimeoutError: RPC call to sample_tokens timed out.`
→ EngineDeadError → APIServer shutdown. Watchdog/relaunch behavior: container Exited(0), watchdog
did NOT auto-restart (watchdog.log tail shows an EARLIER 900s /v1/models timeout from restore-boot
era; watchdog exited on signal — coverage gap during steady-state serving).

## Context (pre-registered expectations, handback §4 Lane B)
8-way aggregate bench, round 1 (first burst after single-request warmup), ~7 running reqs.
Handback pre-registered 100-130 aggregate at MNS=8. Instead: engine died in round 1.

## Significance
1. The "retracted folklore" MNS=8-adjacent concurrency crash is now FIRST-HAND evidence on this
   rig at the sanctioned config. Not thirdhand anymore. Scheduler stats at death:
   num_running_reqs=7. KV arithmetic exonerated (2K demand vs 127K capacity).
2. Same signature family as boot wedges: workers stop responding to the RPC handshake
   (sample_tokens collectives), EngineCore times out at 300s (VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
   multiproc_executor.py:353-435). Boot wedge = the stall at boot; this = the stall mid-serving.
   One underlying native stall, two windows of exposure.
3. Evidence preserved on jobe: fn-recipe-int4/.run/evidence/
   crash-20260919T022142-sampletokens-mns8.txt (full stack + dump), server-20260919T022142-precrash.log
   (full server.log 12,041 lines). Observer saw broadcast warnings 36→40 during bench window.
4. Open question for the profiler lane: what are the workers doing when sample_tokens stalls?
   The wedge_obs captures + a torch-profiler boot under 8-way load are now the SAME investigation.

## Consequences for the program
- MNS sweep must now treat MNS=8-under-load as a suspect, not a baseline. The sweep design changes:
  instrument the stall (VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS raise to 900 to convert death→hang+capture,
  observer wchan on workers during load) BEFORE burning boots on 12/16.
- Watchdog gap: watchdog only covers boot phase (900s /v1/models poll), NOT steady-state serving.
  A serving-time stall needs its own tripwire if we want auto-recovery. Filing as ops item.
- Boot provenance rule stands: next benches happen on the fresh boot after full N=20 gate.

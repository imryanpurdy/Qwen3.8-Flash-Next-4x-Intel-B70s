# 2026-09-19 — Second sample_tokens stall: 8-on-12 config, engine died mid-soak-retry

## Event
03:52:26Z: timer-fixed 8-way soak fired on MNS=12 config (boot 3031727). Engine counters show
the load landing: 30.7 → 171.2 → 58.4 → 0.0 tokens/s with Running: 8 — then silence.
03:53:40 → 04:06:42: 14x shm_broadcast.py:801 60s-warnings (workers never publish).
04:07:40: EngineCore fatal — **TimeoutError from shm_broadcast acquire_read inside the
sample_tokens RPC wait** (stack: multiproc_executor get_response → shm_broadcast acquire_read →
_spin_condition.wait → raise TimeoutError), SchedulerStats num_running_reqs=8.
Engine dead. APIServer 500s. Container Exited(0) 04:07. Watchdog had already exited on signal
(boot-time watchdog ended with the earlier restore; no serving-time coverage — gap confirmed again).

## What this proves
1. **The stall family reproduces at 8 concurrent on the MNS=12 config** — the same signature
   as 02:21 (sample_tokens RPC timeout, EngineDead, ~7-8 running reqs). NOT config-specific
   to MNS=8. Ryan's caution vindicated: the config-specific hypothesis is DEAD.
2. **Corrected mechanism understanding**: the 300s/900s sample_tokens timeout fires INSIDE
   shm_broadcast acquire_read — EngineCore waits on the workers' broadcast ring. The workers
   stop publishing. Same block site as the post-KV boot wedge (wedgedAB/ABr/bisect1).
   Boot wedge and serving stall are now CONFIRMED the same native class (one stall, two windows).
3. **The 900s instrumentation worked as designed** — wait, it did NOT fire at 900s. Timeout at
   04:07:40 is ~840s after throughput went 0.0 (03:53:5x) — close to but under 900s; the
   EngineCore-side dequeue timeout path may be governed by a different knob than
   VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS (the stack went through mq.dequeue(timeout) →
   acquire_read raise TimeoutError). OPEN: verify which timeout actually governs (env was
   exported into the container; confirm it took effect).
4. **Stall onset is FAST and silent**: last healthy counter 03:52:36 (171.2 tok/s), zero
   throughput 03:52:56, first warning 03:53:40. No error precedes the stall. The trigger is
   inside one 10s logging window.

## Confusor present
vxk-build25 (build B's subagent CMake work) was RUNNING during this stall (Up 35 min at
detection). Same RAM/IO environment as the first stall? — 02:21 crash had NO build running.
So a concurrent heavy build is NOT required for the stall, and its presence this time does not
exonerate it either (n=1 each way). Note it, control for it going forward.

## Consequences
1. **The stall is now the #1 program problem, confirmed twice, both under 8-concurrent load,
   ~30-60s after load lands.** Both victims were serving normally first (171 tok/s!).
2. Load landed at 03:52:26; counters healthy through 03:52:36; dead by 03:53:40. Time-to-stall
   from load: 30-70s in BOTH incidents. That's the capture window for instrumentation.
3. Next boot MUST carry: py-spy pre-attached to all 4 workers + EngineCore (or observer
   wchan watcher at 5s cadence during the first 120s under load), VLLM_EXECUTE_MODEL_TIMEOUT
   verification, and the timer-fixed soak as the trigger.
4. 8-way sustained aggregate from soak #1 (163 tok/s) was measured BEFORE this stall on the
   SAME boot — one full clean soak, then death on the second soak ~25 min later. The stall is
   NOT immediate-on-first-load; it is stochastic per-load-event (p roughly 30-50% per burst?).

## Status
- 8-way timer-fixed rerun: FAILED to produce numbers (engine died r1) — not a bench result.
- Boot 3031727 is dead; relaunch pending. Next boot = watchdog v2 live BEFORE more load tests.

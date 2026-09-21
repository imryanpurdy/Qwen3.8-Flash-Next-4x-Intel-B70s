# PLE v3 Host-Staging Campaign — Pre-Registration (2026-09-19)

## Change under test
stage-v24f: v3 PLE host-staging. Runner thread enqueues stream-ordered D2H
copies into pinned double-buffered mirrors, then a 1-elem int32 flag write
carrying a per-boot monotonic seq (flag LAST on the stream). Connector thread
consumes via numpy `copyto` only, gated on `flag_np[slot]==seq` (bounded 2s
poll, rate-limited legacy `copy_` fallback). The (slot, seq) token rides in the
queue payload — atomic handoff, no shared mutable state. Miswire canary logs
`PLE v3 staging wired: staged=True` per rank at init (verified x4 on boot
3915112).

## Prior evidence (smoke, this boot)
- In-order visibility primitive: 300/300 trials clean (test-inorder.py,
  idle-engine caveat logged; campaign is the real proof).
- Known-answer: correct temp-0 output ("Paris").
- Burst 1: 4 rounds 88.0/182.5/182.6/180.1 agg, 0 errors, engine alive.
- Burst 2: mean_agg 170.4, 0 errors, engine alive.
- **L0-absence proof: 170 py-spy dumps mid-burst, 0 contain appendUSMMemcpy**
  (connector thread present in 34, idle at queue-get). Zero flag timeouts.

## Hypothesis
The serving stalls (5/7 bursts pre-fix baseline: 02:21, 03:52, 05:08, 06:30 +
interrupted-burst wedge on boot 3338578) are caused by the connector thread's
L0 USM-memcpy append contending with graph-replay appends. Removing all L0
traffic from the connector thread eliminates the stall.

## Protocol
- 15 bursts, sequential, 8-way soakfix (timer-fixed harness), boot 3915112
  (stage-v24f), watchdog v2.5 (gen-probe health) live.
- Per burst: fire soakfix 8 → capture rounds JSON → gen-probe (1-token
  completion) + models probe → py-spy sampling armed during burst.
- Stall = any of: harness rounds collapse to 0.0 with engine wedged, gen_probe
  != 200 within 90s of burst end, EngineDead/TimeoutError signature in
  server.log, watchdog kill decision.
- Any stall: record boot/phase/timestamp, preserve dumps, restart engine,
  continue remaining bursts (stall rate counts all bursts, pre- and
  post-restart).

## Decision rule (pre-registered)
- 0-1 stalls in 15 → mechanism confirmed, v3 ships as default. Lock diagnostic
  remains available but is not run.
- 2-3 stalls → weak evidence; extend 10 more bursts before any claim.
- ≥4 stalls → fix ineffective; trigger lock diagnostic (serialize PLE transfer
  vs replay) and/or VLLM_PLE_CPU_OFFLOAD=1 discriminator.
- Aggregate throughput is a measurement, not a gate. Stability gates only.

## Confounds declared in advance
- Boot provenance: single boot so far; any engine restart during the campaign
  adds a new provenance record to the per-burst log.
- Idle-engine microtest + mid-burst dumps cover different conditions; the
  dumps are the campaign-relevant evidence (already in hand).
- Watchdog v2.5 was swapped in immediately before the campaign; its kill
  behavior is not the pre-fix watchdog's behavior (gen-probe is stricter —
  stalls will be caught earlier and logged as kills, which count as stalls).

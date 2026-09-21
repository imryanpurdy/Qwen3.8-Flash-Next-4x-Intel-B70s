# PLE v3 Host-Staging Campaign Results — CONFIRMED (2026-09-19)

## Verdict (pre-registered rule, docs/campaigns/2026-09-19-v3-campaign-prereg.md)
**0 stalls in 15 bursts → mechanism confirmed. v3 ships as default.**
Pre-fix baseline: 5 stalls / 7 bursts (0.71/burst). P(0 | 15, p=0.71) < 1e-7 —
luck is excluded at any reasonable standard.

## Conditions
- Image stage-v24f, boot 3915112, single boot end-to-end (READY +857s, 0
  broadcast warnings — clean-class boot).
- Harness: soakfix.py 8-way (4 rounds × 600 tok × 8 streams per burst,
  timer-fixed), ~20s inter-burst gaps, sequential.
- Watchdog v2.5 (engine-true gen-probe health) live the whole campaign; zero
  interventions, zero wedge verdicts.
- One protocol deviation: burst 4 ran under two accidentally-overlapping
  harness loops (~16-way). Logged for honesty: mean 155.0, min round 84.3,
  nonzero throughout, engine healthy after. NOT counted as a stall (stall =
  collapse-to-zero + failed probe). Bursts 13-15 re-ran clean after a
  process-hygiene fix (pkill verify-then-relaunch).

## Per-burst results (mean aggregate tok/s, all post_models=200)
| # | mean | min | max | note |
|---|------|-----|-----|------|
| 1 | 177.2 | 168.6 | 182.1 | |
| 2 | 177.4 | 164.8 | 181.7 | |
| 3 | 181.6 | 179.6 | 182.8 | |
| 4 | 155.0 | 84.3 | 182.2 | double-loop artifact, see above |
| 5 | 180.4 | 177.7 | 182.3 | |
| 6 | 181.1 | 179.4 | 182.8 | |
| 7 | 180.6 | 179.3 | 181.9 | |
| 8 | 183.7 | 181.2 | 189.7 | |
| 9 | 180.1 | 178.8 | 181.5 | |
| 10 | 181.0 | 179.4 | 182.6 | |
| 11 | 180.5 | 179.2 | 182.3 | |
| 12 | 180.2 | 178.4 | 181.9 | |
| 13 | 180.7 | 178.8 | 182.8 | |
| 14 | 179.7 | 176.8 | 181.9 | |
| 15 | 175.8 | 163.3 | 180.9 | |

60/60 rounds errs=0. Zero `flag timeout` fallbacks in server.log — the
staging path never once degraded to the legacy connector-thread copy_.

## Mechanism evidence recap (this boot)
- Miswire canary: 4/4 ranks `staged=True` at init.
- L0-absence proof: 170 py-spy dumps mid-burst, appendUSMMemcpy in 0/170
  (pre-fix: present in every wedge capture, 27/27 connector-thread samples).
- 300/300 in-order visibility trials (idle-engine microtest).

## What shipped (stage-v24f = stage-v24e + v3)
v3 PLE host-staging: runner-thread stream-ordered D2H into 2-slot pinned
mirrors, flag-write-LAST with per-boot monotonic seq, connector consumes via
numpy only (2s bounded poll, logged legacy fallback), (slot,seq) token in the
queue payload (atomic handoff), bounded queue.put (no queue.Full crash),
per-boot miswire canary. Fixes both defects of the connector-thread copy_:
the L0 contention (stalls) and the unordered read (torn n-gram context in
MRV2 era).

## Next (per Ryan's sequence)
1. MTP acceptance re-check (torn reads gone → acceptance should recover;
   prereg docs/campaigns/2026-09-19-mtp-acceptance-check-prereg.md).
2. KV expansion retry under patched watchdog (282,880-token allocation was
   demonstrated once; deserves a clean run).
3. max_model_len raise to what expanded KV supports (4352 was interim).

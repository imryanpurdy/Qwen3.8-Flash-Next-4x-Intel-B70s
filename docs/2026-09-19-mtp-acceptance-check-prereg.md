# MTP Acceptance Recovery Check — Pre-Registration (2026-09-19, post-campaign)

## Motivation (Ryan)
V2-runner era: connector's connector-thread `copy_` read device-written buffers
without ordering — torn/stale token IDs fed the PLE n-gram lookup. Drafts are
verified, so the damage surfaced as rejected drafts, i.e. eaten MTP acceptance,
never wrong output. v3 removes the torn read (runner-thread stream-ordered D2H,
verified complete-before-consume). Prediction: MTP1 acceptance recovers, so
MTP1's gain may exceed the previously measured +13%.

## Design
1. **Log forensics (free):** extract spec-decode acceptance stats (accepted
   draft tokens/step, acceptance rate) from engine logs across the V2 cutover
   and across the v3 cutover if present in metrics lines. Time-series step
   change = diagnosis confirmation.
2. **Controlled A/B (post-campaign, fresh boots):** same 8-way soak prompt set,
   timer-fixed harness:
   - Boot A: MTP_NUM_SPECULATIVE_TOKENS=0, stage-v24f
   - Boot B: MTP_NUM_SPECULATIVE_TOKENS=1, stage-v24f
   Metric: aggregate tok/s delta B vs A (post-v3). Compare against pre-v3
   MTP1/MTP0 delta (+13%, overnight measurement).
3. **Gate:** stability first — this check runs only after the campaign
   decision rule resolves clean (0-1 stalls / 15).

## Prediction (registered in advance)
Post-v3 MTP1 delta > pre-v3 +13% (torn-read removal returns acceptance toward
the lab's +73% ceiling; realistic partial recovery).

## Confounds
- MTP acceptance also depends on prompt distribution; A/B uses identical
  prompt set, same boot recipe, only the spec token count flips.
- Batch-order effects: A/B order randomized (coin flip at run time), 3 rounds
  each after 1 warmup round.

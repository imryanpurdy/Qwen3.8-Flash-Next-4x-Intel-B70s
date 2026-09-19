# 2026-09-19 — PLE-disable stall discriminator experiment (pre-registration)

## Hypothesis under test
The serving stalls (4 reproduced: 02:21, 03:52, 05:08, 06:30 on 09-19) are caused by the PLE
offload connector's per-step USM memcpy contending with graph-replay command-buffer appends
inside Level Zero's command-list manager. Live evidence: TP0 PLE connector thread wedged in
`appendUSMMemcpy` (connector.py:315) while TP0 main blocks on the SYCL submit mutex (sampler)
and TP1-3 spin in `appendCommandBufferExp` during replay — captured in 3 independent wedges
across 3 boots.

## Experiment
Boot with PLE offload disabled. If the mechanism is right, sustained 8-way bursts should run
WITHOUT stalls for at least 4 rounds × 600 tok × 8 streams, repeated across 2 bursts (the
pre-experiment stall rate was 4 stalls in ~6 bursts).

## Method (config-only, reversible)
1. Identify the PLE offload switch: check start.sh/.env for PLE offload flags
   (--cpu-offload-params / ple_offload env / VLLM_PLE_* ). Candidate: remove the
   `--cpu-offload-params ple_embedding.ngram_embedding.weight ...` / ple offload lane,
   or set the PLE offload env off.
2. RAM check first: disabling PLE offload may force PLE weights onto device → VRAM overflow
   OR onto pageable host → slower load. Record KV tokens + boot phase clocks either way.
3. Same bench: soakfix.py 8-way × 4 rounds × 600 tok × 2 bursts, stallspy armed during burst 1.
4. Gates (stability, not aggregate — per Ryan's standing correction):
   - PASS (mechanism confirmed): both bursts complete, zero sample_tokens timeouts,
     post-generation probe 200.
   - FAIL (mechanism refuted or PLE-disable infeasible): stall reproduces with PLE off,
     or boot cannot serve with PLE off (OOM/crash at load).
5. Record boot provenance (boot ID, phase clocks), .env diff, KV tokens.

## Risks
- PLE offload off may be architecturally required for INT4 lane (per-layer embedding table
  38.4GB-equivalent for INT4 ≈ 12.22 GiB/rank pinned). If device VRAM can't absorb it and host
  pageable breaks the serving path, the experiment is infeasible as configured → try
  PLE-enabled but connector memcpy serialization (code fix candidate B/C) next.
- A clean run could be boot luck (stall is stochastic). That's why it's 2 bursts minimum,
  and why the comparison is against a 4-in-6 background rate, not 0.

## Pre-registered prediction (falsifiable)
If PLE-disable boots and serves: prediction is ZERO stalls across both bursts (vs 4-in-6
background). If ≥1 stall occurs with PLE off, the mechanism as stated is WRONG (or
incomplete — L0 contention would then need another writer), and candidate B/C testing
(queue serialization / dedup-before-enqueue) inherits the queue.

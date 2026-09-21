# PLE v3 vs upstream 4e8b849b8d97 — reconciliation (2026-09-20)

**Question (Ryan):** upstream has a fix in the same code path as our v3 patch — does v3 supersede
it, conflict with it, or need re-basing on it?

## Rig evidence (stage-v24g, verified live 2026-09-20)
- `vllm/v1/ple_offload/connector.py:163` — still `queue.Queue(maxsize=1)`; `_input_ready_event` /
  `_d2h_done_event` still a single shared pair (:170-171, re-created :190-191). V3RUNNER markers ×7
  intact — **v3 did not touch this structure**.
- Upstream fix `4e8b849b8d97` ("fix PLE offload for async MRV2 scheduling") is **not in our tree**.
- Verdict: v3 does NOT supersede the upstream fix. The single-slot event race (#53960 stall 2) is
  **live in our build** whenever async scheduling gives `max_concurrent_batches == 2`.

## What each fix actually fixes
- v3 (V3RUNNER, 7 sites): PLE host-staging wedge class — L0-contention + offload staging path.
- `4e8b849b8d97`: event lifecycle under async scheduling — D2H copies on the main model stream +
  per-batch events instead of one shared pair.
- Different mechanisms, same file, overlapping invariants → **not independent patches**. v3 was built
  against the single-slot queue; per-batch events change the synchronization contract v3 was tuned on.

## Rebuild rule (container reconstruction on newer vLLM)
1. The new base likely carries `4e8b849b8d97` upstream — VERIFY in the rebuilt container (grep for
   per-batch event structure / absence of `maxsize=1`), never assume.
2. Re-apply v3 on top of the fixed connector. If a V3RUNNER site edits lines the upstream fix changed,
   prefer upstream's structure and re-derive the v3 behavior there; per-batch events may make some v3
   staging redundant — check each of the 7 sites individually, don't blind-patch.
3. Gate: rebuild-verify.sh L1 counts V3RUNNER (7×connector.py, 1×model_runner.py) — re-baseline the
   expected counts if sites merge during the re-base.
4. **The 12-way stall may disappear for free.** Our stall (EngineCore idle, wave recovery, duplicate-PLE
   skips) is consistent with the single-slot race partially dodged by the fork's duplicate-skip logic
   (fork-side, UNVERIFIED). If the new base carries the fix, test 12/16-way stability BEFORE any v3
   re-tuning work — one variable at a time.
5. Related, same feature, worth carrying into the new base check: PIECEWISE one-step-lag fix
   (peakcrosser7/vllm#13, 11 lines — we run FULL_DECODE_ONLY, likely unaffected but cheap to port),
   offload-worker CUDA context in memory profiling (#54905-adjacent), pidfd_getfd/Yama gates.

## #457 (Xe2 MoE GEMM uncapturable at batch>1) — applies or not?
jobe runs MoE + FULL_DECODE_ONLY graphs at batch 16 with 320 sustained on v0.1.8.3 → either #457 is
already fixed in 0.1.8.3, or it doesn't apply to the INT4 W4A16 MoE GEMM path (#457 concerns the
fused/ref MoE GEMM SYCL-graph path — a different GEMM than W4A16). Either way, treat it as a
**regression vector of the 0.1.8.3→0.1.14 upgrade, not an inherited bug**: after the upgrade, capture
16-way early (L1), and if DEVICE_LOST appears at MoE batch>1 capture, check #457's state in 0.1.14
first before kernel work.

# Lane 6 — custom kernel program (the "we write it ourselves" track)

> Opened 2026-09-14 per Ryan's standing authorization: "if we need to build
> the fix, the custom kernel, we do it." This doc scopes the two real
> kernel/engine builds queued after the phase-3 hardware upgrade, plus the
> config retrials. Nothing here is speculative-without-a-gate.

## Why custom work pays disproportionately on this platform

The B70 is not compute-starved — it is *kernel-starved*:

- Per card: 608 GB/s HBM + Xe2 XMX with **native e4m3/e5m2 FP8, INT8, INT4**
  matrix instructions. Pool across 4 cards: ~2.4 TB/s.
- Realized on Flash-Next today: the slowest vLLM number in the class table
  (46.85) because its FP8 runs on **fallback GEMMs** (oneDNN W8A8 per-linear
  dynamic quant for dense linears, Triton fp8_w8a8 for routed experts) while
  the same-class dense 27B at **106.7 tok/s** rides native INT4 grouped GEMM
  (`libgrouped_gemm_xe_*` — we hold these binaries in `files/a367-kernel/`).
- The delta between 46.85 and 106.7/170.91 on identical silicon is software:
  a small Intel XPU team vs thousands of CUDA kernel-years. Custom kernels
  here are low-competition, high-yield — every one we write lands on a
  platform where nobody else wrote it yet.

## Workstream A — MTP4 at 4K context (days-scale; +34% floor)

State: MTP4 measured **20.727 tok/s @512** (vs MTP3 14.889) but is
QUARANTINED at 4K — engine stall at **3,904/4,096** produced tokens.

Hypotheses to test in order (first light after RAM lands):

1. **KV reservation arithmetic** ignores the +k verify slots near the context
   cap → stall at 96% of max_model_len. Patch the XPU KV manager to reserve
   k+1 extra slots per sequence in the spec-decode path. Most likely culprit;
   it is a reservation-arithmetic bug, not a kernel bug.
2. **Persistent scratch sizing** (`VLLM_XPU_GDN_SPEC_PERSISTENT_SCRATCH=1`)
   at 4K — check allocation vs ctx-length scaling.
3. **Kernel row cap**: A367 (`ad25aa9` / `bbae3c5` over `e421889`) generalized
   exact serial GDN replay from 1 to **2 rows**. MTP4 needs **4-row** exact
   replay. Extend the same kernel — the exactness framework (lossless hash
   gates) already exists and generalizes.

Gates before any speed claim: regenerated lossless battery at 4 rows (the
`afffd211`/`1d833e5f` analog for this config), acceptance-rate ≥ MTP3's
799/852 reference, then e2e bench vs 46.85.

Expected: 46.85 × 1.34 ≈ **63 tok/s** (floor — Lane 2's own 512-ctx data).

## Workstream B — native block-FP8 grouped GEMM on Xe2 (weeks-scale; the ceiling raiser)

**UPDATE 2026-09-14 (recon hit): the native kernel already exists and ships
in our binary.** Evidence: `_xpu_C.abi3.so` (A367 rescue) contains
`xe2_block_fp8_small_m`, `cutlass_grouped_gemm_xe2`, and oneDNN w8a8
fallbacks as separate symbols; vllm-xpu-kernels master
(`fused_moe_interface.py`, PR #598 era) shows block-FP8 native Xe2 grouped
GEMM is **default-on** (`VLLM_XPU_FUSED_MOE_NATIVE_BLOCK_FP8` defaults "1"),
with the slow ref path only when explicitly disabled. There is also a BMG
(Battlemage) nightly benchmark workflow upstream.

So Workstream B collapses from "write the kernel" to **"audit and steer the
dispatch"**:

1. **Day-one dispatch audit (after first light):** instrument which MoE path
   Flash-Next actually takes in the lab stack (0.21.0-b1 + 16-patch overlay
   + kernel stage `2f829747`). The ceiling analysis assumed Triton
   fp8_w8a8 for routed experts — verify, do not assume: the binary has both
   paths compiled in, and which one serves decode is an empirical question
   (grep dispatch logs / wrap `cutlass_grouped_gemm_xe2` with a counter).
2. **If routed experts hit oneDNN/triton:** flip the env to force native
   (`VLLM_XPU_FUSED_MOE_USE_REF=0`, native block-FP8 on) and microbench.
3. **If already native:** the 46.85 line is a *tuned-native* baseline and
   the ceiling-raiser becomes the small-M kernel's tile/occupancy work
   (decode M≤4 shapes) against its own microbench.
4. Either way: microbench first (`benchmark/benchmark_gemm_onednn.py` +
   nightly BMG workflow as templates) at routed-expert shapes M=1/2/4 before
   any e2e claims.

Inputs we already hold:
- `libgrouped_gemm_xe_2.so` / `_default.so` + `_xpu_C.abi3.so` (rescued,
  sha-verified) — the compiled native paths are IN the binary.
- Upstream source mirror cloned locally (`~/xpu-kernels-recon`):
  `csrc/xpu/grouped_gemm/xe_2/` (gemm_xe2.hpp has block-FP8 scale handling,
  e4m3), `vllm_xpu_kernels/fused_moe_interface.py` (routing + env contract),
  `benchmark/` harnesses.

Gates unchanged: quality battery 7/7; acceptance-rate parity ±2%; microbench
≥1.5× before integration flips; e2e A/B vs A367 line; no speed number quoted
without the battery pass.

## Workstream C — graphs retrial (config, not code; after 128GB RAM)

Lane-1 quarantine cause (host-RAM OOM during post-load compile) is removed
by the RAM. Order: PIECEWISE first, FULL_DECODE_ONLY second, watchdog
mandatory (Xe2 wedge 2–6h). Gate: bit-exact outputs vs eager before
adoption. Precedent: same-class fully-built stack ran DSv4 at ~80 tok/s on
this box class.

## Sequencing and expected ladder

| Rung | Work | Expected |
|---|---|---|
| 0 | A367 eager MTP1 (certified) | 46.85 |
| 1 | A: MTP4@4K | ~63 |
| 2 | C: graphs | 80-class precedent |
| 3 | B: block-FP8 kernels | 100+ ambition |

A and C are post-RAM day-one/week-one work. B starts once A stabilizes (its
4-row verify rides the same test rig and battery).

## Risk register

- Xe2 Level-Zero wedge under sustained compile/bench load (2–6h) —
  wedge-watchdog.sh MANDATORY during B; never bench through a wedge.
- `SYCL_CACHE_PERSISTENT=1` poisons the B70 cache (SEGV next boot) — stay =0.
- oneDNN lesson generalizes: any new "faster path" (oneDNN, graphs, new
  kernels) gets microbenched + e2e A/B'd on the SAME client BEFORE adoption;
  three prior "should be faster" paths measured slower.
- Exactness gates are per-config: every new verify path regenerates its own
  lossless hashes; never reuse A367's hashes for a different config.

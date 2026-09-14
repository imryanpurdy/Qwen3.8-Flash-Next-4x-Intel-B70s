# A367 kernel-extension artifacts (exact serial GDN verifier line)

Rescued from jobe `~/xpu_artifacts/` on 2026-09-14 BEFORE the OS rebuild (Alex
wipes the boot NVMe; these were not in the recipe repo until now).

## Files + SHA-256 (full)

| File | Size | SHA-256 (first…last 8) |
|---|---|---|
| `_xpu_C.abi3.so` | 95,865,024 B | `593a7107…abc43b` |
| `libgrouped_gemm_xe_2.so` | 6,644,472 B | `2da4a494…f90d13` |
| `libgrouped_gemm_xe_default.so` | 1,247,176 B | `4b1ca1e6…b5161` |

`_xpu_C.abi3.so` = the A367 record-server kernel extension: rebuilt from the
lab kernel tree head `e421889` + commit `ad25aa9` ("Generalize exact GDN replay
to MTP row count"), patched series `bbae3c5` over `e421889` — accepts 2-row
MTP1 exact serial GDN verify (the stock build hard-gates exact mode to 4 rows).
This is the .so that took the two-row verify step 42.7 → 33.7 ms.

## Usage after jobe rebuild (128 GB RAM phase)

1. `IMAGE=qwen38-flash-next-xpu:0.21.0-b1-rt2f829747` (25.4 GB image) — if the
   image was also wiped, rebuild via repo `Dockerfile`, then drop these three
   .so files into the vLLM extension location inside the container (same paths
   the A367 record server used; see b70-optimization-lab repro
   `qwen38-flash-next-fp8-tp4-mtp1-exactgdn-b70-47tps-20260913/README.md`).
2. Exact-mode selectors (the A367 env):
   - `VLLM_XPU_GDN_SERIAL_SPEC_DECODE=0`
   - `VLLM_XPU_GDN_NATIVE_SPEC_RECURRENT_SERIAL_EXACT=1`
   - `VLLM_SPU_GDN_SPEC_PERSISTENT_SCRATCH=1`
   - `VLLM_XPU_GDN_NATIVE_SPEC_COMPLETION_BARRIER=1`
3. Quality gates on that line: outputs bit-identical to the certified battery
   (exact-2K `afffd211…`, exact-4K `1d833e5f…`, 7/7 exact cases).

## Also rescued

- `hybrid.patch` — the 12-file ucicelos/flashnext-hybrid patch
  (`691a81d`), applies clean 12/12 at llama.cpp lineage `337c8bb58`. Kept as
  the Lane-1 (llama.cpp) record only; Ryan's decision (2026-09-14): the Flash-
  Next lane goes vLLM/XPU after the hardware upgrade, llama.cpp is retired.

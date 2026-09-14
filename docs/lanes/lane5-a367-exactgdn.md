# Lane 5 — A367 exact serial GDN line (46.85 tok/s, MTP1, batch-1)

> Status: **READY-TO-DEPLOY** after the phase-3 hardware upgrade. This is the
> fastest certified Flash-Next line (steveseguin/b70-optimization-lab repro
> `qwen38-flash-next-fp8-tp4-mtp1-exactgdn-b70-47tps-20260913`, cert
> `lab-replay`). Certification: `lab-replay` on A364–A367; quality gates are
> bit-exactness, not benchmarks.

## What this line is

The 37.83 tok/s line verified the single speculative token by running the GDN
(gated delta net) verifier rows one at a time through vLLM's **Python serial
path** (accepted state → spec col 0, row 0 → single-row decode kernel, col 0 →
col 1, row 1 → decode kernel; 8.7 ms of the 42.7 ms two-row step). The kernel
extension already contains an exact mode doing the same per-row decode inside
the single `gdn_attention_spec_decode` op, but hard-gates it to 4 verifier
rows. A367 rebuilds `_xpu_C.abi3.so` from kernel tree head `e421889` + commit
`ad25aa9` ("Generalize exact GDN replay to MTP row count", series `bbae3c5`)
so exact mode accepts **2 rows (MTP1)**, and selects it via env.

- Headline: **46.854250 tok/s**, median of prompt-class medians over 99
  inter-token intervals after TTFT, fixed cold 12-prompt realistic suite
  (A367, 2026-09-13). Ladder to it: 27.0 → 31.9 → 37.8 → 46.9.
- Two-row verify step: 42.7 → **33.7 ms**; every row class +23–24%.
- Model pin: `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` (131 shards,
  185,563,783,127 B) — unchanged from the repo's frozen identity.
- vLLM overlay `6d872457`; kernel stage `2f829747` (= the `rt2f829747` image
  tag); `_xpu_C` rebuild `bbae3c5` over `e421889` (commit `ad25aa9`).

## Exact-mode env (the four selectors)

```bash
VLLM_XPU_GDN_SERIAL_SPEC_DECODE=0
VLLM_XPU_GDN_NATIVE_SPEC_RECURRENT_SERIAL_EXACT=1
VLLM_SPU_GDN_SPEC_PERSISTENT_SCRATCH=1
VLLM_XPU_GDN_NATIVE_SPEC_COMPLETION_BARRIER=1
```

Everything else (MoE kernel + tuned map, offload, placement, collectives,
model) is untouched by this line.

## Kernel artifacts (rescued 2026-09-14, vendored at `files/a367-kernel/`)

| File | Size | SHA-256 (first…last 8) |
|---|---|---|
| `_xpu_C.abi3.so` | 95,865,024 B | `593a7107…abc43b` |
| `libgrouped_gemm_xe_2.so` | 6,644,472 B | `2da4a494…f90d13` |
| `libgrouped_gemm_xe_default.so` | 1,247,176 B | `4b1ca1e6…b5161` |

Provenance: built on jobe 2026-09-05 from the lab kernel tree (`e421889` +
`ad25aa9` + series `bbae3c5`; two inert env-gated disclosure commits on top).
If re-derivation is ever needed instead of the drop-in, follow the lab
series README `xpu-kernels-gdn-exact-serial-bbae3c5`.

Install: replace the extension .so inside the running image's vLLM extension
location (paths per lab repro README), then set the four selectors. The
quality gates below fail loudly on a wrong drop — do not bypass.

## Serving block (phase-3 `.env`)

```bash
MTP_NUM_SPECULATIVE_TOKENS=1
MAX_NUM_SEQS=1
MAX_MODEL_LEN=4352
GRAPH_MODE=eager
# + the four selectors above
```

Record context: batch-1, 4.3K ctx, FULL_DECODE_ONLY graphs were Steve's lane
context — graphs are still Lane-1-quarantined here; the 46.85 receipt stands
on eager. TTFT on this line: median 521 ms, worst 77.2 s (MBT=64 smoking gun,
Lane 4 still open).

## Quality gates (all must pass before quoting throughput)

1. **Exact-2K** output pin `afffd211…` — byte-identical.
2. **Exact-4K** output pin `1d833e5f…` — byte-identical.
3. **Battery**: 7/7 exact cases byte-identical to the certified battery
   (one inherited `code_execution` miss, documented in the lab packet).
4. **Repeats**: 16/16 collapse to one hash.
5. **Long-context needle**: identical (A364/A365/A366/A367 receipts).

Known numbers on the same identity (long-context diag A382/A394): decode
42.7 @8K / 45.6 @16K / 44.1 @32K; TTFT 47.7 s @8K → 206 s @32K; served cap
33,280; first-use rows ~29.3 vs warm ~47.2 (page-cache effect — warm the
table before quoting).

## Open items this lane inherits

- Lane 4 MBT sweep (64 → 512 → 2048 → 4096) — the 77.2 s worst-case TTFT.
- Lane 1 graphs retrial under 128 GB RAM (see `docs/phase3-hardware-upgrade.md`).
- Long-context TTFT (206 s @32K) — placement work, not this lane's blocker.

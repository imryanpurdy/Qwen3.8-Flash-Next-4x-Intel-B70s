# Phase 3 — hardware upgrade transition (2026-09-14, Ryan)

Jobe is being rebuilt by Alex: **OS moves to a new SSD, the 232 GB NVMe is
reformatted as the data volume (FP8 weights + caches), and 128 GB ECC RAM
lands 2026-09-15.** This document is the recipe for everything that changes
when the box comes back. The frozen lab identity does NOT change — only the
floors, the placements, and which lanes unlock.

## What the upgrade changes

| Floor/placement | Before (30 GB RAM) | After (128 GB RAM) |
|---|---|---|
| `PREFLIGHT_RAM_GB=100` | FAILS (30) → `--no-preflight` required | PASSES (128) |
| Swap ≥64 GiB | impossible (no disk headroom) | TRIVIAL — create 64 GiB+ on the NVMe data volume, keep ON |
| PLE pinned-UVA (12.22 GiB/rank × 4 ≈ 51.2 GiB) | fits barely, thrashes | fits with ~70 GiB headroom |
| Lane 1 PIECEWISE graphs (a1–a7 quarantine) | dead (host-RAM OOM during compile) | **RETRY** — the RCA was host-RAM exhaustion during post-load compile; 128 GB removes the death mechanism. Set `GRAPH_MODE=piecewise` once start.sh's guard is lifted (see below) |
| Weights tree (185.56 GB) | squeezed onto OS disk | NVMe data volume, `HF_HOME` bind-mounted there |
| OS disk free ≥40 GiB | 96% full, ENOSPC incidents | SSD dedicated to OS — non-issue |

## Weights + caches layout after rebuild

- Format the NVMe as ext4/xfs (NOT zfs — the 931 GB WDC `sda` remains
  `zfs_member` and inert unless Alex imports it), mount at `/data`.
- `HF_HOME=/data/hf` — the checkpoint re-downloads on first `./start.sh`
  (auto-skip check will pass after; `check-weights.sh` enforces identity
  `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` either way).
- Swapfile: `fallocate -l 96G /data/swapfile && chmod 600 /data/swapfile &&
  mkswap /data/swapfile && swapon /data/swapfile` + fstab entry.
- `PREFLIGHT_DISK_GB=200` expects ≥200 GiB free on the weights mount: 232 GB
  NVMe − 185.56 GB weights ≈ 46 GiB — **this floor will fail**; either trim it
  in `.env` (justified: caches/swap live on the same volume and swap is now
  huge) or point the weights mount at a bigger volume. Documented, deliberate.

## The headline recipe: A367 exact-GDN line (46.85 tok/s)

Target identity after the box returns — see `docs/lanes/lane5-a367-exactgdn.md`
for the full procedure and artifact hashes:

1. Build the image (repo Dockerfile) → tag `qwen38-flash-next-xpu:4xb70`.
2. Drop in the A367 kernel extension `_xpu_C.abi3.so`
   (sha256 `593a7107…abc43b`, rebuilt from kernel head `e421889` + `ad25aa9`,
   series `bbae3c5` — 2-row exact serial GDN verify; stock build hard-gates 4
   rows). Copy of the exact A367 binaries survives in
   `files/a367-kernel/` in this repo as of 2026-09-14 (rescued off jobe before
   the wipe; ~104 MB, committed via LFS-style blob storage note below).
3. `.env` phase-3 serving block:

```bash
MTP_NUM_SPECULATIVE_TOKENS=1          # the A367 record line is MTP1
GRAPH_MODE=eager                      # first light stays eager; graphs after Lane-1 retrial
VLLM_XPU_GDN_SERIAL_SPEC_DECODE=0
VLLM_XPU_GDN_NATIVE_SPEC_RECURRENT_SERIAL_EXACT=1
VLLM_SPU_GDN_SPEC_PERSISTENT_SCRATCH=1
VLLM_XPU_GDN_NATIVE_SPEC_COMPLETION_BARRIER=1
MAX_NUM_SEQS=1                        # A367 record = batch-1 single-stream 46.85
MAX_NUM_BATCHED_TOKENS=64             # Lane-4 sweep still pending; do not raise blind
```

4. Quality gates (must reproduce): exact-2K `afffd211…`, exact-4K
   `1d833e5f…`, needle identical. If outputs diverge, the kernel drop is wrong
   — stop and re-derive from the lab tree, do not serve.
5. Throughput gate: median-of-medians ≥ 46.0 tok/s on the fixed cold
   12-prompt suite (A367 method). Below that, the kernel ext is not the A367
   lineage — check the sha, check the four selectors, check `MAX_NUM_SEQS=1`.

## Lane-1 graphs retrial (un-quarantine checklist)

With 128 GB the a1–a7 OOM cause is gone. Run in this order, one variable at a
time, and update `docs/lanes/lane1-piecewise-graphs.md` with results:

1. `GRAPH_MODE` guard in start.sh still hard-fails any non-eager value — lift
   it for `piecewise` ONLY after this retrial passes capture on the target.
2. Load model eager → confirm health → attempt PIECEWISE capture on the draft
   model first (smallest graph), then target, then full decode.
3. Gate: bit-identical outputs to the eager A367 line on the exact-2K/exact-4K
   pins before comparing throughput.
4. If capture wedges a card (the 2–6 h Xe2 Level-Zero wedge), the mandatory
   wedge-watchdog handles it — do not disable it.

## Speed ceiling and lever ladder (2026-09-14, post-cookbook v1.2.1 analysis)

Reference points from the same hardware class (SergiioB cookbook hub, all
vLLM XPU, single-stream C1 client post-first n=5):

| Model | Engine/spec | tok/s |
|---|---|---|
| Qwen3.6-35B-A3B (3B active) | MTP4 | **170.91** |
| Nemotron-3.5-Lightning-30B-A3B | DFlash n=7 | **186.61** |
| Qwen3.8-27B dense GPTQ-INT4 | MTP4 | **106.7** |
| **Qwen3.8-Flash-Next FP8** | MTP1 (A367) | **46.85** |

Flash-Next is the slowest vLLM number in the table despite being the flagship
— because it carries four compounding handicaps, each a documented lane:

1. **FP8 runs on fallback GEMMs.** Dense linears = oneDNN W8A8 per-linear
   dynamic quant; routed MoE = Triton fp8_w8a8 — while the 106.7-tok/s dense
   27B runs NATIVE INT4 kernels. Lane 3 (native block-FP8 grouped GEMM,
   dormant upstream) is the fix. Expected: large; ungated today.
2. **Speculation capped at k=1.** The lossless-verify line runs MTP1; MTP4
   measured +34% over MTP3 at 512 ctx (20.727 vs 14.889) but is QUARANTINED
   at 4K (3,904/4,096 engine stall). Lane 2 qualification = the single
   biggest *known* lever, ~+34% floor by its own 512-ctx data.
3. **PLE pinned-UVA synchronous reads** during decode (51.2 GiB host).
   128 GB makes it painless but it still costs bandwidth per rank.
4. **GDN+QSA hybrid serial verify** — what A367's exact-mode kernel already
   partially fixed (42.7→33.7 ms/step). Further fusion = kernel work.

Negative result to respect (SergiioB, Sep 13): DFlash2 on 27B = 25 tok/s vs
51 for plain MTP4 — a second full model as drafter LOSES on B70; native MTP
heads win. Flash-Next's native 4B MTP head is the right instrument; the play
is qualifying it at k=4, not adding draft models. (Counter-example kept for
honesty: Nemotron hit 186.61 WITH DFlash n=7 because its native MTP is 0% —
broken native head makes a heavy draft the only option.)

Target ladder for phase 3 (each step gated, none speculative without a gate):

| Step | Lever | Gate/expectation |
|---|---|---|
| 0 (day one) | A367 eager MTP1 | ≥46.0 (certified floor) |
| 1 | Lane 4 MBT sweep 64→2048 | kills 77.2 s worst-case TTFT; GLM-5.3 precedent says 2048 |
| 2 | Lane 2 MTP4@4K qualification | ≥ +30% decode by Lane-2's own 512 data |
| 3 | Lane 1 graphs (PIECEWISE) retrial | bit-exact gates, then DSv4-precedent upside (80-class on this box) |
| 4 | Lane 3 native block-FP8 | removes the fallback-GEMM tax; preregistered gates in lane doc |

Fallback if vLLM stalls: SergiioB published fused multi-token llama.cpp
patches (`qwen4exp-mtp-draft-head.patch`, `sycl-fused-mmvq-mt.patch` @
llama.cpp `52d4268`, Sep 12) — the MMQ-class kernel work our lane diagnosed
as missing, published and hash-pinned. Recorded as plan B; primary stays
vLLM per Ryan's 2026-09-14 decision.

## Retired lane (record only)

The llama.cpp SYCL lane (lineage `337c8bb58` + hybrid.patch `691a81d`, 12/12)
is retired by Ryan's decision 2026-09-14: it delivered first light at 29.8
tok/s solo warm / 22.2 @PAR=4 on GGUF 4.27bpw, and its bottleneck was
diagnosed to PLE page-cache thrash on 30 GB RAM (38.4 GB n-gram table
host-side, bimodal 13↔29.5 reps, 46 MB/s NVMe streaming during decode) plus
the iq2_s/iq3_s SYCL MMVQ chain (~33 ms/token). The patch, build scripts, and
A367 kernels were rescued to this repo and to the operator workstation before
the wipe. GGUF source (91 GB) is NOT retained; re-downloadable from
AtomicChat rev `142262902a46f7daed19c79d0771534c8106ad59` if ever needed.

## Ops notes carried into phase 3

- `docker container prune` killed the stopped 27B container once already —
  NEVER prune on this box; the 27B lane revives via `docker run` from the
  MikeCaldera recipe (image `vllm-xpu-b70:26.31-test`, port 11438,
  `max_num_seqs=4`, MTP5, graphs FULL+PIECEWISE) — after 128 GB the RAM
  conflict with Flash-Next is also gone, but the cards are still exclusive.
- DLE-2026 image ships no oneDNN; llama.cpp `-DGGML_SYCL_DNNL=ON` = 3× slower
  on Battlemage (do-not-retry). Irrelevant to vLLM but recorded.
- B70s train PCIe x1 Gen1 on this riser topology — permanent, activations
  still fit; don't chase it.

# DeepSeek V4 Flash uniform-K160 lane — deep read

Sources (read thoroughly, in order):
- `results/deepseek-v4-flash-k160-b70/` — README.md, bugs-failed-paths.md, validity-gates.md
- `repro/deepseek-v4-flash-k160-b70-80tps-20260718/` — README.md, run.sh
- `experiments/deepseek-v4-flash-reap-xpu-b70/` — ORCHESTRATOR_HANDOFF.md, HANDOFF.md, results/experiment-ledger.md, benchmarks/stage-gates.md, notes/* (60+ dated notes)
- The pinned record diff: `patches/deepseek-v4-flash-reap-xpu-b70/vllm-deepseek-v4-k160-dspark7-80tps-record-20260718.patch`
- Note: there is **no `experiment-ledger.md` inside results/deepseek-v4-flash-k160-b70/**; the ledger lives at `experiments/deepseek-v4-flash-reap-xpu-b70/results/experiment-ledger.md` (linked from the packet).

## 1. Final verified numbers and exact configuration

**Record (approved LocalMaxxing `cmrquta9905w3lg013m5vxoqx`):**
| Quantity | Value |
| --- | --- |
| Strict-suite medians (tokens 1-100 after TTFT, 12 cold prompts ×3) | **80.820052** / 76.900178 / **78.287226** tok/s (median-of-medians 78.287226) |
| p10 on high suite / wall full-128 | 71.669556 / 67.762818 tok/s |
| Validity | 36/36 realistic requests cache-zero; 24/24 ordered exact canaries (4× six-case suites: 1073→437→1073 arithmetic, exact-copy, factual, strict-JSON) |
| LocalMaxxing | `cmrquta9905w3lg013m5vxoqx` |

**Exact identity (fail-closed launcher `run.sh`):**
- Model: `0xSero/DeepSeek-V4-Flash-180B`, revision `7c360e1cd4a5168099dbc54d16d929bf6df04990` (~96.0 GiB; uniform K160 = 160 global experts, 40/EP rank; MTP shard 46).
- Hardware/topology: 4× Intel Arc Pro B70 32 GB (level-zero 0-3), **TP4 + EP4, DP1, PP1, concurrency 1**; one active generation (never aggregate).
- Runtime commits: vLLM `264c7f2f7df21ddeeab32ecca0353133344f1ac9` (public anchor `382bbd51…`); vLLM-XPU-kernels `31315673737d95da0f79179c8f755260ef02c1d6`; **oneCCL `48fda4f0e074db005596d6899d5227d3f0316c12` = oneCCL v2021.17.2 "wide-epoch / size-routed B70" build** (binary SHA-256 `53de2b6d65265803d64773546c1166ceed4ae43737f0fded776f5847b4b461c9`, preloaded `ONECCL_FORCE_PRELOAD=1`, routes only all-reduces >131,072 B through the safe SYCL path — `B70_ONECCL_SYCL_ALLREDUCE_MAX_BYTES=131072`).
- Quantization per component: **FP8 block-scaled dense weights** (UE8M0/E4M3, block 128, oneDNN BF16-act/FP8-weight W8A16 for M≤8 dense families; selective `VLLM_XPU_V4_BLOCK_FP8_W8A16_MAX_M=8`); **MXFP4 (E8M0-scale N128) routed experts** (`VLLM_XPU_MXFP4_SMALL_M_N=128`, exact M8); FP8 KV **UE8M0, block 256**; activations BF16 (clamp-at-10 SwiGLU boundaries preserved).
- **KV dtype: FP8 (UE8M0 scale, compressed)** — KV cache `2.11 GiB/rank`, 5,925 KV tokens at 2K/95% (or 160 tokens @ 64 MiB); 576 data + 8 scale bytes per slot offset (from the divergence-capture gather code in the record patch).
- Graph mode: **target PIECEWISE** and **private breakable draft PIECEWISE** exact-M7 (both `piecewise`); `CompilationMode.NONE` on record lane (breakable graphs disable Dynamo/Inductor).
- Speculative decoding — **DSpark7**: official three-stage, 256-expert DeepSeek V4 Flash DSpark draft (source rev `aa22cb07426656189b2573b8e77a9b7333b8ae0f`, pack `dspark-draft-pack-aa22cb0`), `DSPARK_SPEC_TOKENS=7`.
  - **Exact M=7 capture detail:** `VLLM_XPU_DSPARK_EXACT_QUERY_CAPTURE=1` + `VLLM_XPU_DSPARK_FIXED_M7_TARGET_INPUTS=1` — the private breakable draft graph captures **only the fixed M=7 draft query** (NOT padded to M=8; padded-M8 measured only 60.518 tok/s vs 64.661 for exact-M7; draft-eager only 55.11). A prior capture-design bug warmed M=1/2/4/8 and entered a target-only router specialization that rejected the 256-expert draft; fixed by capturing the smallest fixed descriptor that serves the query.
  - Target verifier runs at **M=8** (M+1 rows), unchanged K160 target, `VLLM_XPU_V4_COMPRESSOR_BATCHED_EXACT_MAX_M=8`, `VLLM_XPU_V4_BLOCK_FP8_W8A16_MAX_M=8`, `VLLM_XPU_V4_ROUTER_NORM_MAX_M=8`.
  - Markov sampler: **persistent Markov with W1-only replication** (`VLLM_XPU_DSPARK_PERSISTENT_MARKOV=1`, `VLLM_XPU_DSPARK_REPLICATED_MARKOV_W1=1`; W2 stays sharded) — removes seven all-reduces; +0.452 ms/cycle over full-sharded.
  - Final verifier: **guarded sharded greedy target argmax + native target-token rejection** (`VLLM_XPU_GREEDY_SHARDED_TARGET_ARGMAX=1`, `VLLM_XPU_GREEDY_FUSED_REJECTION=1`): each rank projects only its local 32,320-token LM-head shard, local top-1, tiny (value,index) all-gather, native SYCL op commits with ordinary rejection/bonus; falls back to canonical full-logit path for any non-greedy/grammar/logprob/penalty config.
  - Draft acceptance (DEV screen): 79.64/74.28/68.83/71.07/70.80/66.88/58.88% conditional per position 1-7; ~3.5 emitted tokens/cycle.
- oneCCL caveat: earlier `40.136` row used the pre-repair readiness rollover allreduce → "Do not use the pre-repair 40.136 row as repeatability authority."

## 2. The MLA / sparse-attention implementation — CRITICAL

**Verdict: the sparse MLA decode attention used on the record lane is TRITON, not SYCL.** It is a two-kernel split-FlashAttention over a deepseek-specific paged *compressed*-FP8 KV layout, with the block-sparse *indexer* present in the model but **bypassed at the 1K record context via "full selection"**. SYCL native ops are used for the *peripheral* MLA pieces (QNorm/RoPE/KV-insert, inverse-RoPE/WO_A, MQA-logits fallback, MHC).

**Files/kernels (from the record patch `vllm-deepseek-v4-k160-dspark7-80tps-record-20260718.patch`):**
- `vllm/models/deepseek_v4/attention.py` — `DeepseekV4Attention` (MLA layer: WQA/WKV fused Q/K/V-latent GEMMs, compressor, indexer, fused Q/K-V RMSNorm+RMSNorm-upcast, MQA row-attention, inverse-RoPE + WO_A + WO_B output).
- `vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` — **the production decode attention: `split_fp8_sparse_attention()`.** Launches **Triton** kernels (file imports `from vllm.triton_utils import tl, triton`):
  - `_fp8_sparse_qk_lse_kernel` (QK/LSE stage), `num_warps=8`, `BLOCK_K=16`, `BLOCK_H ∈ {4,8,16}`.
  - `_fp8_sparse_pv_kernel` (PV stage), grid `(q.shape[0], head_blocks, 8)`, `num_warps=4`, `BLOCK_D=64`.
  - Config knobs used by the record: `VLLM_XPU_V4_SPLIT_FP8_BLOCK_H=4`, `_QK_NUM_WARPS=16`, `_PV_NUM_WARPS=4` (sixteen 4-head QK programs; +17.48% over the 16-head/8-warp geometry, 40.02 tok/s 2026-07-15 record step).
  - These read **paged UE8M0 FP8 KV directly** (no BF16 staging), apply **device-side runtime lengths** and a **learned attention sink**, and mask sparse chunks with a finite sentinel so graph padding can't produce NaNs. The old `_bf16_mla_sparse_kernel` is the BF16 prefill/reference path only — NOT used in record decode.
- `vllm/models/deepseek_v4/xpu/xpu_sparse.py` — SWA/compressed-cache Python wrapper; contains the divergence-capture gather showing the **FP8 cache row format (576 B data + 8 B scales/offset)**; emits `combined_indices`/`combined_lens` (fixed-width device-only index packing, from 2026-07-14 `0ed5ecc5` graph-recovery).
- `vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py` — **Triton** `_xpu_qnorm_rope_fp8_insert_kernel` (fused Q-RMSNorm/RoPE + FP8 KV write, grid `(num_tokens, num_heads+1)`), promoted record component (+2.02-2.08×).
- vLLM-XPU-kernels **SYCL** custom ops (record patch): `csrc/xpu/sycl/deepseek_qnorm_rope_kv_insert.cpp` (native variant), `deepseek_inv_rope_bf16.cpp`, `deepseek_inv_rope_fp8_quant.cpp`, **`deepseek_fused_indexer_q_rope_fp8.cpp` (fused indexer-Q projection + RoPE + FP8 quantization — the XPU implementation of the indexer's query-side projection)**, `deepseek_m1_biased_topk.cpp`, `deepseek_shared_down_m1_fp8_dpas.cpp`, `deepseek_wqb_m1_fp8_gemv.cpp`, `deepseek_scaling_rope.cpp`, and the `mhc/` family (`mhc_pre`, `mhc_post`, `mhc_fused_post_pre`, `hc_head_fused`, `mhc_tp4_allreduce`), plus samplers.
- **Native fallback flash-attn op `fp8_paged_mqa_logits`** (`fp8_paged_mqa_logits_kernel_t` / `xe2`, registered `xpu_ops.impl("fp8_paged_mqa_logits", torch::kXPU, …)`, same source family later reused by the Laguna width-12 record). This is the graph-blocking op (uses `sycl_ext_oneapi_work_group_scratch_memory`, unsupported in SYCL graphs) and is NOT the record decode path.

**Block-sparse / indexer-based selection of KV history:**
- Yes — the model is DeepSeek's **Native-Sparse-Attention-style** MLA: per-layer **SWA (sliding-window resident)** + **compressed KV blocks at compression ratios C1/C4/C128** + a **Lightning Indexer** that scores compressed blocks (index-weight projection + indexer compressor) and selects a subset to attend over (the "indexer select/sort" step, ~0.98 ms/token in the 2026-07-14 profile).
- The indexer lives in `attention.py` (`indexer.compressor.fused_wkv_wgate.weight`, `indexer_weights_proj` producing `indexer_kv_score`/`indexer_weights`); its query side is fused on XPU by the SYCL `deepseek_fused_indexer_q_rope_fp8` kernel; its top-k/radix selection is generic torch (initial hotspot suspicion was disproved by exact trace — it was 40 `[2,160]` K=6 target-router calls, not the indexer).
- **XPU failure modes of the indexer/graph path (all documented):**
  1. **Graph-capture host-wait:** original sparse-FP8 decode sized/packed its index tensor with device-to-host scalar reads starting `combined_lens.max().item()` → Level Zero command graphs can't wait on that device event during capture; even a capture would freeze the sequence-length max (not part of the graph key) and silently truncate indices on replay. Fixed (`0ed5ecc5`) with fixed-width, **device-only** index packing + finite masked-chunk sentinel.
  2. **Native paged-indexer scratch memory can't enter SYCL graphs** — `fp8_paged_mqa_logits` uses `sycl_ext_oneapi_work_group_scratch_memory`, unsupported inside the current SYCL graph → "narrow FULL-mode eager break" only around that op (`436298dcd`); PIECEWISE stayed correct. FULL+narrow break beat PIECEWISE by only 1.22% → graphing the rest of sparse attention wasn't the bottleneck.
  3. **First Triton pack attempt segfaulted the Intel Triton driver** (preserved negative); the checked-in static PyTorch pack is the bootstrap impl.
  4. At the **record lane's 1,024-token context, C4 full-selection bypasses the indexer** — production does NOT execute the 64-wide index-weight or 512-wide indexer-compressor projections (valid because ≤256 compressed C4 candidates exist for a top-512 budget); so the actual top-k indexer select was **never exercised end-to-end on XPU in this lane** — the operational sparse attention was SWA-resident + full C4/C128 + learned sink, run by the two Triton kernels. "Generic C4 fusion" attempts that reintroduced indexer work were rejected (39.68 tok/s) and the record bucket pair (`swa-resident-anchor64` / `compressed-swa-full-anchor512` at anchors 64/512) is fixed.
- Record-lane attention cost (2026-07-15 corrected eager profile): tuned split FP8 QK+PV ≈ **1.452 ms/token**; split QK alone 2.74 ms/token pre-geometry-change; final post-record target-verified profile: sparse QK/LSE ~3.54 ms, PV ~1.78 ms per 27-27.6 ms cycle.

## 3. Model-agnostic vs DeepSeek-specific components (classification)

| Piece | Kind | Class |
| --- | --- | --- |
| Split-FP8 Triton sparse attention (QK/LSE + PV over paged UE8M0 FP8 compressed KV, device-side lengths, learned sink) | Triton kernels | **(b) MLA-specific** (multi-query-row attention over compressed MLA KV; would need re-shaping for other archs but the sparse-2-kernel split pattern is reusable) |
| Fixed-width device-only index packing + masked-chunk sentinel (graph-safe) | vLLM Python | **(a) reusable** — generic graph-safety pattern for ANY sparse/block attention KP |
| Lightning Indexer (weights projection, indexer compressor, select/sort) | checkpoint weights + vLLM | **(b)+(c) MLA/DeepSeek checkpoint-specific** (arch weights); the "full-selection bypass at short ctx" is a reusable *policy* |
| Compressor (C4/C128 KV compression, fused WKV/gate, strided-batch exact M8 FP32 BMM) | vLLM + oneDNN | **(b) MLA-specific** |
| Fused indexer-Q+RoPE+FP8 SYCL kernel, fused QNorm/RoPE/KV-insert (Triton+SYCL), inverse-RoPE/WO_A, WO_B | kernels | **(b) MLA-specific** |
| Native greedy sharded target-argmax + rejection/commit, persistent Markov W1-replicated sampler, exact-M7 draft PIECEWISE | vLLM + SYCL | **(a) reusable** speculation/verification machinery for any greedy single-stream MoE; only LM-head width is model-specific |
| MXFP4 routed MoE (E8M0 N64/N128, grouped GEMM Xe2, biased top-k SIMD16), exact router norm, shared-expert clamped-SwiGLU+FP8-quant | SYCL (grouped_gemm_xe2, deepseek_m1_biased_topk, shared_down…) | **(a) reusable** — sparse MoE routing kernels (names/bias params are V4-shaped but generic MXFP4/int4) |
| MHC (multi-head compressor? — "mHC" = the per-layer KV-compression head stack: post/pre fused 256-lane Xe2 kernel, TP4-ring fused post, exact M4/M8 fixed-width variants) | SYCL | **(b) MLA-specific**, but the fused producer/consumer geometry transfers |
| oneCCL wide-epoch size-routed runtime (24-bit collective epoch), 131,072-B threshold | oneCCL patch | **(a) reusable** — fixes deterministic allreduce corruption on Arc |
| DSpark draft pack (official 3-stage + Markov corrections) | trained weights | **(c) DeepSeek checkpoint-specific** |
| Option-4 raw Level-Zero command-list replay substrate (`zeCommandListImmediateAppendCommandListsExp`, 0 host syncs) | C++/LZ | **(a) reusable** for any fixed-geometry decoder |
| Exact-identity divergence-capture / real-cycle corpora + 70/70 replay worker | harness | **(a) reusable** |

## 4. Negative results / wedges / driver issues (bugs-failed-paths.md + experiment-ledger.md + notes)

From `bugs-failed-paths.md` (closeout grouping):
1. **M=1 decode occupancy/latency starved** — tile prepacking, split-N workgroup expansion, GRF tuning, submission collapse did not improve the endpoint (nonspec roof 151 tok/s; ~200 tiny kernels/token).
2. **Option 4 raw Level-Zero replay** proved feasible but the guarded M1-attention endpoint **regressed (−0.182 ms/token) and failed cross-run exact tokens** (9/12); submission-collapse caps at ≤0.9 ms because device-kernel execution (~9.4 ms) is untouched.
3. **MHC arithmetic shortcuts changed greedy tokens** → quality-rejected (generic TF32 DPAS MHC fast but returns `1053` instead of `1073`; fused RMS changes ~50% of tokens).
4. **Full draft replay and combined sampler/model replay corrupted outputs** (obvious repeated/corrupted output; 62.46 tok/s sampler-only graph with 83 kernels kept but not promoted).
5. **Fixed DSpark5 and several exact DSpark7 fusion/transport candidates slower than record** despite positive isolated gates: DSpark5 53.87; padded-M8 draft 60.52; context-WKV fusion −0.611 ms local but endpoint 64.27; copy-elision 64.76; native-K2, persistent-K-step (~0), M2 chain (+0.11) all sub-gate; IPC-event M7 bundle exact (−0.994 ms) but endpoint 67.23; width-aware split-FP8 attention endpoint 72.46/73.40; ordinary tiny-pair Markov transport 73.46; host-barrier winner transport 74.84/75.76.
6. **oneCCL readiness rollover and large-SYCL-allreduce corruption** (deterministic failures at graph captures 28 & 58; `[4,4096]` wide collectives corrupt 427k mismatches/rank) — repaired by pinned wide-epoch, size-routed oneCCL identity. **Do not use pre-repair 40.136 row.**
7. Projections: ~90 tok/s needs 10-20M-token EAGLE corpus; **151 tok/s is a bandwidth roofline, not measured; 160 nonspec physically impossible** (needs 613 GB/s/card > 608 spec).

Triggering conditions for notable wedges/device issues (from ledger/notes):
- `UR_RESULT_ERROR_DEVICE_LOST` after Xe timeout storm (rank 1, PCI 27:00.0) 2026-07-14 → recover via targeted xe unbind/rebind (PCI reset left GT PF self-config errors); also during persistent-kstep K16 5th reload, MXFP4 tile-major prepack card 0 (reproducible at 4th schedule; isolated retry too), and 0731-era Flash-Next queue losses.
- ASPM performance-policy flip → three cards stuck in D3cold/runtime-error; FLR/unbind/rebind/bus-reset all failed → **reboot required; do not repeat mutation**.
- DPEP all-gather stall (`MoEPrepareAndFinalizeNaiveDPEPModular`) — TP2×DP2×EP4 unusable; fast-SYCL communicator-switch cycle between disjoint TP and crossed DP pairs.
- MTP2 reuse deadlock: second draft position accepted 0.5-2.2%, then request hung + exhausted shared-memory broadcast blocks for 180 s.
- `IndexError`-class packing failures, `-k 40` selector matching E=64 via dim 4096 (fixed), sycl9/urDeviceWaitExp JIT ABI mismatch (quarantined Triton caches), MXFP4 global atomic counter-reset race breaking N32/N128 graph replay (ordered reset fix reverted as immaterial/slower).
- `wo_a` scale-layout corruption: prepack transposed DeepSeek's special WO_A BMM scales → 30.295/33.887 records **invalidated**; corrected `61c87db6` restored canonical `[N/128,K/128]` scales (corrected records 30.239/33.43).
- Approximate Triton compressor GEMV changed all 12 suite hashes (39.72 tok/s) — rejected; strict bitwise parity required for compressor.
- DSpark no-training screen: confidence-gated prefix trimming and M=4 (invalid geometry), M=5, M=6, M=8 all lose to M=7; DEV winner 79.80 < record 80.82 → no held-out candidate; EAGLE head trained on 1M-token corpus plateaus ~67-68% P2-7 (overfitting, DATA-limited).
- 0731-era host blocker: **105 GiB MemAvailable boot policy** — no DeepSeek GPU launch on 2026-08-28.

## 5. The two checkpoint caveats (verbatim) and lane closeout vs 0731 release

**Caveat A (result packet README.md, lines 48-51):**
> "The public uniform-K160 checkpoint is a useful experimental performance artifact, not a quality-certified official REAP construction: **its hash layers are pruned and its calibration/ranking provenance is unavailable.** Keep that caveat attached to every result."

**Caveat B (validity-gates.md, lines 19-21):**
> "The public K160 checkpoint is **hash-pruned and its calibration is not reproducible.** The gate supports a result for this exact experimental artifact; it does not certify the artifact as official DeepSeek V4 Flash or true REAP."

Supporting statements elsewhere: repro README — "experimental hash-pruned uniform-K160 artifact with unavailable calibration; it must not be described as the official checkpoint or as reproducible true REAP ranking"; ORCHESTRATOR_HANDOFF §3 — "Its hash layers were pruned and its calibration is not fully reproducible… K160 is the runnable performance target, not the final quality-certified teacher construction"; experiment-ledger 2026-07-13 k160-provenance-audit — "Public K160 is valid for smoke/performance bring-up but not quality-certified: hash layers are pruned and published observations are not true REAP." Stage gates also specify layers 0-2 preserve all 256 experts, layers 3-42 carry the frozen 160-expert (uniform K) map, and required calibration-v1 provenance before official-source download.

**Closeout vs 0731 release:** The lane formally **closed 2026-07-21** (`2026-07-21-deepseek-v4-flash-frontier-closeout.md`; result packet status "paused/closed frontier"; model-effort-index "paused/closed on 2026-07-21"). The **DeepSeek V4 Flash 0731 release** (`0xSero/DeepSeek-V4-Flash-0731-REAP`, revision `ddc04540efda3d2a0788b129f1fad828ddc19b60`, 160-expert/48-shard) was **not** the lane's target — intake/qualification tooling for it began 2026-08-26 and prelaunch gates were ready 2026-08-28, explicitly replacing K160 only as the *active deployment candidate* (historical K160 record/model/evidence unchanged; old 256-expert DSpark pack never paired with the 160-expert 0731 target). **Despite being released ≥5 weeks after closeout, no 0731 GPU launch or speed result exists as of the last notes** (blocked by host memory policy; "No DeepSeek GPU launch occurred"). Reopen conditions for the K160 lane are only (a) 10-20M-token EAGLE re-capture+training for ~90 tok/s via the hybrid draft, or (b) a new device-kernel mechanism removing device execution time.

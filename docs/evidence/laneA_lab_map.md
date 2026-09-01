# Intel Arc B70 Optimization Lab — Lane Map (/tmp/b70lab)

Compiled 2026-08-31, read-only inventory of the repo at `/tmp/b70lab`
(repo `steveseguin/b70-optimization-lab`). Status vocabulary used by the lab:
**qualified/promoted** (B70-verified, gates passed, often LocalMaxxing-approved),
**closed/banked / paused** (record preserved, lane halted), **research-screen**
(diagnostic, not deployment-qualified), **closed-negative/rejected** (failed gate,
preserved), **quarantined** (contaminated cache/artifact).

## 1. Promoted/closed result lanes (results/) — one line each

| Model | Quantization | TP config | Headline tok/s | Status | Source README |
| --- | --- | --- | --- | --- | --- |
| Muse-Glimmer-30B (meta-models, SHA e63bf23b) | UD-Q8_K_XL + pretrained BF16 DFlash draft, group32 WOQ | 4×B70 TP4, 1 active req | 100.088 & 100.649 canonical full-256; 161.900 frozen first-100 median | Closed/banked no-training Q8/WOQ; LocalMaxxing approved | `results/muse-glimmer-30b-q8-woq-b70/README.md` |
| Poolside Laguna S 2.1 INT4 (rev 4bbfc285) | compressed-tensors INT4 g32 W4A16 target; matched INT4 DFlash draft; BF16 KV | 4×B70 TP4+EP4 | 124.069/122.829 (conventional 125.462) | Confirmed exact record; metric-qualified; LocalMaxxing approved | `results/laguna-s-2.1-int4-b70/README.md` |
| DeepSeek V4 Flash uniform-K160 (`0xSero` 180B rev 7c360e1c) | FP8 block-scaled dense weights, FP8 KV, MXFP4 experts (dense FP8 + MXFP4/FP4 experts) | 4×B70 TP4+EP | 80.820052 strict high; 78.287226 median-of-medians | Paused/closed frontier record; LocalMaxxing `cmrquta…` | `results/deepseek-v4-flash-k160-b70/README.md` |
| Qwen3.6 27B AutoRound INT4 (rev f5750c90) | INT4 W4A16 target, FP16 compute, runtime INT8/INT4 LM-head | 2×B70 TP2, c1, MTP3 | historical 95.384868 (94.431 conv); independent center 98.766 | Closed historical lane — current strict review FAILED (not robust >100) | `results/qwen36-27b-autoround-int4-b70/README.md` |
| Qwen3.6 27B GGUF Q8_0 on ASRock B70 | GGUF Q8_0 target, F16 KV, no draft | 2×ASRock-B70 TP2 | 35.699225 conventional (36.714 hist) | Closed/banked `B70-verified` target-only TP2 record | `results/qwen36-27b-q8-tp2-asrock-b70/README.md` |
| Qwen3.6 27B GGUF Q4_0 DFlash/SYCL | Q4_0 target, native DFlash5, fused Xe2 verifier | 1×B70 | 47.818818 | Closed intensive research lane; 100/200 TP1 objectives missed | `notes/2026-07-13-qwen27-dflash-sycl-closure.md` |
| Qwen3.6 27B MTP GGUF Q4 | UD-Q4_K_XL target + MTP3 draft (also MTP7 n_min=1) | 1×B70 | 30.678767 (legacy); 31.480049 MTP7 | Valid model/runtime reference, not competitive | `results/qwen36-27b-mtp-gguf-q4-b70/README.md` |
| Gemma 4 26B A4B Q8 / INT8 | UD-Q8_K_XL target/verifier + Q4_0 MTP draft, f16 KV | 1×B70 | 124.977140 median (p10 103.836) | Production-servable frontier/reference | `results/gemma4-26b-a4b-q8-b70/README.md` |
| Qwen3.6 35B A3B Quark | W8A8 INT8 | 4×B70 TP4 | 93.550542 strict | Closed reference packet | `results/qwen36-35b-quark-int8-b70/README.md` |
| MiniMax M2.7 INT4 AutoRound | AutoRound INT4 W4A16, FP16 KV | 4×B70 TP4, b1 | 89.314 strict / 83.172 deployable-32K / 94.406 structured | Deployable baseline + historical strict lanes | `results/minimax-m27-int4-autoround-b70/README.md` |
| Gemma 4 12B IT INT4 AutoRound | INT4 AutoRound | 4×? c8 32K endpoint | c8 32K production slot; c10–c64 documented | Production slot + research profiles | `experiments/gemma4-12b-int4-autoround-vllm/README.md` |
| Qwen3.8 27B FP8 (rev 017b9c7a) | official block-scaled FP8, FP16 activations/KV, MTP0 | 2×ASRock-B70 TP2 | 34.031596 (12/12 exact); 1,112.6 agg c128 | B70-verified candidate, strict single-user qualified | `experiments/qwen38-27b-b70/notes/2026-08-28-qwen38-fp8-deterministic….md` |
| Qwen3.8 Flash-Next FP8 | FP8 (MTP0 protected 5.515783) | 4×B70 TP4/MTP3 | ~20.7 (MTP0/MTP1 1K); 8.9–15.5 at 4K | **research screen; not deployment-qualified** | `results/qwen38-flash-next-fp8-b70/README.md` |
| Rapid model snapshots (Qwen3-30B-A3B, Phi-4-mini, GLM-4.7, etc.) | GGUF UD-Q4_K_XL / Q4_K_M, f16 KV | 1×B70 | e.g. 107.484 (Qwen3-30B), 96.548 (Phi-4-mini) | B70-verified rapid snapshots (first-pass) | `results/rapid-model-snapshots-b70/*/README.md` |
| Historical 2026-05 results (Qwen3.6/gguf Q4_0 fp8-vllm-xpu; q4_0-gguf AOT/DNN/driver-wedge; qwen36 fp8-mulmat-dmmv, q8-allreduce, q8cache-mtp, vdr-rootcopy) | gguf Q4_0 / FP8 | 1–4×B70 | diagnostics, no live headline | Closed historical lanes / wedges | `results/q4_0-gguf-2026-05-03-*.md`, `results/qwen36-b70-followup-2026-05-04-*.md` |

## 2. Standalone repro recipes (repro/) — one line each

| Lane | Model/quant | TP config | Headline tok/s | Status | Source |
| --- | --- | --- | --- | --- | --- |
| `deepseek-v4-flash-k160-b70-80tps-20260718` | K160 FP8+MXFP4+FP8 KV | 4×B70 TP4+EP, DSpark7 | 80.820052 / 78.287226 | Closed-lane record repro, fail-closed | `repro/deepseek-v4-flash-k160-b70-80tps-20260718/README.md` |
| `gemma4-26b-a4b-q8-b70` / `-125tps-20260701` / `-95tps-20260624` | Gemma 4 26B UD-Q8_K_XL + Q4_0 MTP | 1×B70 | 124.977 | Validated record + guards (one packet flagged unretained-draft-hash) | `repro/gemma4-26b-a4b-q8-b70*/README.md` |
| `laguna-s-2.1-int4-b70-102tps-20260726` / `-125tps-20260731` | Laguna S 2.1 INT4 g32 | 4×B70 TP4+EP4 | 102.971/101.942 (7-26); 125.462 conv (7-31) | Sealed repro of exact records | `repro/laguna-s-2.1-int4-b70*/README.md` |
| `lfm25-26b-q8-b70` | LFM2.5 2.6B Q8 | 1×B70 | 132.137/133.328 | Intake verified, baseline passed (neural.download guide) | `repro/lfm25-26b-q8-b70/README.md` |
| `minimax-m27-b70-110tps-ubuntu24-20260523` / `-89tps-20260520` / `-94tps-structured-20260522` | MiniMax M2.7 INT4 AutoRound | 4×B70 TP4 | 83.172 out (110.896 total) / 89.314 / 94.406 | Strict + deployable + structured repros | `repro/minimax-m27-*/README.md` |
| `muse-glimmer-30b-q8-woq-b70-100tps-20260813` | Muse UD-Q8_K_XL + DFlash BF16 | 4×B70 TP4 | 100.088/100.649 | Closed/banked repro | `repro/muse-glimmer-30b-q8-woq-b70-100tps-20260813/README.md` |
| `nemotron-35-lightning-30b-a3b-b70` | Nemotron 30B-A3B (hybrid MoE) UD-Q4_K_M | 1×B70 | 72.873 median | Intake verified, benchmarks pending (DRAFT) | `repro/nemotron-35-lightning-30b-a3b-b70/README.md` |
| `ornith-15-35b-a3b-q4km-b70` | Ornith 1.5 35B-A3B Q4_K_M | 1×B70 | 131.460 (130.45 conv) | Model verified, lab decode validated | `repro/ornith-15-35b-a3b-q4km-b70/README.md` |
| `ornith-15-9b-q8-b70` | Ornith 1.5 9B Q8_0 | 1×B70 | 50.109 median | Intake verified, baseline passed | `repro/ornith-15-9b-q8-b70/README.md` |
| `qwen36-27b-autoround-int4-b70` (+ `-determinism-20260818`) | Qwen3.6 27B INT4 + MTP3 | 2×B70 TP2 | 95.385 historical; determinism screen 92.0/96.8 | Historical + determinism-bounded screen | `repro/qwen36-27b-autoround-int4-b70*/README.md` |
| `qwen36-27b-q8-tp2-asrock-b70` | Qwen3.6 27B Q8_0, F16 KV, target-only | 2×ASRock-B70 TP2 | 36.604 conventional (36.719/36.773) | Promoted target-only TP2 | `repro/qwen36-27b-q8-tp2-asrock-b70/README.md` |
| `qwen38-27b-256k-vision-mtp-b70` | Qwen3.8 27B (256K vision) MTP | 1×B70 | 27.004 MTP-assisted | FIT-OFF DECIDED — UD-Q5_K_S ships as flagship quant | `repro/qwen38-27b-256k-vision-mtp-b70/README.md` |
| `qwen38-27b-autoround-int4-b70` | Qwen3.8 27B AutoRound INT4, MTP5 | 2×B70 TP2 | 101.170 honest margin-free anchor | **No promoted record** (MTP5/MTP4 rows output-changing, withdrawn) | `repro/qwen38-27b-autoround-int4-b70/README.md` |
| `qwen38-27b-fp8-vllm-tp2-asrock-b70` | Qwen3.8 27B FP8 | 2×ASRock-B70 TP2 | 34.031596 MTP0 (MTP1 qualified default) | Promoted strict MTP1/qualified MTP0 | `repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/README.md` |
| `qwen38-27b-q4km-mtp2-tp1-b70` | Qwen3.8 27B Q4_K_M + Q4_0 MTP2 | 1×B70 | 42.636988 | Lab-validated single-card | `repro/qwen38-27b-q4km-mtp2-tp1-b70/README.md` |
| `qwen38-27b-q4km-q4mtp-mtp2-tp2-b70` | Qwen3.8 27B Q4_K_M + Q4_0 MTP2 | 2×B70 TP2 | 64.237301 (+29% vs MTP0 oracle) | Lab-validated candidate package | `repro/qwen38-27b-q4km-q4mtp-mtp2-tp2-b70/README.md` |
| `qwen38-27b-q4km-tp1-b70` | Qwen3.8 27B Q4_K_M, no draft | 1×B70 | 27.825726 | Promoted target-only single-card | `repro/qwen38-27b-q4km-tp1-b70/README.md` |
| `qwen38-27b-q4km-tp2-asrock-b70` | Qwen3.8 27B Q4_K_M target-only | 2×ASRock-B70 TP2 | 49.717503 (LocalMaxxing approved `cmsy…`) | Promoted target-only TP2 | `repro/qwen38-27b-q4km-tp2-asrock-b70/README.md` |
| `qwen38-27b-q8-q4mtp-mtp2-tp1-b70` | Qwen3.8 27B Q8_0 + Q4_0 MTP2 | 1×B70 | 37.062028 | Strict fixed-suite headline | `repro/qwen38-27b-q8-q4mtp-mtp2-tp1-b70/README.md` |
| `qwen38-27b-q8-tp1-b70` | Qwen3.8 27B Q8_0, F16 KV, no draft | 1×B70 | 19.66 tg128 depth0 / 18.02 @32K | llama-bench reference (not HTTP suite) | `repro/qwen38-27b-q8-tp1-b70/README.md` |
| `qwen38-27b-q8-tp2-asrock-b70` / `-c2-` | Qwen3.8 27B Q8_0 target-only | 2×ASRock-B70 TP2 (c1 / c2) | 36.726447 strict package (c1); c2 separate | Promoted (c2: concurrency-2 screen) | `repro/qwen38-27b-q8-tp2-asrock-b70/README.md` |
| `qwen38-flash-next-fp8-tp4-mtp3-b70` | Qwen3.8 Flash-Next FP8, MTP0 protected 5.515783 | 4×B70 TP4+EP4 MTP3 | research only | **research-status / runtime-hosted; not a runnable guide** | `repro/qwen38-flash-next-fp8-tp4-mtp3-b70/README.md` |
| `rapid-model-snapshots-b70` | N/A — shared fixed realistic suite only | N/A | — | infrastructure (suite identity) | `repro/rapid-model-snapshots-b70/realistic-suite-v1.json` |

## 3. Lane status per CURRENT.md (2980 lines)

Grep of section headers gives these sections; current (2026-08-31) active work is
almost entirely **Qwen3.8 27B** (AutoRound INT4 TP2, Q8/Q4KM target-only TP1/TP2,
Flash-Next TP4) plus the Ornith / LFM2.5 / Nemotron neural.download intake queue.

- **OPEN / ACTIVE:** "Active Optimization Lane" (§929) and "Active Research: Qwen3.8 27B TP2" (§2157), "Qwen3.8 27B Q4_K_M target-only TP1" (§2442), "Qwen3.8 27B INT4 AutoRound, vLLM/XPU TP2 speculative" (§2512); "Active Qwen3.8 AutoRound INT4 Two-B70 Optimization" (§31), "Active Upstream-Current XPU Integration Policy" (§121), "Pinned Certified Qwen3.8 TP-Scale Frontier" (§655), "Active Product Track: neural.download packets" (§774), "Active Qwen3.8 One-Card Package Work" (§54).
- **CLOSED:** "Closed: Qwen3.6 27B INT4 AutoRound, vLLM/XPU TP2 speculative" (§2640) — includes Qwen36 determinism review, GDN prefill screen, graph-replay-bypass R1/R2, draft-margin, clock-lock, Q64xK32 FA operator, mtp.fc INT4, etc. all terminally closed.
- **PAUSED / BOOKMARKED (closed lanes kept reproducible):** "Paused And Bookmarked Lanes" (§2699) lists exactly: Qwen family map, **Muse-Glimmer-30B Q8/WOQ**, **Laguna S 2.1 INT4**, **DeepSeek V4 Flash K160** (→ `results/deepseek-v4-flash-k160-b70/README.md`), **Gemma 4 26B A4B Q8**, **MiniMax M2.7 INT4**, all model efforts, scoreboard.
- **RESEARCH-SCREEN / NOT QUALIFIED:** Qwen3.8 Flash-Next FP8 (4×B70) — explicitly `research screen; not deployment-qualified`.
- DeepSeek 0731 note (CURRENT.md §2131–2141): the `DeepSeek-V4-Flash-0731-REAP` GPU qualification was staged on 2026-08-26/28 but **no DeepSeek GPU launch occurred**; blocked by the host memory policy.
- **QUARANTINED:** contaminated Triton compile caches were quarantined (e.g. `…/quarantine-sycl9-20260714T1011Z`, deepseek graph-recovery note); "Quarantined contaminated cache entries instead of deleting them."

## 4. Reusable kernel work locations

- **`patches/`** — per-lane patch packets with actual source:
  - `patches/deepseek-v4-flash-reap-xpu-b70/`: full record git-bundles + reviewable `.patch` files — `vllm-…-80tps-record-20260718.patch` (5,333 lines: `vllm/models/deepseek_v4/attention.py`, `…/xpu/{model,mtp,dspark,xpu_sparse,xpu_sparse_decode_fp8,xpu_qnorm_rope_kv_fp8_insert,divergence_capture}.py`, `…/spec_decode/dspark/speculator.py`), `vllm-xpu-kernels-…-80tps-record-20260718.patch` (8,954 lines: SYCL under `csrc/xpu/{sycl,mhc,grouped_gemm,sampler}`), `oneccl-…-wideepoch…patch` (160 lines) + `.bundle` archives; also 10 `0001-xpu-add-guarded-K160-EAGLE-training-capture*.patch` variants.
  - `patches/laguna-s-2.1-xpu-b70/` — biggest: `vllm-xpu-kernels-laguna-…-20260726.patch` includes the SYCL `fp8_paged_mqa_logits_kernel_t` / `fp8_paged_mqa_logits_xe2` attention sources, `vllm-laguna-width12-dflash-fp8-…` (78k lines) etc.
  - `patches/` others: muse-glimmer, gemma4-26b, qwen27-dflash-sycl, qwen36-27b-* (autoround, mtp, q8), qwen36-35b-quark, qwen38-27b-* (q4km, q8, mtp-fc), qwen38-flash-next-fp8, vllm-codex-wip.
- **`experiments/*/` with ACTUAL kernel/source** (not notes-only):
  - `minimax_ar_fused_rms_xpu/` — real C++ SYCL op (`minimax_ar_fused_rms_xpu.cpp`, 14 `sycl::` refs) AR-fused-RMS.
  - `minimax_pair_argmax_xpu/` — SYCL `minimax_pair_argmax_xpu.cpp`.
  - `minimax_qk_rms_xpu/`, `minimax_qk_rms_xpu_ipc/` — SYCL `.cpp` (+ IPC test).
  - `qwen27_fused_postattn_rms_w4a16/` — real SYCL kernel `csrc/fused_postattn_rms_w4a16.sycl`.
  - `qwen27_graphsafe_flash_attention/` — vLLM-nesting patch set (chunk-prefill/completion-barrier, local-accessor, force-chunk-decode) + replay tests + `qwen38-head256/chunk/paged` configs — patch-based not standalone kernel.
  - `xpu_level_zero_peer_probe/` — bare `peer_probe.cpp` (raw Level Zero, not SYCL).
  - `deepseek-v4-flash-reap-xpu-b70/` — **scripts/ has ~120 Python harness/bench microkernels (real Triton `@triton.jit` kernels in e.g. `bench-bf16-gemv-triton.py`, `bench-wqb-fp8-bf16-gemv.py`) but no compiled kernel source in-tree** (the SYCL/Triton kernels live in the pinned vLLM + vllm-xpu-kernels bundles/patches); its `patches/` dir has `0001-deepseek-v4-exact-xpu-moe-shapes.patch`, `0002-deepseek-v4-unambiguous-shape-selectors.patch`, `2026-07-19-native-k2-single-submission.patch`, `deepseek-v4-persistent-kstep-….patch`.
  - `option4-decoder/` (repo root) — raw Level-Zero command-list decoder substrate.
  - Most other `experiments/*/` are **notes-only** (md): deepseek-v4-flash-autoround-vllm, gemma4-12b/26b, laguna-*, minimax-m27-*, minimax_xpu_kv_offload (18 md, 0 code), muse-glimmer-30b, ornith-15, qwen27-dflash-sycl, qwen36-27b-*, qwen36-35b, qwen38-27b, qwen38-flash-next-fp8, rapid-model-snapshots.
- `prototypes/`, `packages/` (qwen38-27b-fp8-tp2-b70 etc.), `scripts/` (shared bench suite), `benchmarks/` hold further reusable harnesses.

## 5. Who did the kernel work

`git log --format='%an'` on the kernel-bearing dirs:

- **experiments/ (incl. all SYCL ops above):** Codex Agent 3,308 commits; Claude 30; steveseguin 12; Steve Seguin 8. The SYCL custom-op dirs (`minimax_ar_fused_rms_xpu`, `minimax_pair_argmax_xpu`, `minimax_qk_rms_xpu[_ipc]`, `qwen27_fused_postattn_rms_w4a16`, `xpu_level_zero_peer_probe`) were authored by **Codex Agent + Steve Seguin (steveseguin)** jointly.
- **patches/:** Codex Agent 401; Steve Seguin 139; steveseguin 95.
- **results/:** Codex Agent 535; Steve Seguin 20; steveseguin 5 (+1 dominick253, 1 Dominick).
- **repro/:** Codex Agent 315; Steve Seguin 2; (dominick253 1).
- **community/:** Codex Agent 42; **dominick253 17**; **bosd 8**; Steve Seguin 3.
- Overall repo: Codex Agent is the dominant committer; Steve Seguin is the lab principal; a handful of Claude-agent commits.

**External contributors (community/) and what they contributed:**
- **dominick253** (most active external): Qwen3.6 27B FP8 TP2 Docker recipe on 2×B70 (PR #9, validated here 2026-07-25 → 30.171 tok/s measured, `B70-tested`); Qwen3.6 27B MTP Q4_K_M llama.cpp/SYCL two-process recipe (38.112 tok/s reported; lab validated narrow one-GPU recipe, 150K/175K identity mismatch — 2026-08-08 validation); Qwen3.6 35B FP8 TP2 (vLLM) & 35B llama.cpp SYCL & INT4 b2 1-GPU recipes; Qwen3.8 27B "Cold Fusion" GAIN V1.1 MTP llama.cpp SYCL (38.4 tok/s on b10472/kernel 7.0.0-29; refresh on b10488-7 fell to 22.73 tok/s with MTP2 acceptance drop — recorded as a staged engine/kernel-switch regression study).
- **sergiiob**: Qwen3.8 27B GPTQ INT4 + native MTP on one B70 (vLLM XPU; reports 32.9 target-only, 83.7 MTP4 at p512/g128) — target/MTP reproduced locally, provisional.
- **mndodd**: Qwen3.6 27B llama.cpp/SYCL fork (`intel-sycl-optimization`) — the maintained TP2 Q8 source fork; validated at 31.025377–31.338765 tok/s (2×B70) and 17.776–17.956 (1×B70); the mndodd *source* (not lab-measured >50 claim) is the basis of the Q8 TP2 record; lab added fused state-I/O/tail/SIMD16-QK/recurrent conv-SiLU-L2 patches on top.
- **bosd**: nine TRX50 (2×B70+B60) platform field reports (PR #16/#17) — community-reported, docs only.
- **boyter**: Qwen3.8 on Arc Pro B65 field report (community-reported).
- **kydo (@0xkydo)**: mlx.fast Qwen3.8-27B Apple-Silicon challenge field report (no B70 reproduction possible; transferable speculation-scoring ideas).
- **0xSero**: Qwen3.8 27B Q4_K_M Dockerized llama.cpp recipe (TP1/TP2, MTP) — contributor of the DeepSeek-V4-Flash-180B K160/0731 checkpoints targeted by the lab; external repo captured, no vendored source.
- **anon (Reddit)**: Arc Pro B70 vLLM XPU "52 tok/s" field report (raw logs not captured).

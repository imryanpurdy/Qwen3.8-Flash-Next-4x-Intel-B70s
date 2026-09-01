# Deliverable 1 — Intel Arc Pro B70 Inference Cookbook assessment (/tmp/b70cookbook)

Author: **SergiioB** (git log: every commit by SergiioB, latest 2026-08-31 13:48 +0200, active through the present day). 229 files. README and AGENTS.md both carry "Last updated 2026-08-09/08-31" and reference sergiiob.dev blog posts. MIT / CC BY 4.0.

## What the repo covers

One cookbook, one page per model family, all for **Intel Arc Pro B60/B70 (Battlemage / Xe2, 32 GB)**. Two engines: **vLLM XPU** (Docker, GPTQ-INT4 + native MTP speculative decoding) and **llama.cpp SYCL** (native, GGUF). Model families with published recipes:

| Family | Engine | Headline | Doc |
|---|---|---|---|
| Qwen3.6-35B-A3B (MoE) | vLLM XPU | MTP4 p512/g128 **170.91** tok/s, 128K exact | docs/qwen36-35a3/QWEN36-MOE-VLLM-XPU.md |
| Qwen3.6-27B (dense) | vLLM XPU | GPTQ-INT4 + MTP4 69.3 tok/s, fp8 KV | docs/qwen36-27/QWEN36-DENSE-VLLM-XPU.md |
| Qwen3.8-27B | vLLM XPU | 4 routes: GPTQ-INT4 single-B70 (**C1 106.7**), FP8 W8A16 dual-B70 TP2 (60.13 t/s), Windows, Pi-agent | docs/qwen38-27/ |
| Nemotron-3.5-Lightning-30B-A3B | vLLM XPU | DFlash n=7, **186.61** tok/s, native MTP 0% | docs/nemotron35-30a3/NEMOTRON-DFLASH-B70.md |
| Muse-Glimmer-30B | llama.cpp SYCL | vision + DFlash n2, **26.8** tok/s @128K | docs/muse-glimmer/MUSE-GLIMMER-B70.md |
| Ornith-1.5-35B-A3B | vLLM XPU | local GPTQ MixedCal-v2, MTP1+DraftINT4, **108.4** tok/s | docs/ornith15-35a3/ORNITH-VLLM-XPU.md |

Plus cross-cutting content: benchmark methodology, LocalMaxxing leaderboard submissions, an Xe2 wedge watchdog, a reliability report, a dual-B70 TP2/PP2 guide, Windows 11 kits, and a benchmark catalog.

### docs/ (each file, 1–2 lines)
- **Root docs**: `BENCHMARK-CATALOG.md` (machine-generated public numeric catalog from data/benchmarks.v1.json); `BENCHMARK-FORMAT.md` (the stable cross-model benchmark contract — campaign identity, exact metrics, VRAM/cache/timing gates); `CAMPAIGN-LOG.md` (19-run narrative from "vLLM 7× slower than llama.cpp" to "MTP beats llama.cpp"); `CONNECTING-CLIENTS.md` (Pi/omp/Hermes client config, ports, tool-calling smoke test); `DUAL-B70-TP2.md` (TP2/PP2 topology authority: ZE_AFFINITY_MASK injection + oneCCL simple-threshold envs, 60s isolation repro); `FULL-SETUP-COMMANDS.md` (everything to reproduce the 2026-08-09 C1 matrix); `IMAGE-AND-PATCH-MATRIX.md` (image digests per family, ordered apply lists, denylists); `POWER-SWEET-SPOTS.md` (150 W MoE / 165 W dense efficiency / 180 W sustained / 230 W burst data); `REAL-WORLD-PI-BENCHMARKS.md` (real Pi workloads, cache/spec matrix, exact 128K results, resident-session tests); `RELIABILITY-REPORT.md` (evidence-linked "what works / what breaks / which layer owns it" map, as of 2026-08-31 — **walls of detail below**); `localmaxxing-submission-schema.md` (flat JSON schema for the leaderboard).
- **Family docs**: `qwen36-35a3/` (MoE recipe + QUANTIZATION-QUALITY.md KLD audit); `qwen36-27/` (dense recipe + DENSE-FP8-GAP.md); `qwen38-27/` (README hub + QWEN38-VLLM-XPU.md GPTQ recipe + FP8-TP2-W8A16.md + DRAFT-INT4-S-M1.md + PI-AGENT-BACKEND.md + WINDOWS-STANDALONE.md); `nemotron35-30a3/` (DFlash recipe, no-spec graph path, CLAIMS.md); `ornith15-35a3/` (MixedCal-v2 conversion, AutoRound-vs-GPTQ parity study, CLAIMS.md); `muse-glimmer/` (llama.cpp vision recipe).

### research/quantization-format-strategy.md
Single research doc. Thesis: **GPTQ-Int4 is the optimal format for vLLM XPU on MoE** — XMX engines are integer-first (INT4/INT8/FP16 native fast path; FP is the NVIDIA priority). Measured table: GPTQ-Int4 133 t/s / 8,718 t/s prefill, GGUF Q4_K_XL 69 t/s, Q5_K_M 70 t/s (best quality, ~2× lower KLD), MXFP4 10.4 t/s (GDN Triton bottleneck, not quant), FP8 block 0.75 t/s (dequant fallback), NVFP4 N/A (proprietary), **native FP8 blocked** (no XPU block-FP8 scaled MM kernel; `KeyError: PlatformEnum.XPU`). Explains why 4-bit INT beats FP4 at same bitwidth (4 mantissa bits vs 2 exp + 1 mantissa). Names the *highest-impact gap*: a native XPU block-FP8 kernel (would unblock dense 27B).

### watchdog/ — what wedges it watches for
Detects and auto-recovers the **Xe2 Level-Zero wedge** — the single most important failure mode in this ecosystem.
- **Trigger**: the `xe` kernel driver resets a compute/copy engine under sustained Level-Zero load returning `Fault response: Unsuccessful -ENOENT/-EINVAL`; userspace L0 context wedges permanently, engine dies (`TimeoutError: RPC call to sample_tokens timed out` → `EngineDeadError` in vLLM). Only a container restart recovers.
- **Two signals, both required by default**: (1) health-poll failures (3 consecutive `curl` misses); (2) kernel-log signature `Engine reset: engine_class=ccs|bcs`, `Fault response: Unsuccessful`, `guc_exec_queue_timedout_job`, `TLB invalidation fence timeout`, `Completion-Wait loop timed out`. Health failure without a kernel sig = `DEGRADED` (app hang, GPU restart won't fix).
- **Corroborated links**: intel/compute-runtime#948 (dual B70 + vLLM TP2, reproduces every 2–6 h on every stack, kernel 7.0.0-27 / GuC 70.58.0); vllm-project/vllm#41663; darktable/darktable#20257 (same ccs signature via OpenCL → driver/firmware-level bug).
- Files: `install-watchdog.sh` (one-command systemd installer), `xpu-wedge-watchdog.sh`, `xpu-wedge-watchdog.service`, `README.md`. Config via env (HEALTH_URL, CONTAINER, TRIGGER_MODE, FAIL_STREAK, webhook). Docs caveat: in-flight requests are lost on restart — front with a retrying proxy.

### patches/ — list & summary (family-tagged; see IMAGE-AND-PATCH-MATRIX.md)
- **Qwen MTP (current pinned nightly `2c427ef`)**:
  - `patch_mtp_nightly.py` — gate `B70_MTP_BF16_DRAFT=1`: null quant_config so draft MTP layers build unquantized BF16 (matches the preserved BF16 `mtp.*` weights).
  - `patch_mtp_boundary.py` — reclassify a partial final MTP4 speculative group as stateful non-spec prefill so exact-128K completes without padding (unpatched MTP4 failed after 124 of 128 outputs).
- **Qwen3.8 champion image `f01e24f6`**:
  - `patch_gdn_mixed_split_v5.py` (a.k.a. `patch_gdn_split_mixed.py`) — root-causes the XPU GDN fused `causal_conv1d → gdn_attention` host `TORCH_CHECK` that refuses mixed spec+non-spec batches; splits into per-group fused calls with `index_select`/`index_copy_` compaction (v1–v4 failed the C++ size-sum/narrow contract; v5 uses `None` for the idle side).
  - `patch_draft_lmhead_int4.py` + `patch_draft_mtp_int4.py` — RTN-INT4 g128 the draft LM head (2.54 GB fp16 → 0.66 GB) and MTP linears (`~0.85 → ~0.21 GB/step`) via existing `torch.ops._xpu_C.int4_gemm_w4a16`; env-gated; **explicitly NOT safe at TP>1** (verification does not protect emitted output — issue #9).
  - `patch_fp8_w8a16.py` — reroute block-FP8 linear to `fp8_gemm_w8a16` with `apply_input_quant=False` (BF16 activations direct; −25% step time, −20–40 W; +34% speedup, 48 tok/s at p1024).
  - `patches/vllm-xpu-kernels/apply_smallm_patch.py` (P1–P6: Xe2 small-M BF16×block-FP8 GEMM kernel, b70-architect authored 2026-08-29), `fix_smallm_placement.py` (fixes an insertion cursor bug), `patch_fp8_stack.py` + `patch_fp8_hybrid.py` (W8A16/W8A8 hybrid numerics for the MTP head; collapsed acceptance fix), `0001-zero-xe2-grouped-gemm-atomic.py` + `0002-muse-paged-decode-tuple.py` (vllm-xpu-kernels#524, source-only, need rebuild).
- **Dual-B70**: `patch_vllm_worker_affinity.py` — per-worker `ZE_AFFINITY_MASK` injection at spawn (fixes IPC handle + ~1 GiB host-RAM-per-GiB-VRAM duplication); patch alone still crashes on some hosts (issue #8) → needs the 4 `CCL_SYCL_*_SIMPLE_THRESHOLD` env vars (Intel llm-scaler#594 workaround).
- **Nemotron**: `patch_xpu_grouped_topk_native_v2.py` — native XPU grouped-topk + `torch.compiler.disable` graph break (vllm#52159); `ssu-b70-b8w4/*.json` (Triton 3.6.0 SSU B8/W4 tuning).
- **Historical (vLLM 0.21, retired)**: `patch_xpu_int4_moe_v4.py` (int4 MoE load fix), `patch_mtp_bf16_draft.py`, `patch_mtp_ptr_wrap.py` (int64 data_ptr wrap; don't wrap uint64 dst_ptrs).
- **pi**: `patches/pi/qwen38-vllm-thinking.ts` (pi extension mapping reasoning_effort → chat_template_kwargs).

### results/ — machine-readable evidence
`prefill-decode-matrix-20260809-summary.json` (MoE, C1 n=5, 165 W): no-spec prefill p512 5,156 → p8192 7,576 tok/s; MTP4 decode p512/g128 **170.91**; full p131071 cold input ~2,678–3,144 tok/s. `qwen36-27/…-dense27-summary.json` (230 W, dense 27B MTP4). `engine-comparison-full-20260806.md` (vLLM vs llama.cpp showdown), `prefill-decode-matrix-20260809-tables.md`, plus llama.cpp power grids (150 W/230 W), 128K cache-spec matrix, realworld-pi summary, multiturn/ctx-scaling JSON, and the qwen38-27-fp8-tp2-k8-n5 closeout (TP2 W8A16+small-M, MTP8, n=5, with power logs + LocalMaxxing approved receipt).

### submissions/ — historical LocalMaxxing payloads
`b70-hardware.json` (hardware profile), and approved-engine cells: `vllm-mtp-gptq-int4` (MoE MTP4), `vllm-dense27-mtp4-gptq-int4`, `vllm-qwen38-mtp4-gptq-int4`, `vllm-qwen38-mtp4-draft-int4`, `llamacpp-dense-q4km-27b`, `llamacpp-q4kxl-moe`, `llamacpp-muse-glimmer-30b`. Explicitly "historical… must not be copied as the current nightly recipe."

## Sparse attention / MoE / large-model memory layout on Intel Xe — what is documented
- **Sparse attention**: **NOT documented.** No content on sparse attention, MLKV, or attention sparsity. The only "sparse" match is MoE expert sparsity (8/256 experts active) in `docs/qwen36-35a3/QUANTIZATION-QUALITY.md` ("This sparse activation… 96.9% of experts are idle per token").
- **MoE**: **Extensively documented** — XpuFusedMoe kernel path, grouped-topk determinism patch, natively fused MoE + INT4 (quantization-format-strategy.md; QUANTIZATION-QUALITY.md; DRAFT-INT4; patch_xpu_int4_moe_v4.py).
- **Large-model memory layout**: **Only obiter dicta, no systematic treatment.** No weight-placement/offload strategy docs (no host-side weight residency, no selective offload, no memory-layout tuning pages). What exists: (a) AGENTS.md "Never load a single GGUF >30 GB with -ngl 99 — it overflows 32 GB VRAM and causes a hard system crash (verified)"; (b) AUTOROUND-VS-GPTQ.md offload disk-space margin (1.2×, 72 GiB model → ~87 GiB) + memory-cgroup guard during conversion; (c) WINDOWS-STANDALONE.md display-attached VRAM pin (0.75 util + 4.30 GiB fp8 KV) because one B70 drives the desktop; (d) DUAL-B70 host-RAM duplication (~1 GiB host RAM per GiB VRAM). Nothing like the lab's selective UVA PLE host-side placement (see Deliverable 2).

## Overlap vs complement with /tmp/b70lab
**The cookbook (SergiioB) is a single-operator production recipe + evidence repo; the lab is a multi-entity, hardware-focused observatory.** The same person appears in both (community/sergiiob-qwen38-27b-vllm-xpu/).

- **Cookbook has that lab lacks**: complete turnkey single-B70 serving recipes (launchers, patch harness, watchdog, Windows kits, Pi-agent integration), exact rendered-token 128K methodology, LocalMaxxing submission pipeline/receipts, the FP8-TP2 W8A16 small-M kernel source patches (applied to vllm-xpu-kernels builds) with closeout evidence, DFlash/Ornith/Muse family recipes, and the Xe2 reliability map with ranked fix list.
- **Lab has that cookbook lacks**: **Flash-Next FP8 4×B70 work (cookbook only goes to 2×B70 TP2)**; the selective UVA / host-side PLE placement for 27B-class memory layout; multi-node/multi-owner field reports and cross-lab community lanes; repro packages with runtime/dependency contract JSON; comprehensive device-level failure catalogue (docs/intel-b70-minimax-feedback-20260523.md); TP2 ASRock rig repros for Qwen3.8-27B.
- Shared dense-27B territory: cookbook = single-B70 GPTQ-INT4 & dual-B70 FP8 TP2 (its own numbers, "official-lab self-report, E2, not independently reproduced"); lab = same model on ASRock dual-B70 rigs with its own repros — good cross-check material, must not be conflated.

## Toolchain facts
- **Pinned images (3 generations, never mixed)**:
  - Qwen3.6 Pi / dense 27B: `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97` → vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`, `vllm-xpu-kernels 0.1.12`. PyPI `0.1.12.2` untested.
  - Qwen3.8 GPTQ (single B70) / Ornith: `…f01e24f6c7ff…` (same vLLM nightly gen, kernels 0.1.12.3 for the FP8 TP2 build).
  - Nemotron DFlash: `…1da0a9548545…` → vLLM `0.26.1rc1.dev668+g3ee2df303`, kernels 0.1.12.3.
  - llama.cpp SYCL: reference build `b10255+` (commit `071327508`), built with **oneAPI 2026.0**, icx/icpx, `GGML_SYCL=ON GGML_XMX=ON`.
  - Torch/Triton: not pinned explicitly (comes from image); the SSU JSON pins Triton 3.6.0 for the Nemotron fast-path tuning.
- **Launch scripts (per model)**: `benchmarks/`: root shared matrix runner `b70-pi-prefill-decode-matrix.sh`; `qwen36-35a3/launch-mtp4-128k-nightly.sh`, `launch-vllm-128k-mode.sh`, `launch-mtp-128k.sh` + `launch-mtp-bf16draft.sh` (both RETIRED — require unpublished 0.21 image), `b70-pi-128k-cache-spec-matched.sh`; `qwen36-27/launch-dense27-128k-mode.sh`; `nemotron35-30a3/launch-nemotron-dflash.sh` + `launch-nemotron-graph.sh`; `ornith15-35a3/launch-ornith-mtp1.sh`. All: refuse a competing inference process, leave host power unchanged, `--restart unless-stopped`.
- **Benchmark methodology**: documented as non-negotiable discipline in AGENTS.md §6/§12 + BENCHMARK-FORMAT.md: one engine at a time; discard JIT warmup + one same-shape warmup; n=5 C1 samples, median+range; exact rendered prompts with unique entropy first and zero prefix-cache-hit delta; exact metric definitions (client post-first rate = (completion−1)/(end−first token), TTFT/TTFC/TPOT/E2E from client monotonic SSE); separate C1 and Cn; real power via energy1_input diff + temp cooldown ≤52 °C; fail-closed compiler (`b70-compile-prefill-decode-matrix.py`). Matrix cells: prefill p512–p8192 (+p131071 full), decode p512/p8192 at g32/128/256/512, p9445/g128 control. Correctness limitation stated: prompt hashes match across modes; **output parity does not — speed is not proof of token/logit/KL parity**.
- `b70-multiturn-128k-test.py`: loads a ~120K-token "document" on turn 1 then measures per-turn TTFT + decode on warm KV (proves cached-context claim).
- `b70-ctx-scaling-sweep.py`: progressives 4K→128K prompts; measures prefill throughput and whether decode degrades as KV fills (attention O(n²) probe).
- `b70-pi-prefill-decode-matrix.sh`: starts no-spec/MTP1/2/4 servers with cache on, generates entropy-first exact prompts, records five C1 samples per cell, aborts on first invalid cell, ≥500 MiB free after load, cooldown ≤55 °C.

## Bottom line
A mature, chronologically-active (2026-08-31) single-author production cookbook for single- and dual-B70 vLLM XPU + llama.cpp SYCL, dense on benchmark rigor and patch engineering. It does **not** cover sparse attention or structured large-model memory layout, and tops out at **two** B70s — Flash-Next on 4× B70 (the lab, Deliverable 2) is outside its scope.

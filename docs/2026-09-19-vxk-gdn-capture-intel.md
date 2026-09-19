# vllm-xpu-kernels Intel — GDN shapes, spec-decode/capture faults, kernel-dev skills

Date: 2026-09-19 · Web research only (no rig access) · Repo: https://github.com/vllm-project/vllm-xpu-kernels
Sources: GitHub REST API + raw file fetches at specific tags/commits (main @ 2026-09-19).

## 0. Version mapping (important)

- Our pinned build: `v0.1.8.3.dev0+g3cab97a.d20260709` → **source = commit 3cab97ad** (2026-05-21, `[XPU][FMHA] Optimize fp8 KV cache paged-decode on Xe2 (#357)`), built 2026-07-09. Tag `v0.1.8.2` points at 3cab97ad; `v0.1.8.2` released 2026-05-26.
- Consequence: our build **predates the native GDN speculative-decode path** (see §2.3) and all GDN spec-correctness fixes landed Aug–Sep 2026.
- Newest releases: **v0.1.14 / v0.1.14.1** (2026-08-31) — v0.1.13/.1 (08-13), v0.1.12/.1 (07-28/30), v0.1.11 (.1) (07-09/17). Relevant to our pin: v0.1.11 shipped 2026-07-09 (same date as our build stamp; includes #439/#455-era work, not the GDN spec fixes).

## 1. GDN kernel code paths & compiled-shape policy

### 1.1 Where the GDN code lives (current main)
- `csrc/xpu/gdn_attn/gdn_attn_interface.cpp` — host op `gdn_attention` (+ split conv/delta stages after PR #402, 2026-07-22): dispatches to XE2 chunk kernels when `num_prefills > 0` on Xe2, else native `causal_conv1d`/`gated_delta_rule`.
- `csrc/xpu/gdn_attn/xe_2/`: `chunk_gated_delta_rule_kernels_xe2.hpp` (chunk GDR — CUTE tiled MMA), `chunk_causal_conv1d_xe2.hpp`, `chunk_causal_conv1d_tiled_xe2.hpp`, `gemm.hpp`, `l2norm*`, `chunk_gated_delta_rule_xe2.{cpp,h}` (42-line thin launcher).
- `csrc/xpu/gdn_attn/gdn_attn_utils.h` (main): `chunk_size_xe2 = 64` (compile-time chunk size; the kernel pads sequences to chunk boundaries: `padding_size = batch_size * (chunk_size-1)` = **63 rows per batch entry**, gdn_attn_interface.cpp:438).
- `conv1d_tile_size = 8`; tiled-conv1d WG params: wg_size 64, 4 sub-groups of 16, elems_per_item 4 (`chunk_causal_conv1d_tiled_xe2.hpp`).
- GEM tile policies (compile-time, in `gemm.hpp`/`chunk_gated_delta_rule_kernels_xe2.hpp`): compute_A = 64×64×32, inverse = 16×16×16, compute_wu = 64×64×32 (2×1), fwd_o = 64×64×32 (4×2) — M=chunk(64), N=64, K=32 tiles; head_k_dim/head_v_dim are **runtime** (looped in K/N tiles), dtype templated.

### 1.2 Critical: GDN kernels have NO shape-policy / conf mechanism
- GDN (and MQA-logits) are **not** covered by `kernel_configs/*.conf` — those only govern the full-attention FA2 kernels (`fmha` chunk-prefill + paged decode). GDN head dims are runtime; no per-headsize compile-time instantiation, no "not compiled for this config" error for GDN. So there is **no per-head-size or per-block-size shape coverage to add for GDN layers** — the shape issue space is: chunk_size fixed 64, tile policy fixed, register pressure driven by head dims at runtime.
- Shape-dependent branches that DO matter for our head dims (gdn_attn_interface.cpp:460-483, main):
  - `use_tiled = non_spec_token >= 8`; `fuse_l2norm = use_tiled ? (2*head_k_dim <= 256) : true` (tiled_feats_per_wg=256). head_k_dim 128 → fused; head_k_dim 256 → NOT fused (standalone l2norm kernel launched instead).
- PR #391 (open) mentions it "guards the XE2 chunk GDN prefill path so it is used only for head dimensions supported by the chunk kernels" — i.e., as of main there is still no committed guard; a shape-guard exists only inside that unmerged PR. **UNVERIFIED** which head dims the chunk GDN kernels support on Xe2; nothing in-tree rejects head_k_dim/head_v_dim outright (tests only exercise head_k_dim=32 defaults — `tests/gdn_attn/test_gdn_attn.py` NUM_K_DIMS/[head_k_dim]=[32]).

### 1.3 Full-attention shape policy: precompiled vs JIT, and what our build ships
- **No JIT anywhere** (no `jit`/DPC++ runtime-compile machinery in tree). All kernels are AOT-compiled into the extension at build time; CMake *generates* one .cpp per (policy × bool-combination) tuple from `*.cpp.in` templates:
  - Chunk prefill: `csrc/xpu/attn/xe_2/chunk_prefill_configure.cmake` + `chunk_prefill_kernel_template.cpp.in`. 12 policy names = 6 head sizes (64/96/128/192/256/512) × 2 tile policies (std `TileShapeQK=32`, `_b16` `TileShapeQK=16` — PR #575). Config axes: paged, causal, local, sink, lse; lse only valid for (non-paged, non-local, non-sink).
  - Paged decode: `paged_decode_configure.cmake` + `paged_decode_kernel_template.cpp.in`. Axes: qgroup {8,16} × headsize {64,96,128,192,256,512,576} × pagesize {16,32,64,128} × bool {causal,local,sink}.
- Config-file mechanism (`KERNEL_CONFIGURATION.md`, `csrc/xpu/attn/kernel_configs/{chunk_prefill,paged_decode}_{default,full}.conf`): `VLLM_CHUNK_PREFILL_CONFIG`/`VLLM_PAGED_DECODE_CONFIG` pick the preset; missing tuple → runtime error `"Chunk prefill kernel tuple not compiled for this configuration"` with the exact conf line to add. `chunk_prefill_default.conf` = 70 kernels (~2 min build), `full` = 240 (~60 min); `paged_decode_default` = 32, `full` = 384.
- **Our pinned build (v0.1.8.2 source) compiles the FULL product for both**: the 0.1.8.2 `chunk_prefill_configure.cmake` loops all 12 policies × all bool combos, and 0.1.8.2 `paged_decode_configure.cmake` loops qgroup 8/16 × head 64/96/128/192/256/512/576 × pagesize 16/32/64/128 × all bools. So our build has complete full-attn shape coverage (incl. local/sink and pagesize 16–128); no missing-shape fallback issue there. (The conf-file default/full split landed later: kernel_configs dir introduced ~2026-06-01 PR #324 refactor; default confs refined 2026-07-24 PR #482.)

### 1.4 Local (sliding-window) coverage per head size/block size — present & default presets
- v0.1.8.2 build: covered (full).
- **v0.1.14 `chunk_prefill_default.conf`** (non-comment lines): local=true only for `64,false,false,true,false,false`, `64,true,false,true,true,false`, `128,true,false,true,false,false`, `96,true,false,1,false,false`, `256,true,false,true,false,false` — note these are **non-causal local**; there is NO `128,true,true,true,false,false` (paged+causal+local) at v0.1.14 (main has it, added later). If our hybrid's local prefill is causal, a **default-config v0.1.14 rebuild loses local-causal-128 chunk-prefill coverage that v0.1.8.2 had** → use `chunk_prefill_full.conf` or add the exact lines.
- **v0.1.14 `paged_decode_default.conf`**: local decode covered at `8,128,16,false,true,false`, `16,128,16,false,true,false`, `8,128,64,false,true,false`, `16,128,64,false,true,false`, `8,96,16,false,true,false`, `8,256,16,false,true,false`, `8,256,64,false,true,false` — i.e., local decode entries exist for h96/h128/h256 but only pagesize 16/64 (no 32) and **causal=false** (single-query decode uses valid-length masking, causal=false per Gemma/Qwen3-Next comments). **No h512 local, no page-size 128** (pagesize=128 added only in open PR #487).
- Block size relevance applies only to full-attn paged-decode (pagesize = KV block). GDN layers don't use paged KV — they use conv_state/ssm_state caches addressed by `state_indices_tensor`, and never consume the attention pagesize axis.

## 2. Tracker search: GDN + spec decode + capture faults (open ⊙ / closed ✓)

Directly on-target for our two faults:

| # | State | Title | Link |
|---|-------|-------|------|
| 389 | ⊙ | GDN spec metadata shape checks reject graph-padded DFlash decode batches | .../issues/389 |
| 391 | ⊙ | [GDN] Accept graph-padded spec metadata for DFlash decode (fixes #389) | .../pull/391 |
| 593 | ⊙ | [GDN] Reduced active speculative width is rejected when state-index cache retains configured width | .../issues/593 |
| 510 | ✓(09-02) | GDN causal_conv1d rejects mixed spec/non-spec batches — MTP unusable under concurrency | .../issues/510 |
| 537 | ✓(08-19) | [GDN][MTP] fix split PR bug  (the #510 fix; ships in v0.1.14) | .../pull/537 |
| 336 | ✓(05-25, closed **unmerged**) | [GDN] Spec-decoding-aware attention kernel | .../pull/336 |
| 368 | ✓(05-26) | [XE2] Support MTP of QWEN model (native spec API; pairs w/ vllm#43565) | .../pull/368 |
| 320 | ✓(05-18) | GDN kernel rejects padded inputs from torch.compile/cudagraph capture (PIECEWISE) | .../issues/320 |
| 344 | ✓(05-18) | [GDN] Accept padded leading dim and slice to num_actual_tokens (fixes #320) | .../pull/344 |
| 544 | ✓(08-25) | Fix token-indexed conv-state layout in causal_conv1d spec-decode kernel | .../pull/544 |
| 545 | ✓(08-25) | Fix GDN conv state length for strided caches | .../pull/545 |
| 599 | ✓(09-17) | gdn_attention handling of ragged n-gram draft lengths | .../pull/599 |
| 600 | ✓(09-16) | Fix ragged speculative token traversal | .../pull/600 |
| 551 | ⊙ | [GDN] Write spec conv-state from registers instead of epilogue roll | .../pull/551 |
| 552 | ⊙ | [GDN] Fix XE2 delta epilogue OOB write with non-contiguous token_indx | .../pull/552 |

Fault (a) — MTP1 + FULL-graph capture → `UR_RESULT_ERROR_DEVICE_LOST` at the gdn_attn.py spec-state gather:
- Kernel-side fact: **our pinned source has no spec API at all.** The v0.1.8.2 `gdn_attn_interface.cpp` (233 lines) exposes only `non_spec_query_start_loc`/`non_spec_state_indices_tensor`; there is no `num_spec_decodes`, `spec_query_start_loc`, `spec_state_indices_tensor` parameter. The native spec path was added by #368 (2026-05-26, five days after our base commit) and reworked by #402 (07-22), then fixed in #537/#544/#599/#600 (Aug–Sep). So in our build the MTP path must run through vLLM's FLA-Triton fallback (`_gdn_xpu_spec_python_path`) — #336 documents that fallback is ~10× slower and exists precisely because the SYCL op didn't carry the spec tensors; #336 also documents that a broken native spec path caused silent divergence (conv-state layout, max diff ≈47) which takes a long time to manifest.
- No in-repo issue names GDN-spec + DEVICE_LOST-during-capture outright ("DEVICE_LOST" full-text search → only #457). Closest documented mechanisms: **#457 (open)** — Xe2 grouped-GEMM (MoE) `cannot be captured into a SYCL graph at batch > 1`: in-kernel reset of a global atomic tile counter without device-wide barrier → `UR_RESULT_ERROR_DEVICE_LOST` on replay; **#559 (open)** — `moe_gather` `Engine memory CAT error`/DEVICE_LOST on B70 with **exactly our workload** (Qwen3.8-Flash-Next, 512 experts top-10, MTP verify, TP=4), inputs proven in-range, wild VAs hundreds of GB out — top-10 may be the first specialization exercising `MoeGather<bf16,TOPK=10,...>`; **#567 (open)** — ref_fused_moe fails during CUDA graph capture on 2× B70 ("wait method cannot be used for an event associated with a command graph" — host `.item()` inside capture); **#535 (✓)** — head_dim 512/576 register-file exhaustion (fixed by splitting V across grid.x in v0.1.14).
- Since our Fault (a) fires during **capture** (not replay) and is adapter-independent, and our build predates all GDN-spec kernel support, the strongest explanations are (i) the Triton fallback path being graph-capture-unsafe (cf. #336/#487: the PyTorch-ref fallback "breaks XPUGraph capture/replay"), or (ii) a MoE kernel from the #457/#559 class. **UNVERIFIED** — no issue pins DEVICE_LOST to the GDN spec-state gather itself. #487 (open) is also directly relevant: it makes missing FA2 shapes **fail closed** instead of silently falling back to the PyTorch ref path, which "breaks XPUGraph capture/replay" — and notes a `tests/flash_attn/test_xpu_graph_capture_safe.py` (exists only in that PR branch, not main).

Fault (b) — capture size 16 DEVICE_LOST while 1..12 are clean:
- Pattern matches #457 exactly (capture at batch >1 → DEVICE_LOST; workaround pins `cudagraph_capture_sizes` to `[1]`). #457 is MoE-GEMM, open since 07-10 (post-dates our pin, but the pattern class is documented in-repo).
- Alternative GDN-specific candidates: chunked GDN kernels pad by 63 rows per sequence (batch 16 → +1008 rows; sizes 1..12 scale linearly), so a batch-16 prefill/chunk geometry edge (cf. the `T % 64 == 5` all-NaN issue #548, closed — root cause was vLLM's `_is_uniform_decode()` classifier dispatching 64N+5-token prefills through the captured FULL spec graph; fix tracked at vllm-project/vllm#53059/#53051/#49918), or #552's epilogue OOB (open). **UNVERIFIED** — no issue documents a size-16 boundary DEVICE_LOST.

Contract bugs hit by current shapes in newer kernels (relevant for the rebuild): #389/#391 (graph-padded spec metadata rejected: `spec_query_start_loc must have size [num_spec_decodes + 1]`), #593 (`spec_token == num_spec_decodes * (num_speculative_tokens + 1)` assertion; open; follow-up shows narrowing `spec_state_indices_tensor` on the vLLM side via vllm-project/vllm#53542 removes the crash), #599/#600 (ragged/partial spec widths — merged 09-16/17). These are the checks at `gdn_attn_interface.cpp:68-126` (main): `spec_query_start_loc.size(0)==num_spec_decodes+1` (L69), `spec_state_indices_tensor` 2D `[num_spec_decodes, num_speculative_tokens+1]` (L89-95), `num_accepted_tokens.size(0)==num_spec_decodes` (L116), `spec_token <= num_spec_decodes*(num_speculative_tokens+1)` (L120).

Cross-referenced vLLM-side fixes (from #510/#548/#593 threads): vllm-project/vllm#48109 (merged 08-19 — split spec/non-spec dispatch, paired with kernels #537; **required for MTP under concurrency**), vllm#43565 (MTP w/ #368), vllm#53542 (open — narrow spec_state_indices_tensor to active width), vllm#53059/#53051/#49918 (uniform-decode `has_prefill` classifier).

## 3. `.claude/skills/` (kernel-dev agent skills)

- Repo root has `CLAUDE.md` (thin: "read AGENTS.md") and `AGENTS.md` (shared agent guidance: build metadata source-of-truth, uv + venv convention, `source /opt/intel/oneapi/setvars.sh`, current `torch 2.14.0+xpu`/oneAPI 2026.0 expectation).
- `.claude/skills/` contains exactly **one skill: `new-sycl-kernel.md`** (~7 KB) — "port a vLLM CUDA kernel to SYCL" workflow:
  1. Declare in `csrc/ops.h`; 2. implement SYCL functor + launcher (template on scalar type via `SyclTypeTrait`; `vllmGetQueue()`; `sycl::domain` 3D grid; `reqd_sub_group_size(32)`; local_accessor; `sycl::reduce_over_group`); 3. `DISPATCH_FOR_2_DTYPE` in the dispatcher; 4. register in `torch_bindings.cpp` (`ops.def/ops.impl` under the right module `_C`/`_xpu_C`/`_moe_C`/`_vllm_fa2_C`); 5. tests in `tests/test_your_kernel.py` with `opcheck` + `MINI_PYTEST_PARAMS`; 6. build `python -m build --wheel --no-isolation`; run `.venv/bin/python -m pytest`.
  - Also documents: arch checks `is_pvc()`/`is_bmg()`/`is_xe2_arch()`, `aligned_vec` vectorization, `vllm::xpu::acc_type`, rank dispatch `VLLM_DISPATCH_RANK234`.
- No skill covers GDN/chunk-prefill policies, capture safety, or the conf presets — if a kernel-level fix lands, the useful in-repo aids are AGENTS.md + this skill + `KERNEL_CONFIGURATION.md`; there is no skill for "make a junk kernel cudagraph-capture-safe" (that lesson is only in issues #457).

## 4. Release/toolchain compatibility for the planned rebuild (torch 2.13.0+xpu / oneAPI 2026.0)

- **v0.1.14 explicitly claims it**: README at tag `v0.1.14` → "PyTorch: 2.13.0+xpu; oneAPI: 2026.0"; `pyproject.toml` at v0.1.14 → `torch == 2.13.0+xpu`. This is exactly the pair we plan — **the supported, documented combo**.
- Our pinned line instead claims: v0.1.8.2 README → "PyTorch: 2.11.0+xpu; oneAPI: 2025.3" (so torch 2.13/oneAPI 2026.0 on 0.1.8.x is off-matrix, **UNVERIFIED support**).
- Current `main` has moved past it: README/pyproject/AGENTS.md → **torch 2.14.0+xpu**, Dockerfile base `intel/deep-learning-essentials:2026.1.2`. If we rebuild from main, torch must be 2.14.0+xpu, not 2.13.
- v0.1.14 release notes headliners relevant to us: "GDN speculative decoding correctness" (and multimodal, DeepSeek-V4, Gemma4, Python 3.14 support); "Split V across grid.x in paged decode to reduce register pressure for MLA head dimensions 512 and 576" (#535), and FP8/grouped top-k updates. v0.1.14 includes the merged GDN spec fixes #537 (mixed batches) + #544; #599/#600 (ragged) came after v0.1.14 (09-16/17) — need v0.1.14.x+ or main.
- v0.1.12.1 (07-30) notes: "0.1.12.1" mainly adds config/packaging; v0.1.13.1: "use default config to control wheel size" — i.e., **starting v0.1.13.1 the default wheel is the small default-conf build**, so a v0.1.14 rebuild must explicitly pass `VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf` (+ paged_decode_full) or verify every needed local/causal/pagesize line exists in `default` (see §1.4 for the gaps).

## 5. Bottom line for the B70 hybrid GDN project

1. Our kernels are ancient relative to the GDN spec-decode work: no native spec API in-tree (v0.1.8.2), and every GDN-spec bug class documented (mixed spec/non-spec abort, conv-state layout divergence, graph-padded metadata rejection, ragged widths, epilogue OOB) landed 08-19 → 09-17, i.e. v0.1.12→v0.1.14/main. A rebuild to **v0.1.14 (torch 2.13.0+xpu + oneAPI 2026.0 = its documented matrix) is the minimum** for MTP1 + graph capture; consider later tags/main for #599/#600, with torch 2.14.0+xpu.
2. GDN layers need no conf-shape entries (no per-headsize/pagesize precompile; runtime head dims, chunk_size 64 fixed). Only full-attention layers use the conf matrices — and our current build already compiles the full matrix, while v0.1.13.1+/default-preset wheels do NOT (esp. `chunk-prefill local=true causal` at 128 and `pagesize=128` decode). Use `chunk_prefill_full.conf`/`paged_decode_full.conf` or add exact lines.
3. Fault (a) (MTP1 + FULL capture DEVICE_LOST) is most consistent with (i) the pre-spec kernel forcing the Triton fallback path that #336/#487 say is not graph-capture-safe/perf, and (ii) the MoE capture-safety class documented in #457/#559/#567 — open issues as of 2026-09-19. No issue isolates DEVICE_LOST at the GDN spec-state gather. `cudagraph_capture_sizes` pinned to `[1]` is the in-repo-blessed workaround (#457), matching our fault (b) pattern (1..12 fine, 16 → fault).
4. If the fix lands in kernel code: only aid in-repo is `AGENTS.md` + `.claude/skills/new-sycl-kernel.md`; no capture-safety or GDN-specific skill exists (worth creating from #457's root-cause analysis if we patch).

### Verification status
- Verified via raw tagged sources: v0.1.8.2 README/pyproject/configure.cmake/interfaces; v0.1.14 README/pyproject/default confs; main kernels + interface + CMake; issue/PR state via GitHub API (all URLs above).
- UNVERIFIED: (a) exact kernel triggering the DEVICE_LOST at the spec-state gather — no in-repo match; (b) which head dims the Xe2 chunk GDN kernels actually support (no committed guard; #391 draft); (c) torch 2.13.0+xpu + oneAPI 2026.0 support on the 0.1.8.x line (off-matrix); (d) vLLM 0.26.1's exact dispatch (native vs FLA fallback) with these kernels — that lives in vllm-project/vllm `_xpu_ops.py`, not researched here.

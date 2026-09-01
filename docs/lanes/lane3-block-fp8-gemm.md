# Lane 3 — Native XPU block-FP8 GEMM for Qwen3.8-Flash-Next-FP8 on 4× Arc Pro B70

Status: **SPEC ONLY — no rig code changed, nothing executed against hardware by this spec.**
Writer: subagent lane-3 (2026-09-01). Operator executes on the rig ~3 h after issue, after an
attended fresh boot (see §7.1 boot rule).

## 0. Scope, position, traceability

**Charter.** Close the cookbook's #1 named gap — `native XPU block-FP8 GEMM` — for the
`Qwen/Qwen3.8-Flash-Next-FP8` serving push on the 4×B70 box: move the *decode-phase* block-FP8
GEMMs (routed MoE experts + dense projections) off the dequant/W8A8-fallback path onto a native
Xe2 kernel that keeps FP8 weights on silicon and skips per-linear activation FP8-casting, then
qualify it under the lab's preregistered gates. **Decode only**: prefill was never
bucket-profiled ([A28 JSON] l.1–32 timing permanently diagnostic; see §7.3) and this lane ships
no TTFT claim. **No TP6** (2 KV heads ⇒ TP ∈ {2,4,8}; current TP4+EP4 stays) [S:11–12].

**Position vs the rig's schedule (collision rule).** [A31] (`notes/...a31-moe-m1-current-prereg.md`)
is the rig's FIRST scheduled MoE test — frozen, M1-only warps-8 Triton map on the current-source
`797769b34` checkout, blocked on the attended reboot (a31:3, a31:27–31, a31:43–55). Lane 3 is the
**follow-on**: it must not run before A31 has claimed the fresh boot's first full-model-load slot,
must not duplicate A31's cells (A31 changes Triton-MoE `num_warps` on key M1; Lane 3 changes the
*linear/quantization* dispatch — disjoint), and must cite [A30] as the known grouped-HC composite
negative (`notes/...a30-hc-grouped-m1-endpoint-negative.md`, −1.82% speed-gate FAIL, `5.415455`
vs protected `5.515783` tok/s; a30:14–19). Lane 3 keeps `VLLM_XPU_QWEN4_EXP_HC_GROUPED_UP` unset.

**Read-first evidence (this machine, read-only):**
`/tmp/flashnext-lanes/00-shared-context.md` (shared facts, all lanes) · `/tmp/laneB_cookbook.md`
(cookbook digest) · `/tmp/b70lab/` (lab) · `/tmp/b70cookbook/` (SergiioB cookbook).

**Citation legend.** `S:` = 00-shared-context.md · `R:` = `b70lab/results/qwen38-flash-next-fp8-b70/README.md` ·
`N:` = `b70lab/experiments/qwen38-flash-next-fp8-b70/notes/` · `D:` = `.../data/` · `T:` = `.../tools/` ·
`CB:` = `/tmp/b70cookbook/...` · `P:` = `b70lab/patches/qwen38-flash-next-fp8-b70/...` ·
`GH:<file>@76cfe1cd88` = upstream vLLM file pinned at the overlay base commit (fetched read-only for
this spec) · `WEB:` = web-verified external claim. **DERIVED / INFER** mark arithmetic or inference,
not observed lab numbers.

---

## 1. Status quo — which GEMM implementations actually execute on XPU today

The runtime is a source overlay at base `76cfe1cd88` + production patches `0001–0010, 0012,
0014–0018` (0011/0013 = diagnostics, NEVER on timing trees) → head `1372c62d…`; kernel stage
loaded at `2f829747…` (tree `d8c4318a…`); installed pip metadata lies [S:27–37]. Patch `0001`
upstream-merge is the only series member touching FP8 quantization code, and it only adds
MoE block-scale-refinement logic in `layers/quantization/fp8.py` (P:0001, hunks in
`vllm/model_executor/layers/quantization/fp8.py`); **none of patches 0001–0033 touch
`kernels/linear/scaled_mm/xpu.py` or `kernels/linear/__init__.py`** (verified by scanning all 33).
Therefore the *live FP8 dispatch is the base-behavior dispatch* below.

### 1.1 Dense block-FP8 linear projections (QKV/O, shared-expert, LM-head class)

- **Dispatch registration** — `GH:kernels/linear/__init__.py@76cfe1cd88`:
  - `_POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.XPU] = [XPUFp8BlockScaledMMKernel]` (l.430–451).
  - Non-block FP8: `_POSSIBLE_FP8_KERNELS[XPU] = [XPUW8A16FP8LinearKernel, XPUW8A8FP8LinearKernel]`
    (l.420–422); `_POSSIBLE_WFP8A16_KERNELS[XPU] = [XPUW8A16FP8LinearKernel]` (l.468–469).
- **What actually runs** — `GH:kernels/linear/scaled_mm/xpu.py@76cfe1cd88`,
  `class XPUFp8BlockScaledMMKernel` (l.192–285):
  - `apply_block_scaled_mm` (l.267–284) calls **`torch.ops._xpu_C.fp8_gemm(A, B.t(), out_dtype, As, Bs.t(), …)`** — the oneDNN-backed XPU scaled-MM op (scale layout contract `[k_blocks, n_blocks]` for oneDNN, l.235–241; ragged-N GCD-16 reshape l.208–226; ragged-K fail-closed l.228–233).
  - Upstream, `Fp8BlockScaledMMLinearKernel.apply_weights` performs a **dynamic per-token-group FP8 activation quantization before every linear** (`GH:.../scaled_mm/BlockScaledMMLinearKernel.py@76cfe1cd88`, `apply_input_quant=True` class default l.47; `quant_fp8(...)` l.119–123). XPU keeps that default — its block kernel has no `apply_input_quant=False`.
- **Interpretation.** Dense block-FP8 linears run the **oneDNN W8A8 path with a per-linear FP8 activation quant** — FP8 weights and FP8 activations fed to oneDNN's Xe2 GEMM with block-scale post-processing. This is the *dequant/requant fallback* family the cookbook documents as "FP8 (block) … Dequant fallback only" (`CB:research/quantization-format-strategy.md` l.59) and as the old `KeyError: PlatformEnum.XPU` hard-absence (`CB:.../DENSE-FP8-GAP.md` l.18–26, that KeyError predates this base). It is NOT a fused native FP8×BF16/FP8×FP8 DPAS kernel — the gap this lane closes. (INFER: oneDNN's internal Xe2 path is int8/FP16-DPAS-oriented; not lab-observable.)

### 1.2 Routed / shared MoE experts

- **Backend selection** — the lab forces **Triton MoE**: `--moe-backend triton` is the explicit
  frozen first-load identity (N:2026-08-26-xpu-overlay-preload-gates.md l.103–106 "only `_C` and
  `_moe_C` are rebuilt … explicitly selects `--moe-backend triton`", l.131 "Triton MoE backend
  (native grouped MoE is not correctness-qualified)"). MTP preregs record "Triton MoE" (N:2026-08-27-tp4-mtp4-512-prereg.md l.19); A28/A27 notes say decode enters "the Triton experts" (N:a28-collective-timeline l.46–50).
- **Per-rank EP4 expert shape** — E=128 local experts/rank, N=640 (intermediate), K=2560 (hidden),
  top-k=10, `dtype=fp8_w8a8`, `block_shape=[128,128]` (T:tools/fullshape-triton-fp8-moe-gate.py
  l.198–217; MoE tuning maps `configs/moe-warps8-{m1,m4}/…dtype=fp8_w8a8,block_shape=[128,128].json`).
  FP8 E4M3FN weights + FP32 block scales (gate tool l.196–197, l.305–322). The non-EP TP4 shape
  (512 experts, intermediate 160, refined `[32,32]` scales) exists for the no-EP alternative only
  (gate tool l.200–201).
- **Activation casting is inside the per-token path**: the fused `silu_and_mul + block-quant`
  kernel (`csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cpp`) is part of the loaded
  kernel stage (P:vllm-xpu-kernels-certified-2f829747/0003 … "include fused block-FP8 and MXFP4
  SiLU-multiply implementation translation units"; README l.35–41). Activation-block FP8 quant
  therefore fires per routed-MoE step.
- **Native grouped MoE is deliberately DORMANT**, and this is the lab's own forward plan verbatim:
  "reconcile the ABI, reapply the upstream block-FP8 scale path onto the local INT8/INT4/MXFP4
  optimizations, build in isolation, and prove numerical parity before promotion"
  (N:2026-08-26-xpu-overlay-preload-gates.md l.111–113; see also l.96–101 on the displaced
  block-FP8 128×128 scale path and wrong-scale-semantics hazard).

### 1.3 Proof this is what A28 measured (not speculation)

The A28 decode profile classifies device events by kernel name
(T:tools/summarize-tp4-target-decode-kineto.py l.160–216: `quant|dequant|fp8|cast` →
`quantization_cast`; `gemm|brgemm|xetla_gemm` → `dense_projection`; `moe::|fused_moe|silu_and_mul`
→ `routed_shared_moe`). Mean ms per retained decode cycle across rank means
([A28 JSON] l.66–79): `routed_shared_moe 26.084` · `dense_projection 10.031` (robust clean
`7.575` after rank-3 outlier audit, l.80–104) · `quantization_cast 4.185` · `elementwise 2.943` ·
`qsa 2.208` · `moe_router 0.789` · `gdn 0.409`. Per cycle: **532 GEMMs, 96 fused_moe events,
97 BF16 allreduces** ([A28 JSON] l.105–114). The `quantization_cast` bucket is exactly the
per-linear `quant_fp8` casting + MoE block-quant that a native path removes. **Timing is
permanently ineligible for speed credit** ([A28 JSON] l.32; N:a28-result l.13) — diagnostic only.

---

## 2. Gap analysis — what a native block-FP8 GEMM replaces

### 2.1 Replaced work (per decode cycle, A28 numbers)

| Bucket | Mean ms/cycle | What a native kernel removes/changes | Mark |
|---|---|---|---|
| routed_shared_moe | 26.084 | FP8-activation W8A8 Triton MoE → native W8A16-style block-FP8 GEMM (BF16 A direct), fused block-scale multiply in inner loop, no separate quant/dequant passes | DERIVED (from §1.1–1.2 paths) |
| dense_projection | 10.031 raw / 7.575 clean | oneDNN W8A8 `fp8_gemm` → native block-FP8 op for M≤40 (decode); removes act-quant feeders | DERIVED |
| quantization_cast | 4.185 | **mostly eliminated** for covered linears/MoE (per-token quant moves inside GEMM or disappears) | DERIVED |
| elementwise | 2.943 | partial: dequant/scatter/functor work absorbed | INFER |
| qsa / gdn / moe_router / ple | 2.208 / 0.409 / 0.789 / 0.002 | **not a speed target of this lane** — A28 explicitly rules these out (`strongest_concrete_noncollective_target` = production M1 routed-MoE + dense projections; `not_supported_as_primary_speed_targets` = GDN, QSA, PLE) [A28 JSON] l.123–128 | — |

The three covered buckets sum to ≈ 40.3 ms/cycle raw (26.084+10.031+4.185). The noncollective
deck is NOT the whole story: the primary bottleneck class is collective critical-path +
cross-rank arrival imbalance (97 allreduces/cycle; Kineto-distorted `collective_allreduce`
residence ≈ 197 ms is not additive wall time) [A28 JSON] l.115–129; N:a28-result l.61–67.
**Honest framing: this lane attacks the strongest concrete noncollective target; it is not by
itself the 3×++.**

### 2.2 Shape contract the native op needs (decode)

- Per-expert (EP4): M=1 (production decode enters routed MoE at **M1** — N:a28-collective-timeline
  l.46–50; A31's whole premise), K=2560, N=640, blocks [128,128] → `K%128==0`, `N%128==0` ✓.
- Dense linears: hidden 2560; TP4-sharded outputs are multiples of 128 by divider convention;
  **must be enumerated and verified at preflight** per layer (the cookbook kernel hard-checks
  `A[M,K] BF16/FP16 · B[K,N] E4M3FP8 · B_scale[K/128, N/128] FP32 · 1≤M≤40`
  — CB:patches/vllm-xpu-kernels/apply_smallm_patch.py l.265–316).
- Prefill (M ≫ 40) and batch decode above M=40 **remain on the fallback path** — Lane 3 is decode-cell only (M=1 single-sequence, matching every MTP0 protected row).

---

## 3. Candidate approaches, ranked with tradeoffs

Every upstream/external claim below is tagged. No invented upstream facts: anything not tagged
LAB-LOCAL or WEB-SOURCED below is the author's inference and marked INFER.

### T1 — Native Xe2 block-FP8 GEMM via the authenticated small-M kernel (PRIMARY)
**LAB-LOCAL, already written for this silicon-family:** cookbook patches
`patches/vllm-xpu-kernels/apply_smallm_patch.py` (+`fix_smallm_placement.py`) implement, against
the vllm-xpu-kernels tree, P1–P6: a `xe_gemm_block_fp8` block-scaled inner loop in
`grouped_gemm/xe_2/gemm_xe2.hpp` using `cute::gemm` XMX/DPAS accumulation with the FP32
`[K/128, N/128]` block scale applied inside the K-loop (`tCrC(i) += tCrC_block(i) * B_scale` at
K-block boundaries), an E=1 impl, arch dispatch, and `torch.ops._xpu_C.xe2_block_fp8_small_m`
registration (apply_smallm_patch.py l.45–453). Hard-contract: A BF16/FP16, B E4M3FP8, M≤40,
K,N %128==0. Authored 2026-08-29 (b70-architect) for the Qwen3.8-27B FP8 TP2 recipe; **needs a
certified-series rebuild** (S:100–101: "needs certified-series rebuild").
- Tradeoffs: real native FP8 compute on Xe2 with fused block scaling (closes the named gap);
  but (a) requires building a NEW sealed runtime stage (certified 7-patch series `0fd18a7c` →
  tree `d8c4318a…` → +P1–P6 → build → gate → package), (b) 120+ GB RSS per compiler process
  (S:33), (c) oneAPI 2025.3 pinned for `libsycl.so.8` ABI (P:README l.36–41; N:preload-gates l.88–92),
  (d) M≤40 ⇒ decode-only, (e) A is BF16 (W8A16-style) not FP8 — FP8×FP8 would need a second kernel.
- Integration effort: medium-high (one focused build, then pure dispatch wiring). **Primary.**

### T0 — Block-FP8 W8A16 reroute on the EXISTING sealed stage (FAST TRACK, ships first)
**LAB-LOCAL prior art, zero rebuild:** the loaded VXK stage exports
`torch.ops._xpu_C.fp8_gemm_w8a16(A_bf16/fp16, B_fp8[K,N], B_scale, bias)` — "consumes BF16
activations DIRECTLY against block-scaled FP8 weights" (CB:patches/patch_fp8_w8a16.py l.6–9; op
exists in the VXK `.so` op set — WEB: vllm#39474 lists `fp8_gemm`, `fp8_gemm_w8a16`).
The cookbook's production Qwen3.8-27B image reroutes `XPUFp8BlockScaledMMKernel` to it with
`apply_input_quant=False`: −25% step time, −20–40 W, +34% at p1024 (CB:patch_fp8_w8a16.py l.11–19;
CB:README l.41). The lab's own older patch (`P:patches/vllm-c51df4300-xpu-qwen36-block-fp8-fallbacks.patch`)
contains the same idea as an env-gated `VLLM_XPU_BLOCK_FP8_REQUANT=1` (per-channel requant variant).
- Tradeoffs: no rebuild, no ABI risk, directly kills the per-linear activation quant that feeds
  A28's `quantization_cast` bucket; BUT uses oneDNN's W8A16 GEMM (an existing op, not a new native
  FP8 kernel), requant/scale-semantics change ⇒ **mandatory parity gate**, and it does not by itself
  give the MoE a native expert GEMM. **Ship T0 first (cheap, isolates the dispatch layer), then T1
  (silicon).** They compose: T0 removes act-quant, T1 replaces the GEMM.

### T2 — Triton block-FP8 linear/MoE kernel on XPU (LOW CONFIDENCE)
**WEB-SOURCED hazards:** Triton's official block-scaled-matmul tutorial targets NVIDIA/AMD general
programmability, not XPU (WEB: triton-lang.org tutorial 10); XPU Triton `tl.dot` FP8 support is
immature (WEB: intel/auto-round#827 "xpu does not support fp8 model as input due to triton";
NVIDIA-side float8 dot silently falls back to FP16 MMA — WEB: triton-lang/triton#7188). The current
Triton fp8_w8a8 MoE configs already run (LAB-LOCAL §1.2) but measure 26 ms — the very bottleneck.
**Worth writing only if T0/T1 both fail; not a 3-hour rig deliverable.** INFER: a hand-written XPU
Triton scaled-MM could still win for odd shapes oneDNN refuses, hence it is ranked not dropped.

### T3 — oneDNN / upstream vllm-xpu-kernels contribution (LONG-HORIZON, not a rig window)
**WEB-SOURCED:** VXK v0.1.13 rebuilds target "Xe2 grouped-GEMM path updates … FP8 and small-K"
(release notes) and upgraded oneDNN to v3.13; upstream added XPU wna16 (`vllm#33973`) and an XPU
scaled-MM kernel (`vllm#34117`) under the kernel-migration RFC (`vllm#33214`). The cookbook itself
names the upstream xpu_kernels FP8 contribution as the medium-term fix with "Medium" likelihood
(CB:quantization-format-strategy.md l.147). If T1's kernel is good, feed it upstream — out of scope
for the 3-hour execution but the right long-term home.

### T4 — IPEX / IPEX-LLM path (LEGACY-IN-VLLM, REJECT)
**WEB-SOURCED:** vLLM XPU formally deprecated the IPEX dependency and moved to vllm-xpu-kernels
(vllm#33214 → "Done; [1/N] Deprecate ipex and switch to vllm-xpu-kernels for xpu platform #33379");
IPEX's own FP8 story is prototype-level "online Dynamic Quantization" for compression/decompression,
not a fused inference GEMM (WEB: IPEX float8 docs). No integration path that doesn't build a parallel
stack. **Rejected for this lane.**

**Ranking: T1 primary · T0 first-to-ship · T2 contingency · T3 upstream follow-up · T4 rejected.**

---

## 4. Integration plan (overlay + vLLM XPU dispatch, TP4+EP4)

### 4.1 Dispatch wiring (the only vLLM-source delta, kept tiny and env-gated)
- Add ONE new kernel class next to `XPUFp8BlockScaledMMKernel` in
  `vllm/model_executor/kernels/linear/scaled_mm/xpu.py`, e.g. `XPUBF16Fp8BlockNativeMMLinearKernel
  (Fp8BlockScaledMMLinearKernel)`:
  - `apply_input_quant = False` (base-supported; BlockScaledMMLinearKernel.py l.47, l.119–130);
  - `can_implement`: XPU only + `weight_quant_key == kFp8Static128BlockSym` + shape contract
    (K%128==0, N%128==0, current M path ≤40 — enforce per-call);
  - `apply_block_scaled_mm` → `torch.ops._xpu_C.xe2_block_fp8_small_m(A, B.t()/*[K,N]*/, Bs/*[K/128,N/128]*/, bias)` (T1) or
    `torch.ops._xpu_C.fp8_gemm_w8a16(A, B.t(), Bs.t(), bias)` (T0), selected by env
    `VLLM_XPU_FP8_BLOCK_BACKEND=B70NATIVE|W8A16|default`.
- Register it in `_POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.XPU]` **ahead of** the stock kernel
  (`GH:kernels/linear/__init__.py@76cfe1cd88` l.450–451) so it wins `can_implement` when the
  contract holds and the stock oneDNN W8A8 path is the automatic fallback otherwise (ragged shapes,
  M>40, prefill). **Default-off**: unset env = byte-identical stock behavior. This is the lab's
  established patch posture (A31: "requires `VLLM_XPU_QWEN4_EXP_HC_GROUPED_UP` unset and false";
  N:a31 l.29–31; 0032 HC grouped-up is default-off, P:README l.37–48).
- Keep `--moe-backend triton` (S:lab state) for Lane-3 first cells; the Triton-MoE block is the
  A31-inherited baseline (see §5.3 factorial rule). The native-block-FP8 MoE expert path (routing
  each M=1 expert GEMM of the EP4 tile to the T1 op) is a **second, separate factorial arm** so
  Lane 3 never confounds with A31's warps question (§5.3). INFER: the E=1 op is literally named
  for this (P3 "E=1 impl", apply_smallm_patch.py l.263–264), so the MoE arm is a dispatch change in
  the XPU fused-MoE/Triton wrapper, not new kernel code.

### 4.2 Runtime-stage strategy (two tracks)
- **Track-0 (T0): NO rebuild.** Reuses the sealed 18-file stage `2f829747…`/tree `d8c4318a…`;
  only the vLLM-side Python deltas above. Stage identity unchanged; launcher/client/supervisor
  hashes re-frozen per A29 discipline (N:a29-prereg l.61–69).
- **Track-1 (T1): one new sealed stage.** Reconstruct certified series from fork base
  `0fd18a7c…` + certified 7-patch series → tree `d8c4318a…` (P:vllm-xpu-kernels-certified-2f829747/
  `README.md` l.1–13), then apply P1–P6 (`CB:patches/vllm-xpu-kernels/apply_smallm_patch.py` +
  `fix_smallm_placement.py`) in an **isolated** build tree (never touching the accepted stage —
  N:preload-gates l.88–113), oneAPI **2025.3** for `libsycl.so.8` ABI, MoE+GDN enabled, B70-only
  AOT, one compile job, ext drive build storage; then gate (import / op-registration / arch /
  grouped-GEMM / numerical), then package with `b70lab/scripts/package-qwen38-runtime-stage.py`
  (P:README l.92–98) as a NEW 18-file hybrid stage with its own manifest SHA. **Freeze the new
  stage tree+manifest SHA before any launch.**
- Kernel-stage identity warning: checkout `ad25aa9f…` is source; the loaded stage `2f829747…` is a
  binary build; the loose `vllm-xpu-kernels/` dir is superseded (P:README l.15–26). Only the
  certified series + verified tree hash qualifies.

### 4.3 TP4+EP4 shape / per-rank expert shard
- EP4 per-rank shard: **E=128 experts, N=640, K=2560, blocks [128,128]** (gate tool l.202–204);
  `--moe-backend triton`, `allgather_reducescatter`, single sequence ⇒ decode M=1, no per-layer EP
  all-gather (decode "enters the routed MoE kernel at M1, without a per-layer EP all-gather",
  `use_all2all_kernels` false when DP=PCP=SP=1 — N:a28-result l.70–73; N:a28-collective-timeline
  l.46–50; A28 profile's `strongest_concrete_noncollective_target` names "the production M1
  routed-MoE path and dense projections" — [A28 JSON] l.123).
- Keep PLE synchronous pinned-UVA placement (`cpu_offload_gb=12.25`, 12.22 GiB/rank) exactly as
  frozen (S:20–22, establish-only; **no async prefetch** — §7).
- Dense linear per-rank (K,N) inventory must be dumped at preflight (a tiny report-only hook or
  offline from `meta_tp_construct.py` — T:tools/meta_tp_construct.py) and every covered layer must
  satisfy the native op contract; uncovered layers silently keep the base path (that is by design —
  the dispatch floor is `can_implement`).

### 4.4 Dependencies / environment (frozen, observed-not-installable S:34–36)
Python 3.12.13 · torch 2.11.0+xpu · triton-xpu 3.7.0 · oneAPI 2025.3.2 · oneCCL 2021.17.2 · pinned
(but NOT pip-reinstallable) — do not "helpfully" upgrade.

---

## 5. Preregistered qualification gates (lab style — freeze BEFORE measuring)

### 5.1 Identity (hash-locked before launch; A29/A31 pattern)
- Model rev `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` + tree SHA
  `4a3793bd…eb2` (R:l.135–139). vLLM overlay tree/head frozen at prereg — recommended
  **current-source checkout `797769b34…`** (A31's line, N:a31 l.27–29) so Lane 3 inherits the
  post-reboot baseline A31 qualified; **MTP cells are the only exception and stay on `1372c62d`**
  (lane2 l.32, not Lane 3's concern). Kernel source `ad25aa9f…`; loaded stage `2f829747…`
  (Track-0) or the new Track-1 stage + manifest. Launcher/client/supervisor/workspace-contract
  SHA-256 frozen (A29 l.61–69). A **prelaunch resolver** must emit the selected MoE key/effective
  config and asserted equal to vLLM's official resolver (A27 lesson — N:a27 l.28–30: a file-load
  receipt is insufficient; emit the selected key).
- Attempt/port: **recommend attempt 32 / port 19704** for the first Lane-3 endpoint cell (next free
  after A31; no lane claims A32 as of this spec — check `flashnext-lanes/*` before freezing); each
  subsequent cell gets a new immutable run/cache/evidence root + port (lane2 l.134).

### 5.2 Speed gate (how A30/A31 define pass/fail — cite verbatim)
- **Pass requires the short-path (p146/o256/c1) median to EXCEED the protected `5.515783 tok/s`
  MTP0 median.** A30 failed at 1.82% below (`5.415455`; a30:18–19); A27 got no credit at 0.255%
  below (`5.501703`; n:a27:15–17); A26 failed at 2.17% below (`5.395973`; n:a26:17–20). At-or-below
  the protected median ⇒ FAIL, no speed credit.
- A passage "is only a candidate" (N:a31 l.50–55); **causal promotion requires a separately booted
  flag-off (env-unset) control on identical source + a fresh candidate repeat**, same client/battery
  (A29 l.37–41; A31 l.50–55). If A31 passed earlier in the same boot family, Lane 3's candidate
  must also clear A31's landed median (record it at prereg as the post-A31 baseline).
- Sanity floors (also gates, from the subset): 3 sequential single-request samples, median-of-3,
  all three rows same output hash, cache-zero (R:l.48–62; A30 l.12–19). Run the full client battery:
  recovery canary, inherited 6/7 semantic boundary (only known code-case miss allowed), 16/16
  repeat single-hash, locked cache-zero 4K needle, then speed rows, then two exact-4K authority rows
  (A30 l.12–27).

### 5.3 Output-parity gate — SEPARATE from speed (cookbook methodology S:68: "speed is not proof of parity")
- **Endpoint token-parity**: exact-4K rows must byte-match the retained MTP0 authority hash
  (`1d833e5f…d5cc` family — A28/A26 exact-4K authority; N:a26:24–25, [A28 JSON] l.28); short rows
  must return the protected short digest; repeats one hash 16/16. Any first-token divergence ⇒
  fail-closed billing (A30 row1 div@18, row2 div@2 — a30:22–27; the frozen-cross-lane MTP3-4K hash
  miss precedent R:l.470–476).
- **Tensor-digest gate (A16/A17 discipline)**: component arms hash raw bytes of model positions /
  inputs / outputs, and for the parity-critical comparison hash **all 149 layer-ordered tensor
  digests** — "if all 149 tensor digests match, classify as stabilizing … do not invent a layer
  cause"; any mismatch is fail-closed (N:a16-prereg l.24–31; N:a17-prereg l.12–20). For the native
  GEMM specifically: run the extended real-weight harness (T:fullshape-triton-fp8-moe-gate.py with
  `--weights layer0-rank0-checkpoint`, `--repeats 100`, 3 seeds) and require one hash across all 300
  invocations vs the frozen fallback path (component-positive precedent N:2026-08-30-moe-m1-warps8-component-positive.md l.6–15).
- **Disclosure stays on**: T0 changes numerics (block-scale handling moves inside fp8_gemm_w8a16);
  T1 changes it again (BF16 A). Neither is "the same arithmetic"; parity gates exist precisely to
  accept/reject. Lossless is REQUIRED for endpoint promotion; a lossy-but-identical-outputs arm is
  at best a bounded screen, never a record.

### 5.4 Factorial rules (collision-proofing)
- Lane-3 first cells vary ONLY the linear/quant dispatch (env backend knob). The MoE block stays
  exactly the A31 inherited identity (map applied/unset per A31's landed outcome — do NOT change
  warps in Lane-3 arms).
- The "MoE experts on native kernel" arm is a SECOND preregistered cell, run only after the linear
  arm resolves, and it must re-freeze its own identity/speed/parity gates (new attempt/port). This
  keeps A31 (warps) and Lane 3 (GEMM) attributable.
- Wedge watchdog MANDATORY in every run recipe (see §6): supervisor owns PID+process-group+port,
  external-kill no-progress bounds (never in-process SIGALRM — lane2 l.250), kernel-journal grep
  `engine_class=(ccs|bcs)|Fault response|timed out|reset` (CB:watchdog + S:46–49), fail run on hit;
  front client with retrying proxy (in-flight requests lost on restart — CB:watchdog README).

---

## 6. Risks

1. **Xe2 Level-Zero wedge (mandatory mitigation, every run):** dies every 2–6 h under sustained
   load; kernel signature `Engine reset: engine_class=ccs|bcs`, `Fault response: Unsuccessful`,
   `guc_exec_queue_timedout_job`; only container/engine restart recovers, in-flight requests lost
   (S:46–49; CB:watchdog/; WB vllm#41663). Use cookbook `install-watchdog.sh` or the lab supervisor
   pattern; no Lane-3 run starts without it.
2. **Host RAM during Track-1 compile:** kernel rebuilds need 120+ GB RSS per compiler process
   (S:33); 64 GiB swap + ~40 GiB root floor enforced by supervisors a5–a11 (S:44–45). Never run the
   rebuild concurrently with a model load; keep one compile job; expect the a5/a6/a7-class OOM
   failure modes if floors are violated (lane1 l.34–40).
3. **Scale-semantics correctness (highest model risk):** the lab already suffered one block-FP8
   128×128 scale-path displacement ("wrong scale semantics" hazard, N:preload-gates l.99–101). The
   native kernel and the W8A16 reroute both move scale handling; the parity/tensor-digest gates are
   the only acceptable referee. Ragged-N GCD-16 and ragged-K fail-closed constraints stay (xpu.py
   l.208–233).
4. **ABI / stage integrity:** oneAPI 2025.3 (`libsycl.so.8`); never overwrite the accepted stage;
   build in isolation then package a new stage; verify with `verify-certified-source-series.py`
   (P:README l.23–26) before trust. `SYCL_CACHE_PERSISTENT=1` poisons cache → SEGV next boot (lane2 l.249).
5. **Wheelhouse gaps (honesty):** prefill never bucket-profiled (no prefill credit possible);
   native op is M≤40 ⇒ decode-only; the collective/arrival imbalance is the primary bottleneck class
   and this lane does not fix it; 532 GEMMs/cycle means even a perfect GEMM caps at the noncollective
   slice (~40 ms/cycle raw). Do not extrapolate tok/s gains from ms/token deltas without the endpoint
   gates (DERIVED).
6. **Quiet-host/clean-host wording:** corrected Samsung-NVMe / root-port events are perennial and
   auto-quarantine any cell that declares clean-host (lane2 l.252); check the kernel journal before
   blaming runtime (S:52–53).

---

## 7. What-NOT-to-do (A26/A27/A30/A28 negatives — no blind retries)

1. **No async-UVA PLE prefetch re-proposal** — A26 endpoint negative: `VLLM_XPU_PLE_UVA_PREFETCH=1`
   rejected at exact-4K repeat/authority (`n:a26:29–35`: "do not promote … do not repeat this
   endpoint arm"). Synchronous pinned-UVA is the established, ONLY placement.
2. **No M4-file-load-only MoE arms** — A27: a loaded tuned file is not a selected key; must emit the
   effective M1 key/`num_warps` (n:a27:28–30). Lane 3 never ships a tuning map without a resolver receipt.
3. **No grouped-HC composite re-testing** — A30: −1.82% speed gate FAIL and unmatched 4K; "Do not
   spend two more full loads on the matched flag-off/flag-on attribution sequence" (a30:38–42).
   `VLLM_XPU_QWEN4_EXP_HC_GROUPED_UP` stays unset/false in every Lane-3 cell.
4. **No BF16-allreduce LL-threshold endpoint spend** — component-neutral 1.13% (median), below the
   5% gate: "do not add it to the protected launcher" (n:2026-08-30-tp4-bf16-allreduce-ll-threshold-neutral.md l.13–26).
5. **No graph capture attempts** — quarantined-negative a1–a7 for Flash-Next (S:89–94; lane1 l.30–40).
6. **No TP6, no TP1/TP2 redesign, no dependency "upgrades"** — TP∈{2,4,8} only (S:11–12); pip install
   of requirements WILL drift (S:35–37).
7. **No profiler timing for speed** — A28 profile timing is permanently ineligible (l.32); wall-clock
   client rows only.
8. **No recombination of kernel artifacts** — loose `vllm-xpu-kernels/` dir is superseded
   (P:README l.15–26); `ad25aa9f` ≠ loaded stage `2f829747`; patches 0011/0013 never on timing trees
   (S:49).
9. **No parallel GPU work with a boot's single model-load slot** (A31 boot rule n:a31:43–55); Lane-3
   cells schedule after A31, same-boot discipline per §5.4.
10. **Do not re-run A31's cells to "verify" them** — A31 is frozen; Lane 3 inherits its outcome, never re-measures its map.

---

## 8. Receipts — exact commands an operator runs (dependency order)

All rig paths below are per the lab's established layout; hashes/roots must be re-frozen at
preregistration (this list is the dependency skeleton, every step immutable-receipted with `sha256sum`).

### 8.0 Boot/slot + environment gates (hard prerequisites)
```bash
# 1) Fresh attended boot ONLY (current boot c36480de… is rejected — lane2 l.251; n:a31:43–45).
# 2) Ordinary-XCCL affinity component first (a31 prereg requirement), then A31 claims the boot's
#    first full-model-load slot. Lane 3 runs AFTER A31 resolves (pass or clean close).
# 3) Wedge watchdog live:
cd /tmp/b70cookbook/watchdog && sudo ./install-watchdog.sh   # or lab supervise-*.sh pattern
# 4) Kernel journal quiet host check BEFORE blaming runtime:
journalctl -k -b | grep -iE 'xe 0000:(23|27|43|47)|nvme.*receive|Fault response|Engine reset' || true
```

### 8.1 Identity verification (freeze before launch; each command produces an immutable receipt)
```bash
cd /tmp/b70lab/patches/qwen38-flash-next-fp8-b70
sha256sum verify-certified-source-series.py
VERIFY=$(python3 verify-certified-source-series.py)   # asserts tree d8c4318a… from 0fd18a7c… + 7 patches
# Model identity
sha256sum -c <<<"4a3793bd4a795ea6761b3d322200b4a1fd8300cdeb75cc127d330d513f590eb2  <model tree receipt>"
# Stage identity (Track-0): sealed 18-file hybrid stage manifest:
sha256sum -c runtime-stage-padding-guard-loadable.sha256   # stage 2f829747…
# vLLM overlay: export overlay git HEAD == frozen (recommended 797769b34… for A31-line source)
git -C <vllm-src> rev-parse HEAD | tee identity-vllm.sha256
```
(For Track-1 only, add §4.2 build/re-gate/package receipts: certified series → tree `d8c4318a…` →
P1–P6 via `VXK_SRC=<csrc/xpu> python3 apply_smallm_patch.py` → isolated oneAPI-2025.3 build → gate
→ `b70lab/scripts/package-qwen38-runtime-stage.py --stage … --archive …` → freeze new manifest SHA.)

### 8.2 Dispatch patch + preflight shape census
```bash
# Apply the §4.1 vLLM-side delta (one file: linear/scaled_mm/xpu.py + one registration line in
# kernels/linear/__init__.py), env-gated default-off. Hash-bound:
sha256sum vllm/model_executor/kernels/linear/scaled_mm/xpu.py vllm/model_executor/kernels/linear/__init__.py
# Preflight: per-layer dense (K,N) census + EP4 expert tile contract (K%128 && N%128, M≤40 decode):
python3 tools/meta_tp_construct.py | tee preflight-shapes.json
# Prelaunch MoE-key resolver (A27/A29 discipline — must emit key M1, num_warps, effective config,
# equality with vLLM official resolver):
python3 tools/rewrite-a29-kernel-workspace-contract.py # (or the A31-line equivalent) | tee resolver-receipt.json
```

### 8.3 Launch / battery (clone A31 trio, isolated identities A32/port 19704 — see §5.1)
```bash
# Env per §5.3/§7: VLLM_XPU_QWEN4_EXP_HC_GROUPED_UP unset · PLE uva prefetch unset · VLLM_XPU_FP8_BLOCK_BACKEND=W8A16 (T0) or B70NATIVE (T1) · no graph · MTP0 · eager.
cd /tmp/b70lab/experiments/qwen38-flash-next-fp8-b70/tools
bash launch-tp4-mtp0-4352-ple-only-a32-blockfp8.sh      # frozen hash
bash run-tp4-mtp0-4352-ple-only-a32-blockfp8-client.sh  # recovery canary → semantic 6/7 → 16× repeat
                                                        # → 3× p146/o256 speed rows → cache-zero 4K needle
                                                        # → 2× exact-4K authority rows
bash supervise-tp4-mtp0-4352-ple-only-a32-blockfp8.sh   # watchdog-owned group, journal classifier
```
Postflight: client/supervisor rc 0, no B70 reset record, four cards ≤ ~43 MiB, host mem/swap
recovered (A30/31 teardown discipline), then `sha256sum` every JSON/log = receipt.

### 8.4 Component parity (A16/A17 discipline) — before or after §8.3, isolated process
```bash
VLLM_XPU_FP8_BLOCK_BACKEND=W8A16 python3 tools/fullshape-triton-fp8-moe-gate.py \
  --distributed-mode ep4 --weights layer0-rank0-checkpoint --routing balanced-global \
  --tokens 1 --repeats 100 --hidden-seed 20260826 --result-dir r1   # 300/300 single-hash = pass
# repeat for the linear-path 149-tensor digest comparison per N:a16/a17
```

### 8.5 Closeout
PASS/FAIL each gate with numbers in a result note following the `2026083x-*.md` naming
(`2026090X-tp4-mtp0-a3X-blockfp8-{result|result.md}`), attach the JSON receipt + hashes, and leave
every protected cell (`5.515783`, MTP1–4 grid, MTP3-4K `15.502`) untouched unless a Lane-3 cell
passes the full §5 promotion ladder.

---

## Appendix — key file:line citation index (no conversation knowledge assumed)

- Protected anchors & validity bar: S:57–69 · A28 buckets: [A28 JSON] l.66–114 (routed 26.084 l.67, dense raw 10.031 l.68/clean 7.575 l.94–103, quant_cast 4.185 l.69, gemm 532 l.108, fused_moe 96 l.107, allreduce 97 l.106; timing ineligible l.32) · MoE M1 w/o allgather: [A28 JSON] l.121–123,122–123 · XPU block kernel reg: GH k/linear/__init__.py@76cfe1cd88 l.450–451 · XPUFp8BlockScaledMMKernel apply_block_scaled_mm→fp8_gemm: GH xpu.py l.267–284 · act-quant default: GH BlockScaledMMLinearKernel.py l.47, l.119–123 · Triton MoE forced: N:preload-gates l.103–106,131 · EP4 tile shape: T:fullshape-triton-fp8-moe-gate.py l.198–217 · block-scale displacement + native-MoE deferral: N:preload-gates l.96–113 · W8A16 reroute op & effect: CB:patch_fp8_w8a16.py l.4–19 · small-M kernel contract: CB:apply_smallm_patch.py l.265–316 (M≤40 l.289) · speed gate precedents: n:a30 l.14–19, n:a27 l.15–17, n:a26 l.17–20 · causal-promotion: n:a31 l.50–55 · parity(149-digest): n:a16 l.24–31, n:a17 l.12–20 · exact-4K authority: [A28 JSON] l.28, n:a26 l.24–25 · wedge: S:46–49, CB:watchdog · kernel OOM floors: lane1 l.34–40, S:44–45 · collective-threshold neutral: n:bf16-allreduce-ll-threshold-neutral l.13–26 · WEB anchors: vllm#33214/#33379/#33973/#34117; vllm#39474 (op set); VXK v0.1.13 release; triton-lang/triton#7188; intel/auto-round#827; triton-lang block-scaled tutorial; IPEX float8 docs.

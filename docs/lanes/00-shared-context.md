# Flash-Next on 4×B70 — Shared Lane Context (all lanes)

Read this FIRST. It is the common ground for every lane spec in this directory.
Operator executes on the rig ~3h after issue (2026-09-01). Everything here is
traceable to /tmp lane evidence docs or the lab tree.

## Mission

Get `Qwen/Qwen3.8-Flash-Next-FP8` from **5.5 tok/s (eager MTP0) / 15.5 tok/s (MTP3 @ 4K)**
to serving speed on the existing 4× Arc Pro B70 box. Hardware is sufficient and FIXED.
**Do not spec more GPUs. Do not propose TP6** (2 KV heads → TP ∈ {2,4,8} only; current
topology TP4+EP4).

## Subject model (frozen identity)

- `Qwen/Qwen3.8-Flash-Next-FP8` @ `bcd9f01ddc9cff2316eb84281bebcd5b058bddce`
- 185.56 GB tree / 152,089 tensors; 125B-main / 6B-active MoE (E=128 routed experts)
- 48-layer hybrid **GDN + QSA** attention (QSA = Qwen Sparse Attention, sparse paged GQA
  + compression + lightning-indexer-style selection; 2 KV heads)
- **PLE**: 51B-token n-gram embedding table, host-placed via selective UVA
  (`cpu_offload_gb=12.25`, 12.22 GiB/rank × 4 ≈ 51.2 GB pinned host RAM). Synchronous
  pinned-UVA is ESTABLISHED; async UVA prefetch is NOT (A26/A27 rejected).
- 4B **MTP head ships in the checkpoint and is currently unused** (Lane 2's subject).

## Runtime identity (frozen — every receipt must pin these)

- vLLM source overlay: base `76cfe1cd88` + 18 production patches (0001–0010, 0012,
  0014–0018; 0011/0013 are opt-in diagnostics, NEVER on timing/qual trees) → tree
  `31ebb778…`, head `1372c62d975c554f4b465c8299bc5f3295301ceb`. Installed pip metadata
  LIES (`0.20.2rc1.dev2+…xpu` editable) — source overlay is authoritative.
- vllm-xpu-kernels: base `0fd18a7c` + certified 7-patch rebuild series → **loaded stage
  `2f829747`** (tree `d8c4318a…`). Checkout head `ad25aa9f` was NOT the loaded stage —
  do not substitute. Kernel rebuilds can need 120+ GB RSS per compiler process.
- Python 3.12.13, torch 2.11.0+xpu, triton-xpu 3.7.0, transformers 5.10.2, oneAPI
  2025.3.2, oneCCL 2021.17.2. NOTE: source commit's own requirements want torch 2.13.0 /
  triton 3.7.2 — dependency status is `dependency-observed`, NOT `dependency-installable`.
  A clean `pip install -r requirements` WILL drift. Do not "helpfully" upgrade.
- Runtime stage tar (18-file hybrid, SHA `6bf1b547…`) publicly hosted in
  `steveseguin/b70-optimization-lab` prerelease `qwen38-flash-next-runtime-2f829747-20260827`
  with unauthenticated-readback proof.

## Host prerequisites (every lane)

- 64 GiB swapfile + ~40 GiB root floor when graphs or kernel builds run
  (watchdog supervisors a5–a11 in the lab enforce mem floors + kernel-journal classifiers).
- **Xe2 Level-Zero wedge**: dies every 2–6h under load. Watchdog MANDATORY (cookbook
  `watchdog/` or lab supervisors). Kernel signature: `Engine reset: engine_class=ccs|bcs`,
  `Fault response: Unsuccessful`, `guc_exec_queue_timedout_job`. Container/engine restart
  recovers; in-flight requests lost.
- Current rig state: blocked on an **attended reboot** after the 2026-08-31 event-chain
  component test device-lost all four queues. Assume fresh boot at execution time.
- Repeated NVMe corrected-link events perennially trip clean-host gates — check kernel
  journal before blaming the runtime.

## Protected anchors (never overwrite, never re-run blind)

| Cell | Value | Status |
|---|---|---|
| MTP0 @ 512 ctx | 5.515783 tok/s | protected anchor (newer than README's 5.22) |
| MTP1/2/3 @ 512 | 9.372 / 11.895 / 14.889 tok/s | screens, monotonic, 26/26 frozen-baseline matched |
| MTP4 @ 512 | 20.727 tok/s (1716/1716 drafts) | screen |
| MTP3 @ exact-4K | 15.502 tok/s decode, TTFT 187.9 s, 799/852 drafts | PREFERRED 4K cell |
| PE 8K formal | 3.980 tok/s, TTFT 386.5 s | one-shot |

## Lab validity bar (non-negotiable, applies to every lane)

cache-zero · cold prompts · median-of-medians · 3-run repeat-hash · immutable receipts
(sha256 everything) · exact-token parity vs eager authority as SEPARATE gate from speed
(speed is not proof of parity — cookbook methodology) · preregister before measuring.

## Evidence map (read-only, on this machine)

- `/tmp/laneA_lab_map.md` — full lab inventory, who wrote what, lane statuses
- `/tmp/laneA_dsv4_lane.md` — DeepSeek V4 Flash K160 80.8 tok/s record (same box): the
  existence proof for graphs+MTP+FP8 at speed, with full negative-results list
- `/tmp/laneB_flashnext_scaffold.md` — Flash-Next lab state: 654 files, MTP grid,
  quarantine list, 18-patch series annotated, UVA PLE mechanics
- `/tmp/laneB_cookbook.md` — SergiioB cookbook assessment (≤2×B70 production recipes)
- `/tmp/laneB_qwen27b.md` — 27B lanes + Intel failure-mode catalogue (10 documented modes)
- `/tmp/b70lab/` — the lab repo itself (notes/, data/, tools/, results/, patches/)
- `/tmp/b70cookbook/` — cookbook repo (patches/, research/quantization-format-strategy.md,
  watchdog/, docs/RELIABILITY-REPORT.md)
- `/tmp/qsa_ops.py` — upstream QSA Triton kernels extracted from merged PR #53896
  (weight-free QSA path; logits workspace 128 MB chunked; narrow-tiles-favor-decode note)
- Lane specs as they land: `/tmp/flashnext-lanes/lane{1,2,3,4}-*.md`

## Key cross-lane facts

1. **Graphs**: quarantined-negative for Flash-Next (attempts a1–a7, 2026-08-28). DSv4
   proved target PIECEWISE + breakable draft PIECEWISE + exact-M capture is worth ~5×
   on this box (15.5 → 80.8 class jump came mostly from graphs+spec+FP8 together).
   DSv4 graph-killers to check for: D2H scalar reads during capture
   (`combined_lens.max().item()` class), `sycl_ext_oneapi_work_group_scratch_memory`
   ops (cannot enter SYCL graphs), padded-vs-exact-M capture descriptors.
2. **MTP**: drafting already works (MTP1–4 screens monotonic). Blockers are
   qualification-class (parity identity drift, engine stability, harness gates), not
   "does drafting work". DSv4 got exact-M=7 capture; padded-M8 lost 6% .
3. **FP8**: Flash-Next runs FP8 GEMMs on fallbacks. Cookbook's #1 named gap. DSv4 got
   80.8 WITHOUT native FP8×FP8 — W8A16-class paths + MXFP4 grouped MoE sufficed.
   Cookbook prior art: `patch_fp8_w8a16.py` (+34% at p1024 on 27B), small-M P1–P6
   BF16×FP8 kernel (b70-architect, 2026-08-29, needs certified-series rebuild).
4. **TTFT**: 187.9 s at 4K (MTP3) / 386.5 s at 8K. GLM-5.3 precedent on DGX Sparks:
   `MAX_NUM_BATCHED_TOKENS=2048` needed; 4096 measured worse; 8192 crashed the indexer.
   QSA indexer transient is the suspected cause on Flash-Next.
5. **Who to escalate to**: Steve Seguin (`steveseguin`) owns the lab, the gates, the
   receipts — actively mid-campaign on this hardware. peakcrosser7 authored merged
   vLLM PR #53896 (model + Triton QSA kernels). SergiioB = cookbook. mndodd = llama.cpp
   SYCL. bosd = TP2 field reports.

## Definition of done (per lane spec)

A rig operator with 3 hours and this repo can: verify identity → run receipts in
dependency order → collect sha256 receipts → declare each hypothesis/gate
PASS/FAIL with numbers. Anything a receipt needs must be in the spec or explicitly
listed as a prerequisite fetch with its URL/hash.

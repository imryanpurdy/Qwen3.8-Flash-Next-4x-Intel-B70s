# Deliverable 2 — Flash-Next FP8 on 4× B70: scaffold check + full state assessment

## 2A. The "empty scaffold" claim is FALSE

`/tmp/b70lab/experiments/qwen38-flash-next-fp8-b70/` is **not empty** — it is 654 files of the lab's most heavily instrumented active lane (a strictly preregistration-driven experiment, densely evidenced from 2026-08-26 through 2026-08-31):

| Subdir | Files |
|---|---|
| configs/ | 2 (MoE tuned-config JSONs: `moe-warps8-m1` & `moe-warps8-m4`, E=128,N=640, fp8_w8a8, block [128,128]) |
| data/ | 157 (per-attempt structured results JSON + sha256 evidence links) |
| fixtures/ | 3 (fixed-vision fixture + 16K semantic suite + its generator) |
| notes/ | 199 (per-attempt preregistrations and result notes, one per attempt) |
| tools/ | 293 (launcher/run/supervise/watch/test scripts + gates) |

**What was being attempted**: run the official `Qwen/Qwen3.8-Flash-Next-FP8` checkpoint (bcd9f01, 185.56 GB, 51B PLE n-gram table, 125B-main/6B-active MoE, 48-layer hybrid GDN+QSA) on **four Arc Pro B70s (TP4+EP4)** with vLLM XPU — a day-zero architecture with no upstream XPU support (`NotImplementedError` on XPU; official PLE offload NVIDIA-only). Every attempt is preregistered with frozen identity (commit hashes, cache bytes, ports), a strict gate contract, and a Grade (A–D) classification. Current status (latest notes, 2026-08-31): **active, mid-campaign, NOT stalled** — but blocked on an attended reboot after the 2026-08-31 event-chain component test device-lost the four queues. Protected results unchanged.

**Newest notes** (2026-08-31): A30 `hc-grouped-m1` endpoint negative (5.415455 vs protected 5.515783 tok/s, −1.82% → fails speed gate + exact-4K authority divergence at token 18/2; Samsung NVMe corrected-link events behind conservative gates); A31 M1-only (eight-warp MoE) prereg — frozen, blocked on reboot; event-chain A1 runtime negative (rejected combined clone-elision + same-queue XCCL event chain: device-lost on all four ranks at first measured sync); EP4 partition/device interaction closed as transient (A2 replicate: 0/4 cycles above 5%); EP4-vs-noEP component (no-EP 7.845% faster but repeats fail byte-parity and 15% screen); count2560 CPU-affinity/event-chain component work in progress. Current protected anchors: MTP0 `5.515783 tok/s` (a newer value than README's 5.22), MTP4 ≈ `20.727 tok/s`.

**Newest tools** (a5–a11): resource-watchdog supervisors for piecewise-graph and vision attempts — 64 GiB swapfile, root floor 40 GiB, mem floors, kernel-journal classifiers — i.e. running servers required tight host-memory supervision on a 128 GiB-class host while 4× 32 GiB cards + 12.22 GiB/rank host placement + graphs competed for RAM.

**Where it "stalled"**: the 8K repeated-serving boundary and all graph/vision/deeper-context arms are quarantined negatives; event-chain test poisoned the current boot. Low-per-token tok/s values (5–20) are the nature of this lane: single-sequence, 173 GiB model, GDN/Triton MoE fallback, eager — decode is ~3.8–5.5 tok/s target-only and MTP lifts it to ~9–21 tok/s, with TTFTs of 100–1000 s at 4K–16K.

## 2B. Results packet — complete state (/tmp/b70lab/results/qwen38-flash-next-fp8-b70/)

**Status: research screen; NOT deployment-qualified** (both README.md and HANDOFF.md, last updated 2026-08-28; the 56 KB README and 35 KB HANDOFF were read fully).

### What works (bounded, Grade-C)
- **First healthy TP4/EP4 server = attempt 19** (text-only, eager/graph-off, MTP0, 512 ctx, auto-KV, prefix cache off): p146/o256/c1 measured **5.14/5.22/5.29 tok/s, median 5.22**; 3/3 same output hash. This is a narrow research screen, explicitly not a conventional 99-interval headline.
- **Configured-512 MTP1–4 grid complete** (each matched 26/26 frozen MTP0 baseline comparisons, 16/16 repeat hashes, cache-zero, exact target hash on all rows): MTP1 **9.372** tok/s (503/505 drafts), MTP2 **11.895** (770/770), MTP3 **14.889** (768/768), MTP4 **20.727** (1,716/1,716). Monotonic within each cell's rows → screens, not ceilings.
- **Exact-4K classification**: MTP0 5.234 (legacy-comparable) / 5.234 current anchor; MTP1 8.904 (headroom32); MTP2 9.893 (32-block, superseding a 21-block quarantine); **MTP3 preferred: 15.502 tok/s decode, TTFT 187.9 s, wall 1.246 tok/s** (799/852 drafts); MTP4 quarantined. Formal p4096/o128 fixture gates all passed (e.g. MTP3 4.670 tok/s conventional).
- **Quality profiles**: baseline direct-answer strict 5/7 → corrected semantic 6/7 (only `30` vs `14` code miss); official-thinking sampler profile passed **25/25** (4/4 scout + 21/21 three-seed), separated reasoning/final fields, `14` correct in thinking. MTP3 thinking transfer: 19/19 correct completed answers but **unqualified** (repeated-session stop → API 500).
- **PE 8K formal cell**: exact p8192/o128 @ 3.980 tok/s, TTFT 386.5 s, one-shot (runtime stopped during 3rd comparison).
- MTP0 1K screen (4.449 realistic median; 5.134 after-first-text) and 2K repeat-v2 (5.228) pass.

### Quarantined (Grade D — no speed/quality/deployment credit)
Full 25/25 classification of the practical TP4 eager-text ≤8K slice: **12 screened, 13 quarantined**; live contract 240 text + 30 vision cells → 7–9 screened, 15 quarantined, 1 statically closed, rest missing.
- **Active-8K**: MTP1 (divergence at token 72; 4.151 tok/s diag), MTP2 (token 26; 6.235), MTP4 (token 26; 4.026, both card engine resets during teardown) — all "scoped cross-runtime parity quarantines" (different vLLM commit/cache than MTP0 authority); MTP3 — no receipt, 900 s bound.
- **Active-16K**: MTP0 A3/A4 = nondeterministic stability boundary (first 16,213-token request correct, identical same-server repeat corrupted to repeated `duct`; fresh server timed out at 1,600 computed tokens, sampling-RPC; 4-card engine resets). A generic p16384/o128 did complete (1,098 s) but Grade-D. MTP2 scheduler treatment exhausted (stopped at 5,440/9,216 tokens twice → 8 card resets, 58 unsuccessful responses). 24K/32K = only deterministic Grade-D legacy estimates with 50–150 % bands.
- **MTP1 active-1K** (stopped at 768 computed, no output, 300 s worker deadline), **MTP1/2K, MTP4/2K** (0-byte body / sampling timeout), **MTP2 active-1K** (perfect 2/2 exact requests but 11 post-cutoff corrected NVMe records → clean-host gate), **MTP3 active-1K** (external SIGTERM), **MTP4 active-1K** (2 exact requests 13.33/17.29 but supervisor failed teardown gate), **MTP4 exact-4K** (stalled at 3,904, engine resets). MTP2 exact-16K scheduler-64/32 negatives.
- **Piecewise-graph attempts a1–a7 (2026-08-28)**: graph (PIECEWISE, capture [1], 64 GiB swap) never qualified — repeated compile/host-memory/runtime-conflict quarantines; graph remains NOT established. Vision attempts a1–a11: all administrative closeouts/prereg stops — max-len 512 vision one-image arm never produced a qualified result; **vision NOT established** (30-cell vision contract all missing).

### What is NOT established (explicitly)
Production recipe, quality-qualified MTP speed, stable repeated serving at 8K or 16K, usable 24K/32K, vision, graph mode, fresh-server/clean-host replay, TP1/TP2 (TP1/EP1 statically closed on this host: text-only MTP0 exceeds host+one-card capacity; remaining TP1/TP2 need a new memory design), and deployment qualification. TTFT reduction + fresh-boot stability named as the next production problems.

### Exact runtime identity (authoritative Git hashes, not pip metadata)
- **Model**: `Qwen/Qwen3.8-Flash-Next-FP8` @ `bcd9f01ddc9cff2316eb84281bebcd5b058bddce`; tree SHA `4a3793bd…0f590eb2`; 144 files / 131 shards / 185,563,783,127 bytes / 152,089 tensors.
- **vLLM**: current source `1372c62d975c554f4b465c8299bc5f3295301ceb` (tree `31ebb778…`); legacy MTP0 authority used `658965050f…3771cd644`. Base for patches: `76cfe1cd88d30d525eec8be5bff75f8b77471c88`. Installed metadata lies (`0.20.2rc1.dev2+…xpu` editable) — source overlay is authoritative.
- **XPU kernels**: source checkout head `ad25aa9f69a2171612b9c6b83dfa82c69559f9e4`; **the staged runtime actually loaded** for measurements was built at `2f829747503c77d4814834dffd0840fb1dd9f75a` (rebuildable to tree `d8c4318a…` from base `0fd18a7c08…` via the certified 7-patch series). `ad25aa9f` was NOT the loaded stage — do not substitute.
- **Patches**: vLLM — base `76cfe1cd` + patches `0001–0010, 0012, 0014–0018` in order (**18 patches**); `0011` and `0013` are opt-in diagnostics, never on a timing/qual tree (see list below). All 33 patch files (0001–0033) exist under `/tmp/b70lab/patches/qwen38-flash-next-fp8-b70/vllm/`; the extra `0019–0033` are post-series candidates (QSA determinism, repeatability traces, PLE shard-validation, async-UVA prefetch, HC-grouped-up) — NOT in the production series. XPU kernels — **5 patches** (0001–0005) on base `0fd18a7c`; the certified reconstruction series is 7 patches in `vllm-xpu-kernels-certified-2f829747/` (0001 restore arch-probe bindings, 0002 restore local MoE prologue source, 0003 include fused-quant TUs, 0004 link core memory info, 0005 restore registered core implementations, 0006 link core host registration to Level Zero, 0007 ignore padding sentinels during alignment).
- **Host/toolchain** (from dependency-contract): Python **3.12.13**, torch **2.11.0+xpu**, triton-xpu **3.7.0**, transformers **5.10.2**, oneAPI/Intel runtime **2025.3.2**, oneCCL **2021.17.2** (⚠ observed post-measurement; conflicts with the source commit's own requirements — torch 2.13.0, triton 3.7.2+xpu — so a clean `pip install -r requirements` would drift; dependency status = `dependency-observed`, NOT `dependency-installable`).

The 18 production vLLM patches, what each does:
1. **0001** Merge of the model-support PR head `02f2b4c15d…` (qwen4_exp architecture) into base main (884 KB — the model registration itself).
2. **0002** Support PLE Offload for Qwen3.8 Flash-Next (the 51B n-gram embedding offload machinery).
3. **0003** Support the eager PLE-offload transport on XPU (the XPU-side worker/copy path).
4. **0004** Enable Qwen4Exp model dispatch on XPU (route the model onto the XPU platform).
5. **0005** Add Qwen4Exp XPU hyperconnection fallbacks (hybrid-attention HyperConnection ops).
6. **0006** Enable Qwen4Exp QSA (Qwen Sparse Attention: sparse paged GQA/index/compression) kernels on XPU.
7. **0007** Fix PLE target-device selection across accelerators.
8. **0008** Restore weight-skip filters for Qwen4Exp (loader must skip the host-placed weights).
9. **0009** Port QSA compressed cache to the tokens-per-state cache API.
10. **0010** Avoid copying uninitialized PLE weights during offload.
11. (diag) opt-in XPU MoE phase-sync trace.
12. **0012** Allow selective UVA offload of Qwen4Exp embeddings ★ (the key memory-layout patch: host-resident GPU-addressable PLE/embedding weights).
13. (diag) capture routed-MoE replay inputs on demand.
14. **0014** Normalize QSA caches from logical layout (GGUF/checkpoint cache-layout normalization for XPU).
15. **0015** Support the legacy XPU GDN ABI for target decode (the fused GDN attention path kept alive).
16. **0016** Fail closed on XPU GDN schema mismatches (spec-group metadata guard).
17. **0017** Port Qwen4Exp MTP tests to the tokens-per-state cache API.
18. **0018** Route legacy XPU GDN speculative decode (the MTP draft path through the legacy GDN kernel).

Optional sources behind candidates (not in 18): XPU-kernel 0006 (skip unused 512-expert top-k workspace), 0007 (grouped-gemm build contracts), 0008 (TP4 event-chain bridge); oneCCL `0001-Add-Qwen-count2560-event-chain.patch`; greedy-decision-trace patch; patches 0019–0033.

### Selective UVA placement for PLE n-gram embedding — how it works & does it work?
- **Mechanism**: patch **0012** (selective UVA offload) plus 0002/0003/0010/0028/0029. The accepted TP4 launcher does NOT use `VLLM_PLE_CPU_OFFLOAD=1` (the official NVIDIA `PleOffloadLayer` worker path, which is NVIDIA-only and refactored into `amd/ple_layer.py` on XPU). Instead the generic offloader pins **`ple_embedding.ngram_embedding.weight`** (the 51B n-gram lookup table, TP-sharded) **and** `embed_tokens.weight` into **host system RAM** via **UVA (Unified Virtual Addressing) — `cpu_offload_gb=12.25`**, making them GPU-addressable without a copy. Each of the four ranks reports **12.22 GiB placed** (13,117,911,040 bytes); 4× ≈ 51.2 GB total in pinned host RAM. Generation does **not** stream from the external USB/NVMe checkpoint.
- **Does it work?** **Yes — it is the centerpiece that makes 4×B70 fit and it is now the preferred placement.** Iterations: original 12.22 GiB (PLE+input-embedding) → PLE-only refinement A9 (2026-08-29) keeping only `ngram_embedding.weight` host-side (input embedding back on device, KV cut to 128 MiB): exact 51.200 GB (12,800,061,440 B/rank = 11.92 GiB receipts) and became the preferred MTP0 placement candidate (+4.55 % short median, +10.82 % exact-4K vs protected). The more ambitious **async** UVA PLE prefetch (overlap host lookup with layer 0, `VLLM_XPU_PLE_UVA_PREFETCH=1`) passed exact component gates (100-repeat hash parity vs CPU oracle) but its full endpoint arm A26 (and A27 MoE-warps8) was **rejected** at the exact-4K repeat/authority gate — so async overlap is NOT established; synchronous pinned-UVA is. Trace work localized a fresh-start variability first difference to layer 1 (the sole PLE-bearing layer) in A21/A25, but the PLE arithmetic itself was exonerated by A24/A25 fixed-input gates. Caveat in the audit: this is a valid RAM-resident UVA deployment, but it must not be described as proving the official PLE prefetch/overlap; also repeated NVMe corrected-link events perennially trip the lab's clean-host gates.

## 2C. Repro package (/tmp/b70lab/repro/qwen38-flash-next-fp8-tp4-mtp3-b70/)
Contains: README.md, RELEASE-NOTES.md, `runtime-contract.json`, `dependency-contract.json`, `model-contract.json`, `wheelhouse-contract.json`, `requirements-runtime.lock` (177-entry), `pip-freeze-observed.txt`, `publication-readback.json`, `runtime-stage.sha256`, plus scripts `prepare-sources.py`, `prepare-runtime.py`, `prepare-dependencies.py`, `verify-model.py`, `preflight.sh`, `run-server.sh`, `quality.sh`, `stop.sh`, and unit tests. Status: **`research-status` / runtime hosted; deliberately NOT a runnable guide** — the four launch scripts intentionally exit nonzero until gates pass.

- **runtime-contract.json**: the exact **18-file hybrid runtime stage** (only `_moe_C.abi3.so` freshly rebuilt at kernel commit 2f829747; the other 17 files — `_C.abi3.so`, `_xpu_C.abi3.so` 248.7 MB, `flash_attn_interface.py`, `fused_moe_interface.py`, `libattn_kernels_xe_2.so` 1.66 GB, `libgdn_attn_kernels_xe_2.so`, `libgrouped_gemm_xe_2.so`, `libmqa_logits_kernels_xe_2.so`, etc. — retained from the prior loadable stage). Uncompressed tar 1,968,250,880 B, SHA-256 `6bf1b547…`, split into two GitHub-release parts (1 GB + 894.5 MB) with sizes + SHA-256 per part + per-file hashes/sizes. Publicly hosted in the `steveseguin/b70-optimization-lab` prerelease `qwen38-flash-next-runtime-2f829747-20260827`; `publication-readback.json` proves an **unauthenticated public readback** (curl) installed all 18 files and every hash matched (installer receipt SHA `9ce34cf0…`).
- **dependency-contract.json**: Python environment observed post-measurement (`observed_runtime`: torch 2.11.0+xpu, triton-xpu 3.7.0, transformers 5.10.2, oneAPI 2025.3.2, oneCCL 2021.17.2, py 3.12.13); import precedence = kernel stage → vLLM source overlay → site-packages; **closure not complete**: `hash_addressed_binary_lock_complete: false`, `portable_dependency_install_authorized: false`, wheelhouse `wheelhouse-unavailable` (0 verified wheels) because the measured versions conflict with the source commit's own requirements. The installer refuses to run until each of the 177 lock lines has a verified wheel+hash.
- **model-contract.json**: repo/revision/144 files/131 shards/tree hash + historical verification receipt.
- **How to reproduce**: ① `verify-model.py` on the 185 GB tree; ② `prepare-sources.py` reconstructs vLLM tree `31ebb778…` = base 76cfe1cd + 18 patches, kernel tree `d8c4318a…` = base 0fd18a7c + certified 7-patch series (both replayed 2026-08-27); ③ `prepare-runtime.py` downloads the 2 parts and installs the 18 files into a fresh `$KERNEL_STAGE` (SHA-rechecked, fail-closed); ④ dependencies still blocked — must complete the hash-addressed wheelhouse first, then a fresh CPU-only offline install; ⑤ only then do the preflight/run/quality/stop gates become enabled. TP4+EP4, eager, MTP3, max-len 4352, 25 cache blocks (`294195200` B), BLHNC, PLE/input-embedding UVA. Historical reference numbers (must not be presented as a runnable recipe's promises): formal p4096/o128 4.670 tok/s @ TTFT 266.08 s; three p4096/o256 15.502 tok/s median after first text, wall 1.246 tok/s, TTFT 187.90 s.
- **Test suite**: `test_prepare_dependencies.py` / `test_prepare_runtime.py` cover fail-closed offline-install, observed-only refusal, part-hash/reassembly/traversal/member-mismatch cases (CPU-only, no GPU).

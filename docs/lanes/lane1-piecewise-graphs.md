# Lane 1 — PIECEWISE graphs on Qwen3.8-Flash-Next (TP4+EP4, 4× Arc Pro B70 32GB)

**Spec**: rig-ready remediation + qualification plan. **Written**: 2026-09-01 (UTC). **Executor**: operator, ~3 h after handoff.
**Lab**: `/tmp/b70lab` (read-only evidence of record). All citations are `path:line`. `[INF]` = my inference, not lab text.
**Web**: 2026-09-01 search. Generic upstream docs (vLLM CUDA-graphs `https://docs.vllm.ai/en/stable/design/cuda_graphs/`) describe PIECEWISE = per-mode graphs ("longest to capture"); NO authoritative XPU-graph-specific upstream fact was found by search — XPU-graph behavior below is lab-local or operator-briefed, not web-sourced.

---

## 1. Status line (protected anchors, all eager, graph OFF)

| Quantity | Value | Source (path:line) |
|---|---|---|
| Model | `Qwen/Qwen3.8-Flash-Next-FP8` @ `bcd9f01ddc9cff2316eb84281bebcd5b058bddce`, 185.56 GB tree | scaffold §2B:46 |
| vLLM (production series) | `1372c62d975c554f4b465c8299bc5f3295301ceb`, base `76cfe1cd` + 18 patches (0001–0010,0012,0014–0018; 0011/0013 diag-only) | scaffold §2B:47,49,52–70 |
| Loaded XPU-kernel stage | `2f829747503c77d4814834dffd0840fb1dd9f75a` (build head; `ad25aa9f` MUST NOT be substituted) | scaffold §2B:48 |
| MTP0 eager decode (target-only median) | **5.515783 tok/s** | notes/2026-08-31-tp4-mtp0-a30-hc-grouped-m1-endpoint-negative.md:17 |
| MTP3 eager exact-4K decode median | **15.501565106 tok/s** (rows 16.578976110/15.501565106/14.615697889, 3× salted p4096/o256/c1, no warmups); TTFT median 187.899186 s; wall 1.246260 tok/s | results README.md:631–634; data/20260827-tp4-mtp3-4352-attempt1-result.json (service_screen) |
| MTP3-4K eager output hash (authority) | `5f40744644b98ddd58a0c202fe855af324c0b1c33e1a6275afd74c12488f89f0` (3/3 rows) | data/20260827-tp4-mtp3-4352-attempt1-result.json |
| MTP4 eager @512 decode median | 20.72717637199404 tok/s | results README.md:833 |
| MTP0 exact-4K eager | 4.757818102 tok/s conventional | results README.md:38–39 |
| Graph state | **QUARANTINED-NEGATIVE**; "graph … qualification" NOT established; all a1–a7 (2026-08-28) Grade-D | scaffold §2B:40; results README.md:43 |
| Host | 128 GiB RAM [op-briefed; lab: pre-load MemAvailable ≈115 GiB, recovery plateau ≈103 GiB], PLE UVA pinned 12.22 GiB/rank (13,117,911,040 B) | a11 recovery result.md:14; scaffold §2B:75; a5 supervisor:223 |

**Boot state (hard prerequisite)**: the current boot is NOT eligible for GPU work. The 2026-08-31 event-chain A1 test device-lost all four queues at the first measured sync; tiny single-card compute then stalled on device 0; reboot authorized as the next device-lane action, not yet performed. (notes/2026-08-31-tp4-count2560-event-chain-a1-runtime-negative.md:37–41; notes/2026-08-28-post-a11-host-memory-recovery-result.md:16.) A31 (MTP1) and the graph successor are frozen behind that reboot (scaffold §2A:17).

---

## 2. Evidence inventory — what a1–a7 actually hit

All 7 = Grade-D, zero graph/quality/speed credit. **No attempt ever reached graph capture or produced a model-request.** The "compile / host-memory / runtime-conflict" trio in the scaffold maps exactly to the rows below.

| Attempt | Quarantine class | Failure detail (exact strings/numbers) | Line |
|---|---|---|---|
| a1 | **Host-memory (global OOM), compile phase** | Post-load graph compilation; rank 0 logged compile-range cache event `(1, 64)`; kernel reported "only 252 kB free from 8,388,604 kB total swap" (10:21:35); OOM selections killed `wireplumber`, `pipewire-pulse`, then `PID 1852126, worker TP3` (`VllmWorker-3` dead → executor shutdown), then `PID 1852043, worker TP2` (196 kB swap free). API never healthy; zero requests. Explicitly "does not by itself prove a leak… not a clean-host run." | attempt1-result.md:7–20, 24–38 |
| a2 | **Administrative admission (harness)** | 64 GiB swapfile created correctly but `swapon` reported priority `-1`; frozen packet required `-2`; `swapon(8)` accepts only `-1..32767` → supervisor failed closed before model start. | attempt2-result.md:5–11 |
| a3 | **Harness-path** | Derived supervisor staged in isolated resource dir; wrapper/client not staged beside it; first wrapper hash gate failed before GPU/preflight/model. | attempt3-result.md:5–11 |
| a4 | **Environmental (journal policy)** | 78/131 shards loaded; frozen broad journal gate tripped on an **APEI-corrected, non-fatal** receiver event from root NVMe `0000:01:00.0` (11:20:28); two storage-shape errors after are shutdown fallout. Temp swap peaked 6,077,232 KiB; no OOM, no B70 event. | attempt4-result.md:5–10, 14–15 |
| a5 | **Host-memory (global OOM), compile phase — with 64 GiB swap** | Passed ext4 staging + narrowed journal policy; all 131 shards × 4 ranks (31.27 GiB model-load mem, 86.9–87.0 s); rank 0 Dynamo transform 8.20 s, `(1, 64)` compile-range cache event 11:43:16. Watchdog 240 samples/256 s: `MemAvailable` 115,055,792 → 9,070,068 KiB; temp-swap peak 6,400,744 KiB; **12-GiB floor latched 11:44:20, one second before first OOM (11:44:21)**. First wave killed desktop-audio; second (12:56:35) killed TP2/TP3. Journal: 9× OOM + 9 killed, 42 corrected APEI + 43 RxErr (root NVMe `0000:01:00.0` + `0000:00:03.1`); **no B70, no fatal PCIe**. Verdict: "A further swap-only retry is not justified: any successor needs a material way to reduce compilation host-memory pressure and a monotonic wall-clock shutdown deadline… bounded under swap thrash." | attempt5-result.md:7–28, 49–53 |
| a6 | **Runtime-conflict scan race (harness)** | Schema-v2 structured scan reached PID 1922600 after it exited; first identity read `/proc/1922600/stat` → **ENOENT** → `rc=2`; 506 processes scanned, 0 conflicts, 1 read error; launcher exited before vLLM entry. `TORCHINDUCTOR_COMPILE_THREADS=1` treatment never reached. Cleanup exact; watchdog min MemAvailable 114.6 GiB. | attempt6-result.md:6–16, 30–34 |
| a7 | **Host-memory (floor trip), compile phase — compile_threads=1** | Passed schema-v3; reached treatment; all 131 shards (84.84–84.89 s); Dynamo 8.09 s (14:22:04), `(1, 64)` cache event 14:22:25. Phase guard latched 14:23:27 when MemAvailable fell **9,025,608 KiB in one sample to 25,947,668 KiB** (below the 30-GiB phase floor); TERM 14:23:29; first page-allocation + TTM + global-OOM 14:23:31; TP3 killed 14:23:32; KILL 14:23:40. Journal: **1 page-allocation failure, 18 TTM buffer-eviction failures, 9 OOM, 9 killed**; 1 allowed corrected root-NVMe; temp-swap peak 6,034,112 KiB (≥57 GiB of swap FREE at OOM `[INF: 67108860−6030436 KiB]`). | attempt7-result.md:6–26 |

**Cross-attempt reads (the RCA core):**
- Every compile-phase death (a1, a5, a7) is **host-RAM pressure during post-load Dynamo/Inductor compile + JIT on all 4 ranks**, *before* any capture. Zero evidence exists about capture-region correctness (`[INF]` follows from "no healthy API" in all 7).
- TTM buffer-eviction + page-allocation failures + ≥57 GiB of free swap at OOM ⇒ the dying allocations are **non-swappable / non-sleepable** (xe TTM, pinned), not gross fragile-swap exhaustion `[INF: pattern in a5/a7 counts is lab text; the "non-swappable" attribution is my inference]`. More swap is therefore NOT a material successor design — the a5/a7 closeouts independently say exactly that.
- a6's procfs race was fixed by schema-v3; a7 passed it — the runtime-conflict class is closed.
- The post-A11 xe reload could not restore host memory above ~103 GiB (floor 105 GiB) → recovery gate failed → reboot (a11 recovery result.md:6–16).

---

## 3. Ranked root-cause hypotheses

### H1 — Host-RAM exhaustion during compile/capture (incl. driver TTM pressure) — **SUPPORTED**
**For**: a1/a5/a7 all died in compile with near-identical timeline (load OK → Dynamo ~8 s → `(1,64)` marker → memory cliff → OOM). MemAvailable fell 106 GiB in <100 s (a5: 115.06→9.07 GiB). Watchdogs a5–a11 exist because of exactly this (supervise/watch scripts, mem/swap/root/PSI floors, classify kernel journal — a5 supervisor:253–461, watchdog:64–107, classifier:111–148). Unique vs DSv4 (which graphed the same 4×B70 box successfully): **+48.9 GiB of pinned, GPU-addressable UVA** (12.22 GiB/rank) plus 4× 31.27 GiB device model load and near-full 32 GiB cards leave no host slack for 4× Inductor/JIT; DSv4's 96 GiB model had no equivalent host-resident PLE. Difference in host load is exactly the DSv4/Flash-Next discriminator `[INF]`.
**Against**: notes explicitly decline to prove a leak or attribute every byte (attempt1-result.md:35–38; attempt7-result.md:31–33). 8 GiB→64 GiB swap did not change the failure class (moved the floor trip earlier via the 30-GiB phase guard, a7) — consistent with non-swappable demand.
**Verdict**: primary blocker. Any successor must reduce host-memory pressure materially (see §5 P1). ~57 GiB free swap at OOM is a falsifiable check for any "needs more swap" arm.

### H2 — GDN path can't be captured (needs DSv4's breakable/exact-M treatment) — **PLAUSIBLE, unproven**
**For**: anchor prereg records vLLM warning "XPU graph support is experimental and currently supports only single-GPU execution" (anchor-prereg.md:12–13); GDN is a recurrent, stateful attention with in-order-queue requirements (`gdn_attention_interface.cpp` TORCH_CHECK for in-order queue, kernel patch 0005); vLLM patches 0015/0016/0018 keep a *legacy* GDN ABI alive (scaffold §2B:67–70) — a sign the native path was fragile on XPU.
**Against**: the strongest counter-proof is the DSv4 record: **TP4 PIECEWISE graphs at 80.8 tok/s on this exact box** (results/README.md:30–46; laneA §1), plus PIECEWISE-only capture predating that was correct (xpu-graph-recovery.md:45–48). PIECEWISE never captures collectives (DSv4 launcher: piecewise, no forced comm capture — run.sh:26–27). Recurrent state is not inherently capture-hostile (DSv4 indexer state replays exactly). No GDN-capture error string exists **in evidence** because capture never ran — "not in evidence" for any specific GDN failure.
**Verdict**: the mode/ABI risk is real but entirely unobserved; the exact-M generalization already exists as uncertified kernel patch `0005-Generalize-exact-GDN-replay-to-MTP-row-count.patch` (fixes `total_spec_tokens == 4` → `> 0`; the DSv4 exact-M-transfer analog) — carry it as P2.

### H3 — QSA / PLE host-side gather inside the capture region — **PLE: DISPROVEN as-stated. QSA: PARITY hazard, PLAUSIBLE, unobserved**
**For (QSA)**: the QSA indexer top-k at exact 4K emits strict winners via *atomic reservations* and tie-breaks by reservation position — 100 identical launches produced **32 distinct hashes** (qsa-topk-diagnosis-and-a13-prereg.md:14–21). If QSA selection is in the captured region, replay parity at 4K is already broken in *eager*; a graph cannot fix it, and it is exactly the class of issue the validity bar must catch. DSv4's analog capture-freezer was a D2H scalar read `combined_lens.max().item()` (xpu-graph-recovery.md:17–23), repaired by fixed-width **device-only** packing + finite masked-chunk sentinel — reusable pattern (laneA §3a,b).
**Against (PLE)**: the current XPU PLE path is **device-side UVA over host-pinned RAM** — `vllm/models/qwen4_exp/amd/ple_layer.py` `Qwen4ExpNGramEmbedding` is a plain `nn.Module`, NOT the NVIDIA `PleOffloadLayer` cross-process host-worker (ple-deployment-audit.md:35–45; intake note:114–118 describes PR-53899's *host-process* lookup — a path the XPU port does not take). Host *side* gather is therefore not in the current capture region. No D2H-sync evidence exists for QSA/GDN either — "not in evidence" (capture never ran).
**Verdict**: PLE host-gather = disproven for the current tree (code-level re-confirm on rig is cheap). QSA determinism = must-fix **before** any exact-4K graph quality gate; candidate vLLM patch 0019 (stable descending argsort, masks visible-count rows, 5/5 suite, isolated 32-repeat exact-tie pass — qsa-topk:26–33) is the treatment.

### H4 — Triton-vs-SYCL-graph interaction (DSv4 scratch-memory class) — **DISPROVEN in evidence (PIECEWISE); residual risk noted**
**For**: DSv4's only native graph blocker was `fp8_paged_mqa_logits` using `sycl_ext_oneapi_work_group_scratch_memory` (unsupported in SYCL graphs) — FULL-only break; "PIECEWISE stayed correct" (xpu-graph-recovery.md:37–48).
**Against**: In DSv4, **Triton kernels are IN the record graph** (split-FP8 QK/LSE + PV — laneA §2). Grep of the entire Flash-Next kernel patch surface (`/tmp/b70lab/patches/qwen38-flash-next-fp8-b70/vllm-xpu-kernels{, -certified-2f829747}`) for `work_group_scratch|scratch_memory` → **zero hits**. Residual: grep covered *.patch text, not the full reconstructed kernel tree `[INF]`; and an op could still be reachable via JIT/AOT beyond the patched surface.
**Verdict**: not the known blocker. Cheap component ATP (P4) closes the residual.

---

## 4. Minimal-repro receipts

**Frozen identity (all receipts):** model `bcd9f01d…ddce` via `/mnt/fast-ai/llm-models/Qwen3.8-Flash-Next-FP8`; vLLM `1372c62d…1ceb` clean = base `76cfe1cd` + 18 patches; loaded kernel stage `2f829747…f75a` (17 retained files + freshly rebuilt `_moe_C.abi3.so`; runtime-contract `SHA-256 6bf1b547…`, parts hosted `steveseguin/b70-optimization-lab` prerelease `qwen38-flash-next-runtime-2f829747-20260827`) — scaffold §2C:81; TP4+EP4, `max_model_len 4352`, 1 seq, MBT 64, BLHNC, prefix-cache off, `--enforce-eager` ABSENT, no legacy graph controls; capture treatment `VLLM_XPU_ENABLE_XPU_GRAPH=1` + `--compilation-config {"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1}` + `--cudagraph-metrics` (anchor-prereg.md:40–51). Fresh port/paths/attempt per arm; temporary ext4 64 GiB swap `-p -1`, precreate floor 64 GiB + 40 GiB root, postcreate root/swap floors, watchdog per a5–a7 hash-pinned pattern; record **every**: MemAvailable min, temp-swap peak, `MemAvailable`-at-OOM, psi_full streak, TTM/page-allocation/OOM/killed counts from `journalctl -k` via the a5 classifier contract (classify-q38-piecewise-graph-a5-kernel-journal.py:111–148).

**Receipt R1 — H1: materially-reduced compile host pressure.** Treatment: (T1a) **pre-seeded Inductor compile cache** — one offline CPU-only pass per rank populates the inductor/compile cache directory *before* GPU launch (compile cache ≠ prompt cache; cache-zero gate refers to prompt/KV caching, scaffold-normal language); fall back to (T1b) serialized rank compile if the kitchen doesn't expose a shared-cache seed path `[INF: neither appears in lab notes; TORCHINDUCTOR_COMPILE_THREADS=1 alone failed (a7), so this is the material step a7 demanded]`. Keep `TORCHINDUCTOR_COMPILE_THREADS=1` + the a7 resource contract. Expected stdout: all 4 `Worker_TP{0..3}_EP3` load markers, single compile-range `(1, 64)` event, then healthy API. Record sha256 of launcher, wrapper, client, watchdog, both receipts (`identity.torchinductor_compile_threads=1`). **Pass**: capture completes, health OK, 12/30-GiB floors never trip, swap-free-at-OOM if any. **Fail**: any floor trip / OOM / TTM-eviction → record sample ─ do NOT retry blind under the same identity (a5:52–53, a7:58–59 mandate material change).

**Receipt R2 — H2: GDN capture-compat component (isolated, rig-only, no full server).** Apply uncertified kernel patch `0005-Generalize-exact-GDN-replay-to-MTP-row-count.patch` (from `vllm-xpu-kernels/`) to a scratch kernel tree — **not** the certified 2f829747 stage — record new tree hash. Driver: call `gdn_attention_spec_decode` under XPU-graph capture at `total_spec_tokens=1` then `=3` with `VLLM_XPU_GDN_SPEC_PERSISTENT_SCRATCH=1`, `VLLM_XPU_GDN_NATIVE_SPEC_RECURRENT_SERIAL_EXACT=1`, in-order queue satisfied (TORCH_CHECK in patch). Expected artifact: capture + **changed-length recurrent-state replay** (same pattern DSv4 required: `1073→437→1073`), i.e. exact token/hash set over 3 input lengths. Record op-build + capture sha256. **Pass**: both row counts capture and replay exact. **Fail**: op refuses capture → that string IS the first GDN capture error in evidence; log verbatim.

**Receipt R3 — H3: QSA determinism + index-pack audit.** (a) Code audit (read-only, rig): grep the Qwen4Exp XPU decode path (patches 0006/0009/0014 region + staged `_xpu_C.abi3.so`) for `\.item()|\.cpu()|\.max()|to(device|host)` / host-size scalar reads that would run only under capture — DSv4 precedent names the exact class (combined_lens.max().item(), xpu-graph-recovery.md:17–23). (b) Treatment: build vLLM at `1372c62d` + 18 + **0019** (deterministic QSA argsort, sha256 `df44c39f…9afcd` — qsa-topk:36–38). Expected: component gate = 100 identical QSA selections → **one hash** (vs current 32). Pass: 0019-focused XPU suite 5/5 and 32-repeat exact-tie 1-hash. Fail: >1 hash → P3 unresolved, no 4K arm.

**Receipt R4 — H4: op-by-op capture ATP (component).** Sweep the decode-step op set from a real eager trace (the a5–a7 identity server or the A25 64-record trace, notes/2026-08-30-tp4-mtp0-4352-ple-only-a25-…-result.md:8–14): each SYCL custom op + Triton kernel used in one decode step, captured individually under `VLLM_XPU_ENABLE_XPU_GRAPH=1` with changed-input replay. Expected stdout: per-op `CAPTURED`/replay-hash list; fail-closed on any op refusing capture (look for the DSv4 scratch-memory signature or Level-Zero command-graph rejection). **Pass**: 100% capture + exact replay. **Fail**: record the offending op name verbatim (may feed a FULL-mode narrow break, per DSv4 `436298dcd` precedent — xpu-graph-recovery.md:40–43).

---

## 5. Remediation work plan (dependency-ordered)

1. **P0 — Attended reboot (hard start gate).** After-reboot qualification per post-A11 recovery prereg: four fresh BDF/UUID mappings, MemAvailable ≥ 110,100,480 KiB (105 GiB) over 3 passive samples (0/60/300 s) without >256 MiB drift, PSI zero, idle ~43 MiB/card, one copy/compute `2097152.0` on 4/4, peer-access pass, 4-rank XCCL all-reduce `allreduce ok 4.0`, known-good TP4/EP4 eager MTP0 exact-`OK` canary (a11-recovery-prereg.md:36–48). No GPU arm before this.
2. **P1 — H1 (top-ranked)**: R1 (T1a then T1b). Dependency-gated on P0 only. Success = healthy capture.
3. **P2 — H2 (component, may overlap P1 on a separate slot)**: R2 kernel-patch 0005 exact-M GDN capture. Dependency: none (component). On pass, the exact-M transfer for the MTP draft is proven; on fail, log the capture blocker.
4. **P3 — H3 (component + code)**: R3 audit + 0019 build. Dependency: none. **Mandatory before any exact-4K MTP3 arm** (parity would otherwise be gated by the eager-phase 32-hash nondeterminism).
5. **P4 — H4 (component)**: R4 ATP. Dependency: none. On fail, carry the single narrow-break candidate into P6.
6. **P5 — First healthy full-model PIECEWISE arm**: MTP0 **first** (lowest capture surface; the a1–a7 cell shape, but with P1 material change + P2–P4 verdicts applied), then **MTP3 exact-4K graph arm** under §6 identity. Cache for the MTP3 arm = **294,195,200 B (25 blocks)** to match the 15.501565106 authority (README.md:615–616); the a1–a7 201,326,592-B cache (anchor-prereg.md:29) applies to MTP0 arms only.
7. **P6 — DSv4 transfer queue (only after the first healthy graph)**, each a separately preregistered treatment with its own hashes:
   - **Exact-M capture**: kernel 0005 enables `total_spec_tokens=3` exact capture vs padded M=4 — DSv4 lost **4.1 tok/s** padding M7→M8 (exact-M7 64.661 vs padded-M8 60.518; dspark-piecewise-exact-m7-record.md:60–78). Measure both on Flash-Next MTP3.
   - **Breakable draft PIECEWISE**: DSv4's "private breakable PIECEWISE" for the draft (record note:60–63); Flash-Next analog = MTP draft through legacy GDN (vLLM 0018) wrapped in the same breakable runner. CompilationMode.NONE on record lane because breakable graphs disable Dynamo/Inductor (run.sh:26–27 + xpu-graph-recovery.md:218–219).
   - **Device-only fixed-width index packing + finite masked-chunk sentinel** (0ed5ecc5 pattern, xpu-graph-recovery.md:25–28): apply to any QSA pack found capture-hostile in R3.
   - **oneCCL size-routed wide-epoch identity** under its own prereg: `B70_ONECCL_SYCL_ALLREDUCE_MAX_BYTES=131072` + `ONECCL_FORCE_PRELOAD=1` + build `48fda4f0…c6c12` (DSv4 identity) — graph capture is where oneCCL readiness defects historically surface (deterministic wrong arithmetic, `1113` vs `1073`; graph captures 28 & 58; bugs-failed-paths.md:16–18; laneA §4.6). Requires qualification on Flash-Next identity before use.
   - **Do NOT** port DSv4's M=1-occupancy or raw Level-Zero replay paths (closed negatives, bugs-failed-paths.md:7–11); not this lane.

---

## 6. Acceptance gates (falsifiable)

**Baseline (eager authority)**: MTP3 exact-4K decode median **15.501565106 tok/s**, TTFT **187.899186 s**, output hash **`5f40744644b98ddd58a0c202fe855af324c0b1c33e1a6275afd74c12488f89f0`** (3× p4096/o256/c1, no harness warmups, cache-zero) — data/20260827-tp4-mtp3-4352-attempt1-result.json.

**Target — qualify iff ALL hold on a fresh server, MTP3 arm**: ① runtime evidence of PIECEWISE capture (`CUDAGraphMode.PIECEWISE` metric row + compile-cache event) with NO fallback to NONE (anchor prereg:48–51); ② **exact-token parity**: every graph row output_sha256 == authority hash (all 9 rows across 3 servers); ③ 16/16 fixed-repeat single-hash + exact 4096-needle + 24/24 ordered quality requests (MTP3-4K authority bar, README.md:622–625); ④ cache-zero: zero cached + created-cache tokens in every row; ⑤ no paging: zero `pswpin`/`pswpout` delta and no temp-swap growth in measured rows (attempt7-prereg:116–118); ⑥ watchdog journal clean (no OOM/killed/TTM-eviction/page-allocation, no B70 xe reset/fault/timeout, only allowed corrected root-NVMe) (classifier contract; a5 watchdog:83–96); ⑦ **speed**: median-of-medians over 3 fresh graph servers × 3 salted rows ≥ **X = +50%** → ≥ **23.25 tok/s**.
- **X justification**: same-box DSv4 graph-hop = +229–245% (2.616→8.616 diag; 10.32→35.53 strict-suite, xpu-graph-recovery.md:66–70, kv-repeatability:59–66); Flash-Next cell noise ≈ ±7% (16.58/15.50/14.62 rows) and endpoint negatives at −0.26% / −1.82% vs protected anchors. +50% ≈ 1/6–1/5 of the demonstrated same-host graph ceiling and >3× the observed noise band — filters warm-cache/JIT artifacts while remaining a real claim. Graph runs that meet ①–⑥ but land 0–49% are a **bounded screen** (reportable as parity-qualified, zero speed credit — the DSv4 lane likewise holds non-record rows as screens, laneA §4.5).
- **Repeat-hash rule**: 3 independent fresh graph servers (fresh ports/paths/identity each), each 3 salted p4096/o256/c1 rows; all 9 hashes must equal authority; per-server median; median-of-medians vs 23.25 tok/s.

**Failure condition**: after P1–P5 with at most 2 materially-distinct compile designs (T1a,T1b) and max 2 full-model arm runs total — if no run reaches healthy capture within the 7,500-s supervisor bound (a5 supervisor:49,393–395) or fails ②/④/⑥, the lane remains quarantined; only a genuinely new design (not a swap/floor retune) may appear next. A single qualified failure of ② (any row hash ≠ authority) also fails the arm closed with the full evidence manifest preserved.

---

## 7. What NOT to do

- **Do not re-run any a1–a7 config blind**: identical retries are explicitly rejected by the closing notes (a5: "further swap-only retry is not justified", attempt5-result.md:52–53; a7: "an identical retry is not justified", attempt7-result.md:58–59). The compile range `(1,64)`, MBT 64, capture `[1]` shape may be kept; the *host-memory plan* must change.
- **No admission-floor lowering**: a11 recovery forbids lowering floors to hide retained state (a11-recovery-prereg.md:10) — carry the 12 GiB absolute / 30 GiB phase / 40 GiB root / PSI-full<30 gates forward.
- **Do not touch protected results**: MTP0 `5.515783`, MTP3-4K `15.501565106`, MTP4 `20.727176372`, MTP1/MTP2 rows, every legacy value under its own identity (each a7/a5 closeout re-affirms).
- **Keep identity rigid**: loaded kernel stage **2f829747** (never `ad25aa9f`), vLLM source overlay authoritative (pip metadata lies), 18-patch production series (0011/0013 diag never on timing trees) — scaffold §2B:47–50.
- **Do not run on the poisoned boot**; reboot is P0 and non-negotiable (event-chain device-lost, §1).
- **Watchdog is mandatory end-to-end** — the Xe2 Level-Zero wedge (operator-briefed: kills the job every 2–6 h under load) plus the proven compile-phase OOM class make an unattended graph capture a host-kill risk (a1 killed desktop services). Reuse the hash-pinned a5–a7 supervise/watch/classify stack (source supervisor `414dd8ad…`, watchdog contract, classifier `440d7d06…`); monotonic TERM→KILL escalation; preserve temp swap if teardown fails (swapoff only after exact-layout proof).
- **Do not treat the DSv4 40.136 pre-repair row as authority** and do not adopt the wide-epoch oneCCL build on the Flash-Next tree without its own prereg (path `48fda4f0…`, binary sha256 `53de2b6d…`, `.so` quality-qualified for DSv4 identity only).
- **No LocalMaxxing / deployment / site claims** off a graph screen: this lane can, at best, move PIECEWISE out of "not established"; deeper-context (16K+) and deploy-qualification remain out of scope (scaffold §2B:43).

---

## 8. Evidence provenance note
All lab numbers above were read from the paths cited (attempt results a1–a7, watchdog tool sources, results README/HANDOFF, DSv4 record/notes, qsa/PLE/ple-audit notes, a11 recovery). Anything not in those files is marked `[INF]` or "not in evidence" — including: any GDN/QSA capture error string (none exists; capture never ran), the 2–6 h wedge cadence (operator-stated constraint, not lab text), and the "+50% ceiling extrapolation" (derived from DSv4 numbers). If a cited file's content conflicts with the rig, the file wins; update this spec and re-qualify before proceeding.

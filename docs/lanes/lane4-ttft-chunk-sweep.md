# LANE 4 — TTFT attack spec: chunk-size sweep + prefill bucket profile + fixed-overhead decomposition

**Qwen3.8-Flash-Next-FP8 @ `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` on 4× Intel Arc Pro B70 (TP4+EP4, eager, MTP3-preferred).**
Spec-writer: Lane-4 research subagent · 2026-09-01 UTC · research artifact — the operator executes on the rig (~3 h window, assumed fresh attended boot per shared context). **Spec only: no rig changes, no code execution against hardware by the writer.**

> **Read-first.** Evidence pack `E:` = `/tmp/flashnext-lanes/evidence-lane4-ttft.md` (compiled 2026-09-01, every claim verified against the lab tree — spot-checks confirmed; deltas noted at `[VERIFY]`). Shared context `C:` = `/tmp/flashnext-lanes/00-shared-context.md`. Every number below is cited to the lab repo under `/tmp/b70lab` unless marked `DERIVED` (arithmetic on cited inputs, no new measured facts) or `INFER` (hypothesis, explicitly flagged).
>
> **Citation keys.** `R:` = `results/qwen38-flash-next-fp8-b70/README.md` · `N:<note>` = `experiments/qwen38-flash-next-fp8-b70/notes/<note>.md` · `D:<json>` = `experiments/qwen38-flash-next-fp8-b70/data/<json>` · `T:<tool>` = `experiments/qwen38-flash-next-fp8-b70/tools/<tool>` · `Q:` = `/tmp/qsa_ops.py` (upstream QSA Triton kernels, merged PR #53896) · `LANE1/LANE2` = sibling lane specs in `/tmp/flashnext-lanes/`.

---

## 0. Status line — the untested axis and the two cost classes

TTFT at exact-4K is **187.9 s** (MTP3 best row) while short prefills measure ~10–12 s ⇒ there is **both** a large per-prompt-token cost **and** a large fixed per-request/per-step overhead [E:§3; R:222,252,280,633]. This lane attacks both with three preregistered experiments: **(E1)** the first-ever prefill-phase device-bucket profile (extend the A28 machinery), **(E2)** a sweep of the never-tested `max_num_batched_tokens` knob, **(E3)** a functional decomposition `TTFT = F + c·prompt` that decides which cost class the next treatment must target.

**Smoking gun (verified).** Every production Flash-Next launcher freezes `max_num_batched_tokens=64`:
- `T:launch-tp4-ep4-eager-mtp0-512.sh:354,392,449,484` — `max_num_seqs=1, max_num_batched_tokens=64` (:354), Python `assert ... == 64` (:392), identity printf (:449), CLI `--max-num-batched-tokens 64` (:484). This is the base launcher the MTP3-4K cell actually runs through [D:20260827-tp4-mtp3-4352-attempt1-result.json identity vllm_head `1372c62d…`, kv_cache_memory_bytes 294195200; T:launch-tp4-ep4-eager-mtp3-4352.sh:26–30]. Base sha recorded: `62b40c9268…` (see §5.3 for the recorded `[VERIFY]` full value).
- Same freeze in `T:launch-tp4-ep4-eager-mtp1-long-context-base.sh:346,384,441,476` and the vision/sampler clients `T:run-tp4-mtp0-current-vision-a8-client.sh:112,114` (`max_num_batched_tokens=64`), `T:run-tp4-mtp2-16512-scheduler32-client-a2.sh:84–85` (the one exception, frozen at **32**).
- A 4096-prompt prefill at 64-token chunks = **4096/64 = 64 serialized full-model scheduler steps** [DERIVED], each carrying fixed per-step host overhead (schedulal step, PLE UVA gather, per-layer collectives at tiny token counts, kernel-launch serialization). A28 confirms the p4096 request is chunked into exactly 64 prefill engine iterations [N:2026-08-30-tp4-mtp0-a28-target-step-profile-prereg.md:34–39].

**No chunk-size sweep exists anywhere in lab evidence** — a `max_num_batched_tokens` search across `experiments/qwen38-flash-next-fp8-b70/tools/` and `notes/` returns only the frozen 64 and 32 values (verified by grep for `max_num_batched_tokens|max-num-batched-tokens`). This is an **untested axis, not a failed one** [E:§1].

**GLM-5.3 precedent (context-cited; NOT in the lab tree — `INFER` transfer).** On a DGX-Spark box with the same QSA indexer class: `MAX_NUM_BATCHED_TOKENS=2048` was required, 4096 measured worse, 8192 crashed the indexer [C:§4; E:§1]. `[VERIFY]` a tree-wide search for `GLM-5|MAX_NUM_BATCHED_TOKENS=2048|Spark.*2048` in `/tmp/b70lab` returned no hit — treat as operator-briefed context, not lab text. Expect the sweep's optimum to be **non-monotonic and gated on indexer stability**.

**Operator prerequisites (all lanes, non-negotiable):** attended fresh boot after the 2026-08-31 event-chain device-lost; post-boot four-rank XCCL collective + generation canary; watchdog mandatory end-to-end; 64 GiB swapfile + ≥40 GiB root floors; clean-host NVMe journal policy [C:§Host-prerequisites; LANE1 §5 P0; LANE1 §7]. No GPU work before those pass.

---

## 1. Anchors table — every TTFT number cited

| Cell (fixture) | TTFT (s) | Companion numbers | Source |
|---|---|---|---|
| **MTP3 exact-4K** (p4096/o256/c1 ×3, no warmups) | **median 187.899186**; rows 184.027160 / 189.004167 / 187.899186 | decode median 15.501565106 tok/s (aft-1st-text); wall 1.246260 tok/s; 799/852 drafts (93.78%); **parity hash `5f40744644…f89f0` (3/3 rows)**; fixture (needle) sha `1583fb31…` | D:20260827-tp4-mtp3-4352-attempt1-result.json `service_screen/ttft_s`, `median_ttft_s`, `median_decode_tok_s_after_first_text`, `output_sha256`, `quality/needle_sha256`; R:613–645, R:631–634 |
| MTP3 exact-4K formal (p4096/o128) | 266.080895 (formal row) | hash `6949154fec506375c9cb8e5f4b52df25a057734ae7ffa01c3e21784f85b1cbbd` | D:same `formal_exact_depth/ttft_s`, `output_token_ids_sha256`; R:625–627 |
| **MTP1 exact-4K** | formal 317.104665 (p4096/o128) · service median 232.079233 | decode 8.904420575; 528/539 drafts; | R:303–307 |
| MTP0 exact-4K | formal 217.909692 (p4096/o128) · legacy-comparable median 123.391275 (p4096/o256) · two-row conventional 149.330 / 145.607 | decode 4.456026475 formal / 5.233664732 legacy | R:700–704; R:81–83 |
| 512-ctx screens (p146/o256; cache-zero needle at **317 actual prompt tokens**) | MTP2 **11.278097** · MTP3 **11.817638** · MTP4 **10.023315** | — | R:222; R:249–252; R:277–280; needle length R:243 |
| 1K (MTP0 screen, p1024/o256) | median **29.043115** | decode 5.133587561 (3× p1024/o256) | R:654–656 |
| 2K (diag-only, quarantined) | MTP3@2K **150.769910** (p2048/o128) · MTP2@2K **310.712871** | parity-quarantined, zero credit | R:463–467; R:515–519 |
| 8K (MTP0 formal, p8192/o128) | **386.534332** | decode 3.979729240; runtime stopped before 3rd comparison | R:709–727, R:716–717 |
| Decode-cycle cost (reference) | — | MTP0 5.515783 tok/s ⇒ **≈181.3 ms/cycle** [DERIVED: 1000/5.515783]; A28 noncollective bucket sum ≈ **46.1 ms/cycle** [DERIVED from D:a28 `device_buckets_mean_ms…`], i.e. collectives/host gaps dominate decode too | N:a30:17,43; D:20260830-tp4-mtp0-a28-target-step-profile-result.json:66–79, 122 |

**Spread warning (read before comparing anything):** the same MTP0/MTP1 config family measures **123–317 s TTFT at 4K across boots and sources** (MTP0 legacy 123.4 vs formal 217.9; MTP1 232–317; MTP3 184–266 across fixtures). The lab's own language: "the wall-rate and TTFT rows vary widely" and cross-source comparisons are "descriptive workload-aligned cross-run … not causal" [R:310–314, R:637–643]. **Every comparison inside this lane is within one boot family on one identity** (§3, §4); never treat a cross-boot TTFT delta as a treatment effect.

**A28 decode-phase reference profile (the only bucket-level timing in the lab).** Decode cycles, NOT prefill: routed_shared_moe 26.084 ms, dense_projection 10.031 raw / 7.575 clean, quantization_cast 4.185, elementwise 2.943, qsa 2.208, other 1.698, moe_router 0.789, gdn 0.409, device_mem_op 0.163, normalization 0.051, **ple 0.0019**; per-cycle **97 BF16 allreduces**, 532 GEMMs, 96 fused_moe, 86 qsa events, 12 qsa_sparse_splitk, 72 gdn (36 causal_conv + 36 gated_delta); `collective_allreduce_distorted 197.065` (Kineto-inflated, diagnostic only). Lab's own labels: `primary_bottleneck_class = collective_critical_path_and_cross_rank_arrival_imbalance`; `not_supported_as_primary_speed_targets = [GatedDeltaNet, QSA, PLE lookup]`. Identity: vLLM `d14396e2…`, stage `2f829747…`, MTP0, graph off, max_model_len 4352, ple_only_synchronous_uva; profile captured rank-0..3, gzip-valid, 4/4 rank tables, 3 retained decode contexts/rank, 5,586 device events/context/rank; **timing permanently ineligible for speed credit** [D:a28:1–129; N:a28-result:35–79].

**TTFT was never bucket-profiled.** A16/A17 were digest-stability traces: 51 boundary records + **149 exact tensor digests** (positions, model input, delayed-hyperconnection tuple after each of 48 layers, final outputs), report-only, no timing credit; A16 failed at exact-4K repeat row 2 (first diff at token 62, 66 positions differ) [N:2026-08-30-tp4-mtp0-4352-ple-only-a16-late-prefill-trace-result.md:9–20; D:20260830-tp4-mtp0-4352-ple-only-a16-late-prefill-trace.json].

**Related negatives already banked (do not re-run blind):** A26 async-UVA prefetch endpoint-negative (3 short rows 2.17% below the 5.515783-protected median; both exact-4K rows non-authority) [N:…a26-async-uva-endpoint-negative.md]; A27 moe-warps8 treatment inert + rejected at 4K repeat gate [N:…a27-moe-warps8-endpoint-negative.md]; A1 event-chain device-lost all 4 ranks (caused the current boot block) [E:§6]. A30 hc-grouped-m1 −1.82% speed-gate fail; A31 M1-only warps-8 MoE is FROZEN and overlaps Lane 3, not this lane [E:§6].

---

## 2. E1 — Prefill-phase bucket profile (report-only, NO speed credit)

**Objective.** First device-bucket map of a single 64-token prefill step at exact-4K — the unanswered question behind both E2 and E3: where do 187.9 s / 64 steps ≈ **2.94 s per prefill step** [DERIVED] actually go on device?

**Method — extend the A28 machinery, do not rebuild it.**
1. **Identity (frozen, clone of A28):** model `bcd9f01d…`, vLLM `d14396e27247c1b251da0ce24a0942772c4b002f` (NOTE: different head than the production `1372c62d` — E1 reuses A28's identity so prefill buckets compare 1:1 with A28's decode buckets; see *Provenance* caveat), kernel checkout `ad25aa9f…`, loaded stage `2f829747…`, TP4+EP4, eager/graph off, MTP0, max_model_len 4352, 128 MiB KV cache, prefix cache off, synchronous selective-UVA PLE only, default MoE config [N:a28-prereg:15–31].
2. **Profiler window — the only machinery change:** new wrapper derived from `T:vllm-serve-with-q38-a28-profiler.py` (sha `b2093aaf…`); change `profile_delay_iterations` **65 → 2**, keep `profile_max_iterations` = 4. A28's 65-start proves the counter counts engine iterations and that 64 of them are the p4096 prefill [N:a28-prereg:34–39] ⇒ a 2-start window captures **prefill chunk steps 2–5** (all inside prefill), not decode. CPU+XPU activities, shapes on, stack/memory/FLOP off, gzip on, frontend ignored — unchanged from A28 [N:a28-prereg:32–34]. Freeze the wrapper's sha256 before first use.
3. **Analyzer — reuse, do not rewrite the unit conversion:** copy `T:summarize-tp4-target-decode-kineto.py` (frozen corrected sha **`a4d4c54cce53e1a2c1c3795caf742e7216e8d646ed0ca002f517cf09bd4ab7b6`**, 5/5 tests, sha `b58b76dd…`) to `T:summarize-tp4-prefill-kineto.py`. **Only** two deltas: (a) the per-trace contract frozen to `DEFAULT_CONTEXT` (decode `execute_context_0(0)_generation_1(1)`) is relaxed to accept the observed prefill `execute_context` annotation — the tool already fail-closes on foreign annotation families, so the window's single-family discipline is preserved [T:summarize…py:228–265]; (b) the `classification` string becomes `qwen38_flash_next_tp4_prefill_kineto` [T:summarize…py:608]. **Keep byte-identical**: the base-time contract (`baseTimeNanoseconds` required, rejects missing/malformed) [py:79–96], the normalization **`(raw_anchor_ns − baseTimeNanoseconds) / 1000`** with fall-back chain `submitted → appended → sycl_enqk_begin` [py:98–113], `classify_device_event` bucket heuristics [py:141–216], `_collect_contexts` overlap/foreign checks [py:219–265], all summary math [py:380–491]. The frozen `a4d4c54c` is exactly the analyzer that fixed the unit bug that killed A28 — its predecessor `53620dc6` compared absolute-nanosecond anchors with Kineto-relative microseconds and failed the run [D:a28:46–56; N:a28-result:15–24]. **Do not re-derive that conversion.**
4. **Client/supervisor:** clone the A28 client/supervisor pattern (exactly 4 rank-qualified gzip traces + 4 rank tables, hashes bound in a manifest, fresh run/cache/compile/RPC/evidence roots, new port). Attempt = next free ordinal after A31 (A31 is frozen/handled per Lane 2 §6 — confirm ordinal in `AGENT_HANDOFF`; default **A32**, port 19732, profile dir `/mnt/fast-ai/q38-profiles/attempt32`).
5. **Frozen battery for E1 (reduced, report-only):** recovery canary pass; the profiled p4096/o128 request must pass the generic exact-depth and cache-zero transport gates (output hash recorded, authority **not** required for profile credit — A28's own profile row was byte-identical to a known non-authority family and still scored capture credit [D:a28:22–33]); 16/16 short repeat one-hash (cheap determinism read); teardown contract: supervisor rc 0, no listener/model/compile/RPC residue, 4 cards < 43 MiB, kernel-journal clean (no B70 reset/fault, only allowed corrected root-NVMe) [inherit A28 lifecycle rules: D:a28:131–144; N:a28-result:81–85].

**Deliverables.** Same schema as A28: per-bucket mean/median ms per retained prefill context × rank, per-cycle multiplicity (GEMMs / fused_moe / qsa_select + sparse_splitk / allreduces per prefill step), top device-event rows with shapes, cross-rank context-start skew, summed noncollective vs collective (distorted) share. **No timing/quality/speed/reliability credit under any outcome** — A28's interpretation block applies verbatim [D:a28:115–129].

**Decision pivot (preregistered reading).** (a) A noncollective bucket ≥ 2–5 ms/step earns a treatment-candidate rank [N:a28-prereg:68–69]; (b) a dominant collective/cross-rank class points at the same collective critical path A28 already named — feed E2/E3, not a new kernel bet [D:a28:122]; (c) a large `other_noncollective`/host-gap share points at host-side per-step work → confirms E3's F term; (d) **if `qsa` or `ple` is large in prefill** (they are tiny in decode: 2.208 / 0.0019 ms) — that is itself a finding: prefill rows run through the chunked 128-MiB logits-workspace loop with per-chunk device allocation [Q:751–790] and per-step PLE UVA lookups.

**Provenance caveat (`INFER`):** E1's profile is on vLLM `d14396e2`, while E2/E3 run production `1372c62d`. Bucket-level targets found in E1 must be re-verified against `1372c62d` before any treatment is built (cheap code audit, not a rerun).

---

## 3. E2 — Chunk-size sweep 64 → 512 → 2048 → 4096 (+8192 probe behind a crash gate)

**Purpose.** Test the one untested axis. Mechanical prediction [DERIVED]: 64-token chunks serialize the prefill into 64 full-model steps; 512/2048/4096 cut steps to 8/2/1 — a 8–64× reduction in per-step host overhead IF per-step cost dominates (E3's F). GLM-5.3 says the optimum is non-monotonic and an 8192-class indexer crash is a live risk [C:§4; E:§1].

**Frozen identity per cell (all 5 parts — any deviation voids the cell):**
- Model `Qwen/Qwen3.8-Flash-Next-FP8` rev `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` [C:§Subject]; vLLM `1372c62d975c554f4b465c8299bc5f3295301ceb` (base `76cfe1cd` + 18 production patches 0001–0010,0012,0014–0018; **0011/0013 never on timing trees**) [C:§Runtime-identity; LANE2 §0.1]; loaded kernel stage **`2f829747503c77d4814834dffd0840fb1dd9f75a`** (never `ad25aa9f`) [C:§Runtime-identity]; TP4+EP4 eager/graph-off MTP3, max_model_len 4352, **KV cache fixed 294,195,200 B (25 blocks)** — the MTP3-4K authority geometry — BLHNC, prefix-cache off, async off, `enable_thinking=false`, selective-UVA PLE 12.22 GiB/rank [D:20260827-tp4-mtp3-4352-attempt1-result.json identity; R:615–620; LANE2 §0.1].
- **ONE variable per cell:** `max_num_batched_tokens` ∈ {512, 2048, 4096}. Baseline cell **64 is the protected anchor — do NOT re-run**; it is sealed at TTFT 187.899186 s / hash `5f407446…` [D:20260827-tp4-mtp3-4352-attempt1-result.json; R:631–634].

**Launcher derivation (A28-style, freeze before use).** The 64-freeze lives in the runnable base `T:launch-tp4-ep4-eager-mtp0-512.sh` at :354/:392/:449/:484, which the MTP3-4K wrapper execs [T:launch-tp4-ep4-eager-mtp3-4352.sh:26–30]. Per cell: derive a new shell source replacing all four 64-sites with the cell value — including the Python scheduler assert (:392) and the identity.txt printf (:449) — wrap with MTP=3 / MAX_MODEL_LEN=4352 / KV_CACHE_MEMORY_BYTES=294195200 / fresh ATTEMPT+PORT, then `sha256sum` the wrapper AND the derived source and record both before first use (exact sed recipe in §5.3). Fresh run/cache/compile/RPC/evidence roots per cell. **One server boot per cell.**

**Fixture & parity (every cell identical):** exact-4K service-screen shape p4096/o256/c1, fixture byte-identical to the MTP3-4K cell (needle sha `1583fb31…` [D:20260827-tp4-mtp3-4352-attempt1-result.json `quality/needle_sha256`]); cache-zero (zero cached + created-cache tokens in every row); **3 salted rows** (salts `context-r1/r2/r3`), no harness warmups; per-row TTFT + output hash + decode median. **Parity is a SEPARATE gate from speed** (lab validity bar, [C:§Lab-validity]): every row hash must equal the accepted authority **`5f40744644b98ddd58a0c202fe855af324c0b1c33e1a6275afd74c12488f89f0`**; also run the 16/16 fixed-set repeat one-hash on the **2048 cell** as an internal determinism read. A cell that completes but fails parity is **quarantined as qsa-determinism-class** and receives zero speed credit (see §6 for why QSA ties must be checked first).

**Stability gate per cell (fail-closed):** kernel journal clean of Xe2 engine resets (`Engine reset: engine_class=ccs|bcs`, `Fault response: Unsuccessful`, `guc_exec_queue_timedout_job`) and of B70-addressed faults; no `RPC call to sample_tokens timed out`; no shared-broadcast wait splurges; no 300 s worker-deadline expiry; teardown rc 0; 4 cards < 43 MiB; no corrected-NVMe records in-window after the frozen cutoff (clean-host rule) [C:§Host; LANE2 §6].

**Indexer-crash watch:** server log + kernel journal swept for the GLM-8192-class signature (indexer fault / engine reset during prefill). **If ANY cell ≥ 2048 trips it, STOP the sweep** — publish the crash cell as direct evidence of the boundary; do not proceed to 8192.

**8192 probe (stability ONLY):** runs **only if the 4096 cell passed every stability gate AND completed with parity**. Single p8192/o128 row on a fresh boot; **NO speed credit, NO parity credit** (the 8K row is quarantined regardless [R:709–727; LANE2 §1.3]); purpose = pin the QSA workspace-chunk boundary predicted in §5.2.

**Expected shape (`INFER`, GLM-5.3 + step arithmetic):** TTFT should fall from 64 → 512 (step-count 64→8) if per-step overhead is large; the optimum is plausibly sub-max (2048), with 4096 regressing or crashing and 8192 tripping the crash gate — because wider rows raise per-step device cost and cross the `rows_per_chunk` workspace partition on the indexer path [Q:753–761]. **Non-monotonic results are the expected reading, not a defect.**

**Credit rule:** a cell earns comparison credit only if parity + stability + repeat-hash all pass; medians reported as screens (Grade-C), never as qualified speeds; any cell that looks like a winner requires the lab's median-of-medians 3-boot replay before promotion — outside this lane's 3 h budget.

**Budget:** 1 boot/cell ≈ 90–100 s load + 3×TTFT + gates ⇒ ~8–12 min/cell; 4 cells (512/2048/4096 + optional 8192) ≈ ≤ 1.5 h. E1 + E3 ≈ 45 min. Total fits the 3 h window with margin.

---

## 4. E3 — Fixed-overhead decomposition: TTFT = F + c·prompt

**Purpose.** A 2-parameter functional model over prompt length to answer: is 187.9 s mostly **fixed per-request/per-step overhead F** or **marginal cost c per prompt token**? This picks the next attack surface (host/step work vs device/token work) and reads E2's shape correctly.

**Method (one boot family — the authoritative fit):**
1. Same frozen production identity as E2 (vLLM `1372c62d`, stage `2f829747`, MTP3, max_len 4352, 25-block cache, MBT **64** — i.e. the production baseline config, so the fit describes what we currently ship).
2. Fixtures: **p317** (the exact 317-actual-token cache-zero needle [R:243], fixture content from the MTP3 configured-512 receipt `D:20260827-tp4-mtp3-512-attempt4-result.json`), **p1024**, **p2048**, **p4096** (exact needle `1583fb31…`). All o128, cache-zero (zero cached/created-cache tokens per row). One request per fixture, plus one **second p4096 row** for a within-boot variance band. All four + repeat on a single server boot.
3. **Fit (command in §5.3):** OLS `TTFT = F + c·P` on the four points; report F (s), c (s/token, and s/1k-token), R², per-point residuals. Decision rule:
   - **`F / (F + c·4096) ≥ 0.5` ⇒ F dominates ⇒ attack per-step host work** (scheduler step cost, PLE synchronous UVA gather per step, D2H scalar reads/syncs, kernel-launch serialization, oneCCL per-step init) — chunk size is then a *secondary* lever (it only moves the `(prompt/chunk) × per-step-overhead` term), and the truly productive surface is host/step elimination.
   - **c dominates ⇒ attack per-token device work** — the collective critical path (97 BF16 allreduces/token [D:a28:106,122]), MoE fallbacks, dense projections [D:a28:123] — chunk size will do little.
   - **Cross-check with E2:** a strong TTFT drop from 64 → 512 chunks is direct evidence the per-step term is large, whichever way F/c split numerically (the 2-parameter fit absorbs the per-step term into F if step count is constant, so read them together).
4. **Prior anchors (cross-boot/source — DIAGNOSTIC ONLY, never the fit):** {317-actual-token → ~10–12 s (R:222/252/280, includes decode tail) · 1024 → 29.04 s (R:656) · 2048 → 150.77 s diag (R:467) · 4096 → 187.90 s (R:633) · 8192 → 386.53 s (R:717)}. A rough line through these is consistent with a large intercept (~90–130 s [DERIVED, rough]) but is contaminated by the documented 123–317 s cross-boot spread — the single-boot fit in this section is the authoritative one.

**Output & credit.** JSON receipt with (P, TTFT, output-hash) rows, OLS result, residuals, and the **dominance verdict**. Zero speed credit (single-request rows are not median-of-medians); the verdict feeds E2's interpretation and future host-side receipts.

---

## 5. Receipts — workspace math, predicted QSA launch counts, exact operator commands

### 5.1 rows_per_chunk math (from the merged kernel source)
- `_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024` = 134,217,728 B → **33,554,432 int32 slots** [Q:15].
- `columns = page_table.shape[1] * k_cache.shape[1]` [Q:751]; the kernel consumes them as `PAGE_SIZE = k_cache.shape[1]` and `PAGE_TABLE_WIDTH = page_table.shape[1]` constexprs [Q:659–660].
- `rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns*4, 1))` = `max(1, 33_554_432 // columns)` [Q:753].
- Sequential host-computed chunk loop `for row_start in range(0, rows, rows_per_chunk)` with a `torch.empty` workspace alloc per chunk [Q:755–761]. **`rows` = the query-token count of the step** ⇒ in prefill, `rows` is the chunk size (64 today, 512/2048/4096 in the sweep), so wide prefill rows are what push the loop; a 8192 probe crosses it when `rows_per_chunk < 4096`.
- Kernel-level note: "Narrow tiles favor decode; wide tiles improve throughput for prefill" [Q:873] — authority that prefill geometry is a distinct, untuned regime.

### 5.2 Predicted QSA-select launch counts [DERIVED — arithmetic on §5.1]
Per prefill of P tokens at chunk C: steps = ceil(P/C). Per step, per layer, the QSA selection path [Q:742–790] runs with rows = C and issues **ceil(C / rows_per_chunk)** sequential workspace-chunk launches (plus a per-row expansion path [Q:704–706]). Model depth 48 [C:§Subject]. Total QSA-select launches = **48 × ceil(P/C) × ceil(C / rows_per_chunk)**.

| `columns` (read on rig) | `rows_per_chunk` | chunk C | steps @P=4096 | launches/layer | **launches total (48 lyr)** |
|---|---|---|---|---|---|
| 4096 | 8192 | 64 | 64 | 64 | **3,072** |
| 4096 | 8192 | 512 | 8 | 8 | **384** |
| 4096 | 8192 | 2048 | 2 | 2 | **96** |
| 4096 | 8192 | 4096 | 1 | 1 | **48** |
| 8192 | 4096 | 64 | 64 | 64 | **3,072** |
| 8192 | 4096 | 512 | 8 | 8 | **384** |
| 8192 | 4096 | 2048 | 2 | 2 | **96** |
| 8192 | 4096 | 4096 | 1 | 1 | **48** |
| 16384 | 2048 | 64 | 64 | 64 | **3,072** |
| 16384 | 2048 | 512 | 8 | 8 | **384** |
| 16384 | 2048 | 2048 | 2 | 2 | **96** |
| 16384 | 2048 | 4096 | 1 | **2** | **96** |
| 32768 | 1024 | 64 | 64 | 64 | **3,072** |
| 32768 | 1024 | 512 | 8 | 8 | **384** |
| 32768 | 1024 | 2048 | 2 | **4** | **192** |
| 32768 | 1024 | 4096 | 1 | **4** | **192** |

Reading: at every geometry the sweep cuts QSA-select serialization 8–64× (3,072 → 48–192 launches for the whole 4096-prompt prefill). **The chunked regime (ceil(C/rows_per_chunk) ≥ 2) begins once C > rows_per_chunk** — i.e. at C=4096 when `columns ≥ 16,384`, and at C=8192 (rows=4096) whenever `rows_per_chunk < 4096`. That transition is where the "8K/16K instability and indexer-crash class" plausibly lives (`INFER` — evidence pack §4's attribution; the exact `columns` value must be read on the rig, §5.3 step 4). `[VERIFY]` the real `columns` from the E2 server before trusting any single row of this table.

### 5.3 Exact operator commands (lab repo is the rig's `/tmp/b70lab`; freeze hashes first, fresh ports/roots per attempt)

**1) Identity verification + artifact hashes (run before any launch):**
```bash
cd /tmp/b70lab
sha256sum experiments/qwen38-flash-next-fp8-b70/tools/launch-tp4-ep4-eager-mtp0-512.sh \
          experiments/qwen38-flash-next-fp8-b70/tools/launch-tp4-ep4-eager-mtp3-4352.sh \
          experiments/qwen38-flash-next-fp8-b70/tools/summarize-tp4-target-decode-kineto.py
# expect: 62b40c9268… (base launcher) · 405f2b102f… (MTP3 wrapper) · a4d4c54cce… (corrected analyzer)
```

**2) Derive one E2 chunk-cell launcher per value V (V ∈ {512, 2048, 4096}), then freeze:**
```bash
V=2048
sed -e 's/max_num_seqs=1, max_num_batched_tokens=64/max_num_seqs=1, max_num_batched_tokens='"$V"'/' \
    -e 's/scheduler_config.max_num_batched_tokens == 64/scheduler_config.max_num_batched_tokens == '"$V"'/' \
    -e 's/max_num_batched_tokens=64\\n/max_num_batched_tokens='"$V"'\\n/' \
    -e 's/--max-num-batched-tokens 64/--max-num-batched-tokens '"$V"'/' \
    experiments/qwen38-flash-next-fp8-b70/tools/launch-tp4-ep4-eager-mtp0-512.sh \
  > tools/lane4-e2-mtp3-4352-mbt${V}.sh
sha256sum tools/lane4-e2-mtp3-4352-mbt${V}.sh        # RECORD before first use
# Preferred alternative: an A28-style derive wrapper (T:launch-tp4-mtp0-4352-ple-only-a28-profile.sh:12–36 pattern)
# that also injects MTP=3, MAX_MODEL_LEN=4352, KV_CACHE_MEMORY_BYTES=294195200, fresh ATTEMPT/PORT,
# and the identity.txt printf — so artifact provenance is reproducible, not byte-collaged.
```

**3) E1 prefill-profile analyzer invocation (on the copied, re-named analyzer):**
```bash
/home/steve/.venvs/vllm-xpu/bin/python \
  experiments/qwen38-flash-next-fp8-b70/tools/summarize-tp4-prefill-kineto.py \
  /mnt/fast-ai/q38-profiles/attempt32 \
  --context-name "<observed prefill execute_context from client-reported observed_execute_context_annotations>" \
  --drop-first 1 --expected-retained 3 \
  --output /mnt/usb-models/bench-results/qwen38-flash-next-fp8-b70/<…>/lane4-e1-prefill-profile.json
```

**4) Read the real QSA geometry on the rig (E2 server, rank 0) — feeds §5.2:**
```bash
# instrument the qsa select call once: print(page_table.shape, k_cache.shape)
# columns = page_table.shape[1] * k_cache.shape[1]
# rows_per_chunk = max(1, 33554432 // columns)     # = max(1, 134217728 // (columns*4))
```

**5) E3 linear fit (on the E3 receipt's rows):**
```bash
python3 - <<'PY'
import numpy as np
P = np.array([317, 1024, 2048, 4096])
T = np.array([…])          # TTFT s from the E3 receipt, same order
F, c = np.polyfit(P, T, 1)
share = F / (F + c * 4096)
print(f"TTFT = {F:.1f} + {c:.6f}·prompt   R² = {np.corrcoef(P, T)[0,1]**2:.3f}")
print(f"F share at 4K = {share:.2%} →", "F dominates → attack per-step host work"
      if share >= 0.5 else "c dominates → attack per-token device work")
PY
```

**6) Journal watches (mandatory every run):** xe engine-reset/wedge signatures `xe 0000:(23|27|43|47):00.0.*(reset|fault|timeout|fatal|wedged|failed)` + `guc_exec_queue_timedout_job`; NVMe corrected events `0000:01:00.0` (clean-host gate); memory floors (≥40 GiB root, 64 GiB swap) [C:§Host-prerequisites; LANE2 §6].

---

## 6. What NOT to do (hard constraints)

- **Do not re-run the 64 baseline cell** — the MTP3-4K anchor (187.899186 s, `5f407446…`, D:20260827-tp4-mtp3-4352-attempt1-result.json) is the sealed comparison point. The sweep starts at 512.
- **Do not compare TTFT across boots/sources as causal** — the documented 123–317 s spread at 4K makes any cross-boot delta uninterpretable [R:310–314, R:637–643]. Every E2/E3 delta is within one boot family on one identity.
- **Do not rewrite (or "improve") the A28 unit conversion** `(raw_anchor_ns − baseTimeNanoseconds)/1000` — the frozen analyzer `a4d4c54c` is the corrected one; its predecessor `53620dc6` compared absolute-ns vs relative-µs and failed the whole A28 run [D:a28:46–56]. Extension = copy + freeze, never re-derivation.
- **Do not read a failed E2 parity row as chunk-caused without first checking QSA determinism** — the exact-4K indexer's atomic-reservation tie-break produced 32 distinct hashes over 100 identical launches (400 strict winners / 624 cutoff ties) [N:2026-08-30-tp4-mtp0-qsa-topk-diagnosis-and-a13-prereg.md:8–20]; the deterministic-selection patch 0019 (sha `df44c39f…`) is a candidate outside this lane.
- **No 8192 probe unless the 4096 cell passed every stability gate**; the probe is stability-only — zero speed/parity credit.
- **No deadline-only retries anywhere** — raising `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` is explicitly unauthorized [LANE2 §6 · N:2026-08-27-tp4-mtp4-4352-context-prereg.md:120–122]; a stall is a blocker to quarantine with logs, never a knob to whack.
- **No GPU work before the attended fresh boot** + four-rank collective + canary; no work on poisoned boots [C:§Host-prerequisites; LANE1 §5 P0, §7].
- **Watchdog mandatory end-to-end**; memory floors enforced; `SYCL_CACHE_PERSISTENT=1` never set on B70 [LANE1 §7].
- **Do not stray identity**: loaded stage `2f829747` (never `ad25aa9f`), production 18-patch series only (0011/0013 diagnostics never on timing trees), dependency versions observed-not-installable [C:§Runtime-identity; LANE2 §0.1, §6].
- **Zero speed/quality/deployment credit from E1, the 8192 probe, quarantine rows, or any diagnostic rate**; screens are Grade-C at best until the lab's 3-boot median-of-medians replay [D:a28:115–120; LANE2 §6].
- **Do not re-run A16/A17/A26/A27 or the A1 event-chain test blind** — all negatives with named blockers [E:§6; N:a16-result; N:a26-negative; N:a27-negative].

---

## 7. Provenance & identity summary

- All lab numbers were read from the cited paths (verified during spec writing: launcher 64-freezes at the cited lines; A28 JSON fields incl. `offline_contract_failure`, `a4d4c54c…`, 5/5 tests, bucket tables; QSA kernel lines 15/753/761/873; README anchors R:222/252/280/304/307/467/498/519/633/656/701/704/717; A16/A26/A27 notes). Anything not in those files is marked `DERIVED` or `INFER`.
- **GLM-5.3 DGX-Spark precedent** is context-cited (C:§4, E:§1), not lab text — tree search found no source; treat as operator-briefed, `INFER` transfer to Flash-Next.
- **Fixed / frozen identity (all experiments):** model `bcd9f01…`; vLLM `1372c62d…` for E2/E3 and `d14396e2…` for E1 (A28-comparability choice, flagged); kernel stage `2f829747…` everywhere; TP4+EP4 eager, BLHNC, sync-UVA PLE 12.22 GiB/rank. E2/E3 additionally fix MTP3, max_len 4352, 25-block 294,195,200-B cache.
- If any cited file's content conflicts with the rig at execution time, **the file/rig wins** — update this spec and re-qualify before proceeding (LANE1 §8 pattern).

---

*End of Lane 4 spec. Deliverable: `/tmp/flashnext-lanes/lane4-ttft-chunk-sweep.md`. Writer executed no rig changes and no hardware code; all receipts are for the operator.*

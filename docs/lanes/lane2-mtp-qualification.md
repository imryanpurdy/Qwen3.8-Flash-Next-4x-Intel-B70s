# LANE 2 — MTP Speculation Qualification Spec (k-optimal + qualification blockers)

**Qwen3.8-Flash-Next-FP8 @ `bcd9f01` on 4× Arc Pro B70 (TP4+EP4) — rig-ready run plan for the lab operator (~3 h window).**
Spec-writer: Lane-2 research subagent · 2026-09-01 UTC · Research artifact — operator executes on the rig.

> **Citation keys.** `R:` = `results/qwen38-flash-next-fp8-b70/README.md` (read in full, 921 lines);
> `S:` = `/tmp/laneB_flashnext_scaffold.md`; `A:` = `/tmp/laneA_dsv4_lane.md`;
> `N:<note>` = `experiments/qwen38-flash-next-fp8-b70/notes/<note>.md`;
> `D:<json>` = `experiments/qwen38-flash-next-fp8-b70/data/<json>`.
> `DERIVED` = computed here from cited cumulative counters (no new measured facts added).
> `INFER` = hypothesis, explicitly marked — not lab evidence.

---

## 0. Status line

**Lane 2 is SCREENED, NOT QUALIFIED.** The MTP grid at 512 ctx is complete and every cell is a **SCREEN** (bounded, monotonic-within-cell, Grade C), not a ceiling. Exactly **one deployment-shaped MTP cell is qualified at 4K (MTP3); all active-8K MTP cells are quarantined; MTP4@4K is quarantined on an engine-stability stop; every 1K/2K MTP cell is quarantined** (harness, host, or runtime-parity blockers — none is a quality exclusion *except* the 8K acceptance decay, which is context-length, not k-causative). Positions argued for here, with falsifiable gates in §5:

- **4K serving target → k=3** (MTP3, qualified 15.502 tok/s decode) until MTP4@4K clears **median-of-medians ≥ 18.0 tok/s + exact-token parity + 20-request zero-reset battery**.
- **8K serving target → k=0** (MTP0 formal 3.979729 tok/s, screened) — no MTP row at 8K is qualified.
- **Acceptance RATE beats k** (DSv4 lesson, §3.3): MTP4 wins at 512 only because acceptance was 100%; at 4K it is 93.8% (MTP3), at 8K 52.4% (MTP4 diag). Deep-k is a bet on deep-position acceptance on a stack whose deep-position acceptance provably decays.

### 0.1 Frozen identity (all five parts — any deviation voids the run)
| Part | Identity | Source |
|---|---|---|
| Model | `Qwen/Qwen3.8-Flash-Next-FP8` rev `bcd9f01ddc9cff2316eb84281bebcd5b058bddce`; tree SHA `4a3793bd…0f590eb2`, 144 files / 131 shards / 185,563,783,127 B | S:46 |
| vLLM | `1372c62d975c554f4b465c8299bc5f3295301ceb` (= base `76cfe1cd…` + **18 patches 0001–0010,0012,0014–0018**; **0011/0013 are diagnostics, never on a timing tree**; 0019–0033 are post-series candidates) | S:47,49,52–70 |
| XPU kernels (LOADED) | stage built at **`2f829747503c77d4814834dffd0840fb1dd9f75a`** (rebuild: base `0fd18a7c…` + certified 7-patch series). **`ad25aa9f…` is the source head, NOT the loaded stage — do not substitute** | S:48–49 |
| Topology/config | TP4+EP4, allgather/reduce-scatter, eager, graph off, prefix cache off, async off, 1 seq, batched cap 64, BLHNC, `enable_thinking=false`, selective UVA PLE+input embed (12.22 GiB/rank; 13,117,911,040 B) | R:47–49,148; N:m4-4352prereg:26–33 |
| Host/toolchain | py 3.12.13, torch 2.11.0+xpu, triton-xpu 3.7.0, transformers 5.10.2, oneAPI 2025.3.2, oneCCL 2021.17.2 (*dependency-observed*, NOT installable — do not reinstall or drift) | S:50 |

**Identity note.** The newest MTP0 source checkout is `797769b34` (two descendants: grouped-HC dispatch), used only for A31 [N:a31:27–31]. **Lane-2 MTP cells must stay on `1372c62d`** so parity with the sealed MTP grid is meaningful. `Qwen4Exp` MTP support ships in patches **0017** (MTP tests on the tokens-per-state cache API) and **0018** (legacy XPU GDN speculative decode — the MTP draft path runs through the legacy GDN kernel) [S:69–70]; patch **0016** fail-closes on GDN schema mismatch [S:68]. All MTP cells here therefore exercise the legacy-GDN draft path — the engine-stability tickets in §1 live in that neighborhood.

---

## 1. Evidence inventory (per-k × per-context; every cell cited)

### 1.1 Configured-512 grid — COMPLETE, all SCREENS (26/26 MTP0 comparisons, 16/16 repeat one-hash, cache-zero needle, exact target hash)
| k | median tok/s (aft-1st-text) | drafts→accepted | per-position (DERIVED) | wall med / TTFT med | Source |
|---|---|---|---|---|---|
| 0 | **5.515783** (protected anchor; newer than README's 5.22) | — | — | — | N:a30:17,43 |
| 1 | **9.37225436776222** | 505→503 (99.60%) | 503/505 = 99.6% | 7.046 / (not in receipt) | D:m1-512-attempt3; R:201–207 |
| 2 | **11.895061402541456** | 770→770 (100%) | [385,385] = 100%×2 | 7.805 / 11.278 s | D:m2-512-attempt1; R:218–229 |
| 3 | **14.88878979448863** | 768→768 (100%) | [256,256,256] = 100%×3 | 9.011 / 11.818 s | D:m3-512-attempt4; R:249–262 |
| 4 | **20.72717637199404** | 1716→1716 (100%) | [429,429,429,429] = 100%×4 | 11.560 / 10.023 s | D:m4-512-attempt1; R:277–289 |

Marginals (tok/s added per k, DERIVED): k0→1 **+3.856** (+69.9%), k1→2 **+2.523** (+26.9%), k2→3 **+2.994** (+25.2%), **k3→4 +5.838 (+39.2%) — the DEEPEST position adds the MOST at 512**, because 100% acceptance means every extra position yields a verified token for near-zero extra compute. Screen rows span 12–33% of their medians, so none is a ceiling [R:256–259,280–283].

### 1.2 Exact-4K (p4096, deployment shape) — MTP3 PREFERRED; MTP4 QUARANTINED
| k | decode median (aft-1st-text) | drafts→accepted (acc%) | per-position (DERIVED) | TTFT / wall | Status | Source |
|---|---|---|---|---|---|---|
| 0 | 5.233664731906276 (legacy-comp.) / formal 4.456026475 | — | — | 123.391 / — | screened | R:691–707 |
| 1 | **8.904420575355882** | 528/539 (97.96%) | 528/539 = 98.0% | 232.079 / 0.981 | screened (headroom32) | D:m1-4352-h32; R:291–315 |
| 2 | **9.89315479235244** | 719/748 (96.12%) | [365/374=97.6%, 354/374=94.7%] | 263.279 / 0.891 | screened (headroom32; supersedes 21-block) | D:m2-4352-h32; R:530–562 |
| 3 | **15.50156510641242** | 799/852 (93.78%) | [275/284=96.8%, 268/284=94.4%, 256/284=90.1%] | 187.899 / 1.246260 | **PREFERRED (4K target)** | D:m3-4352-a1; R:613–645 |
| 4 | — (no valid row) | — | — | — | **QUARANTINED** — stalled 3,904/4,096 computed, HTTP 500, 300 s worker deadline, 4× card resets | D:m4-4352-a1-bn; N:m4-4352prereg:104–124; R:592–611 |

4K marginals (DERIVED): k0→1 **+3.671** (+70.1%), k1→2 **+0.989 (+11.1% — k=2 is a weak 4K cell)**, k2→3 **+5.608 (+56.7%)**. MTP3 dominates the 4K row; its formal 4K output hash is `6949154fec506375c9cb8e5f4b52df25a057734ae7ffa01c3e21784f85b1cbbd` [D:m3-4352-a1].

**MTP4@4K stall detail (the central blocker):** last scheduler state **3,904 computed / 64 scheduled, KV usage 0.9286, four spec slots**; 5× one-minute shared-broadcast wait messages (8 total in server log); engine fatal `TimeoutError: RPC call to sample_tokens timed out.`; helper wrote no partial JSON → zero credit; cleanup logged **compute/copy resets on all four B70s**; no post-reset collective [D:m4-4352-a1-bn; N:m4-4352prereg:104–124]. **Raising only `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` is explicitly NOT an authorized retry** [N:m4-4352prereg:120–122]. The **identical 3,904-computed signature** appears in the MTP2@4K 21-block first attempt (stopped during prefill, zero output, resets) — later **passed at 32 blocks** [R:564–590,530–562]. A recurring prefill-boundary stall near-full KV, cleared once by cache-geometry enlargement (INFER: cache/workspace pressure at the prefill→spec-decode boundary; both observations consistent).

### 1.3 Active-8K (p8192) — entire row quarantined
| k | diag tok/s (NO credit) | drafts→accepted | per-position (DERIVED) | divergence (0-based) | Blocker class | Source |
|---|---|---|---|---|---|---|
| 0 | 3.979729240 formal, TTFT 386.534 s (screened) | — | — | — | n/a | R:709–727 |
| 1 | 4.150862265053902 | 76→51 (67.1%) | 51/76 = 67.1% | **72** | cross-runtime parity | D:m1-8448; R:771–795 |
| 2 | 6.234518099077392 | 53 dr / 106 tok → 76 (71.7%) | [41/53=77.4%, 35/53=66.0%] | **26** | cross-runtime parity | D:m2-8448; R:729–747 |
| 3 | — (no receipt; 900 s bound, 0 output tokens) | — | — | — | engine stability (no-receipt) | D:m3-8448-bn; N:m3-8kprereg:111–138; R:749–769 |
| 4 | 4.02562915156999 | 41 dr / 164 tok → 86 (52.4%) | [30/41=73.2%, 24/41=58.5%, 18/41=43.9%, 14/41=34.1%] | **26** | cross-runtime parity (+8 teardown resets) | D:m4-8448; N:m4-8kprereg:46–61; R:797–829 |

Parity-authority asymmetry: the frozen MTP0 8K authority used **vLLM `658965050` + 192 MiB cache** (raw `2a8bfbb133…`, output `0efd150b868d…`) [N:m3-8kprereg:57–60]; every candidate ran `1372c62d` + larger fixed caches → all divergences are **scoped cross-runtime/cache parity quarantines, NOT proof of MTP-caused corruption** [R:738–742]. **Cross-validation:** MTP2@8K output `d3ce0631…` and MTP4@8K candidate `d3ce0631…` are **byte-identical** (each ≠ authority at index 26) [D:m2-8448 vs D:m4-8448 comparator]. Two different spec depths produced the same divergent string: the divergence is target-side (runtime/cache), k-independent (INFER: strongly supported).
**8K acceptance-collapse is real and context-mediated (INFER):** pos0 acceptance drops 98%→77%→73%→67% as context goes 4K→8K, and MTP4 pos3 falls to 34.1% — deep-position acceptance is the first thing to decay at long context.

### 1.4 Active-1K/2K family — six distinct quarantines
| Cell | Evidence | Blocker (named) | Class | Source |
|---|---|---|---|---|
| MTP1@1K | stopped at **768 computed, 0 output**, 300 s worker deadline; request 2 blocked | first-request worker-response stall | engine stability | R:317–341; D:m1-1536-bn |
| MTP1@2K | 2xx HTTP but **0-byte body**; engine sampling-RPC timeout; 448 computed/64 scheduled, counters zero; teardown resets ×8 | sampling-RPC timeout | engine stability | R:343–370; D:m1-3072-bn |
| MTP2@1K | **2/2 exact** (10.682699 / 12.641866; 170/170 accepted, perfect pos0+1); **11 corrected NVMe records AFTER frozen cutoff** | clean-host gate (storage link) | host gate | R:485–508; D:m2-1536-host-q |
| MTP2@2K | completed; divergence from MTP0 auth at token **12** (4.526752827 diag) | cross-runtime parity | runtime identity drift | R:510–528; D:m2-3072-parity-q |
| MTP3@1K | external **SIGTERM** at 00:05:06 during request 1 (6/6 accepted, 1.0 at all 3 positions — transport-only) | external signal (source unassigned) | harness/external | R:372–394; D:m3-1536-ext-stop |
| MTP3@2K | completed; divergence from MTP0 auth at token **4** (5.931661201 diag) | cross-runtime parity | runtime identity drift | R:455–483; D:m3-3072-parity-q |
| MTP4@1K | **2/2 exact** (13.326165 / 17.290937; 204/204 accepted, 51/51/51/51 perfect); supervisor rc **143 — stop sentinel never reached detached server group** | teardown-gate failure | harness lifecycle | R:396–425; D:m4-1536-teardown-q |
| MTP4@2K | 0-byte body / sampling timeout; 384 computed / 0 output; 1 comp + 1 copy reset per card, 60 fault responses | sampling-RPC timeout | engine stability | R:427–453; D:m4-3072-bn |

### 1.5 Thinking-transfer (MTP3 official-quality) — UNQUALIFIED, content-side clean
19/19 completed answers correct; all 19 final answers matched MTP0 (normalized-final agreement 19, raw-final-hash agreement 19; reasoning-hash agreement 1 — sampled profile, expected) [R:121–130; D:m3-official-quality-2]. The 20th request (seed `2026082713`, `copy_phrase`) stopped at **98 computed / 33 output** → 300 s worker timeout → **API 500**; shutdown logged **8 engine-reset messages**; 6 grid rows never ran. **No answer-quality failure observed; blocker = repeated-session stability.** MTP0 target-only official thinking passed 25/25 [R:107–114; D:m0-official-quality-2].

### 1.6 DSv4 comparison point (same box — the k-wisdom source)
- Record `cmrquta9905w3lg013m5vxoqx`: strict-suite medians **80.820052 / 76.900178 / 78.287226** (median-of-medians **78.287226**; 12 cold prompts ×3, tokens 1–100) [A:15].
- **Exact M=7 capture** (fixed descriptor, NOT padded): **padded-M8 measured 60.518 vs 64.661 exact-M7**; M=8 verifier; ~3.5 emitted tokens/cycle; per-position conditional acceptance **79.64/74.28/68.83/71.07/70.80/66.88/58.88%** (positions 1–7) [A:28,29,32].
- Machinery: persistent Markov sampler with W1-only replication (removes 7 all-reduces; +0.452 ms/cycle); guarded sharded greedy target argmax + native target-token rejection [A:30–31].
- **MTP2 reuse deadlock (directly relevant failure class):** second draft position accepted 0.5–2.2%, then request **hung + exhausted shared-memory broadcast blocks for 180 s** [A:93]. Flash-Next MTP4@4K's stall is the same *family* (deep draft + long ctx → shared-broadcast wait + timeout), not necessarily the same mechanism (INFER).
- k-tuning screen: M=4/5/6/8 all lost to M=7; DEV winner 79.80 < record 80.82 [A:97].
- Cookbook counterpoint (task-supplied, not re-verified here): on single/dual-B70 smaller Qwens **MTP4 is the standard winner** (Qwen3.6-35B-A3B 170.9 tok/s MTP4; dense 27B 69.3 MTP4) — k=4 winning elsewhere while Flash-Next MTP4 quarantines at 4K is a **Flash-Next-on-XPU-stack-specific blocker, not a law of nature**.
- vLLM platform fact (web, 2026-09-01): MTP is model-family-gated; `num_speculative_tokens` controls speculative depth; model-based methods (EAGLE/MTP/drafts) give the best latency reduction — https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/ and https://docs.vllm.ai/en/latest/features/speculative_decoding/

---

## 2. Qualification-blocker analysis (named blocker per cell; ranked cheapest-to-clear first)

| # | Cell(s) | Named blocker | Evidence | Class | Cost |
|---|---|---|---|---|---|
| **B1** | MTP4@1K, MTP3@1K | **Harness lifecycle: stop-not-forwarded-to-detached-group + unassigned external SIGTERM** | rc 143, stop sentinel never reached server group [R:412–414]; SIGTERM at 00:05:06 [R:381–388]. A descendant-aware supervisor ALREADY exists and passed zero-rc on the MTP4@2K teardown [R:441]. | harness bug | **cheapest** — port helper, re-run; known-perfect 1K acceptance already on file |
| **B2** | 8K row (k1,2,4) + 2K row (k2,3) | **Runtime identity drift: parity vs legacy MTP0 authority (`658965050`, 192 MiB)** | all four 8K/2K parity quarantines cite authority vLLM `658965050` vs candidate `1372c62d` [R:480–481; N:m3-8kprereg:57–61]; MTP2=MTP4@8K outputs byte-identical [§1.3] | runtime identity drift | cheap–medium — re-base MTP0@2K/8K on `1372c62d` same-cache, then re-run per k |
| **B3** | MTP2@1K | **Host gate: 11 corrected NVMe records post-cutoff** | exact transport/parity/acceptance all passed [R:493–505] | host gate | cheap but **uncontrollable** — needs a quiet-run window (no corrected link events in window); rerun is valid |
| **B4** | MTP3 thinking transfer | **Engine stability: repeated-session API 500 on 20th request + 8 shutdown resets** | D:m3-official-quality-2 (98/33 computed/output; scheduled spec tokens −1); content 19/19 clean [R:121–130] | engine stability | medium — fresh-server-per-batch sidesteps session accumulation |
| **B5** | MTP4@4K (highest value) | **Engine stability: prefill-boundary worker stall + 4-card resets** | 3,904/4,096 computed @ 92.86% KV, 5×/8× shared-broadcast waits, sample_tokens RPC timeout [D:m4-4352]; same signature cleared once for MTP2 by 21→32 blocks [R:564–590] | engine stability | **most expensive** — material treatments in order (§4 R5a→R5b→R5c); deadline-raise alone is forbidden |

Ranking rationale: B1 unlocks two cells with zero engine risk and known-good acceptance; B2 clears a whole row by *re-baselining the yardstick*, not new engineering; B3 is free if the storage link is quiet; B4 is a run-discipline change; B5 is the only one needing material engine work — and it also decides the 4K headline (18 tok/s gate).

---

## 3. k-optimal determination

### 3.1 Answer
- **@4K: k=3 (MTP3, 15.502)** is the serving target. k=4 would be preferred *on paper* (512 shows k=4 = 1.392× k=3, extrapolating ≈21.6 tok/s at 4K, DERIVED) but MTP4@4K has **no valid measurement** — quarantined pending the §5 gate 1. **Do not silently serve k=4.**
- **@8K: k=0 (MTP0 formal 3.98)** is the only qualified option. All MTP 8K cells are quarantined; none may be served.

### 3.2 Marginal-gain-per-k vs stability tradeoff
- Marginal gain in tok/s per k is **NOT monotone** (§1.1/§1.2, DERIVED): at 512 the *deepest* position (k3→4) adds the most (**+5.838, +39.2%**); at 4K k1→2 adds almost nothing (**+0.989**) while k2→3 jumps (**+5.608**). Rationale (INFER): each added position over the native draft head costs only a small incremental GDN/main-decode step, so on an *easy, low-entropy* prompt (100% acceptance) deeper k is nearly free throughput; on a *realistic* prompt (93.8% and falling) the deepest position is the least likely to accept and the first to wedge the engine.
- The stability axis is measured directly: the two engine stalls (MTP2@4K-21blk, MTP4@4K) and the 8K no-receipt/parity cluster all live at the deep-k × long-ctx corner. **k=4 at 4K is a bet that the 4th position both accepts ≥ 50% and does not wedge the legacy-GDN draft path (patch 0018) at near-full KV.** Until both are measured, k=3 is the stable optimum.
- Flip conditions: k=4@4K wins iff gate 1 (§5) passes (≥ 18.0 tok/s, parity, 20× zero-reset). At 8K, the first *qualified-completing* MTP k re-ranks the row (MTP2@8K's diag 6.23 is the only 8K candidate above MTP0's 3.98 — if it re-qualifies on same-runtime parity, it becomes the 8K target).

### 3.3 Acceptance-rate-matters (DSv4 lesson, applied)
- **1716/1716 (100%) @ MTP4-512 vs 799/852 (93.78%) @ MTP3-4K**: the 4K acceptance drop is the single biggest determinant of where k lands. DSv4's record runs ~3.5 emitted tokens/cycle at **59–80%** positional acceptance with k=7 — it won because *per-position acceptance stayed high deep into the draft*. Flash-Next's MTP4@8K diag is 52.4% with pos3 at 34.1%: at those rates the 4th position pays draft/decode cost for a ~1-in-3 payout. **Acceptance rate, not k, is the decision variable.**
- Why the drop (INFER, three parts): (a) the 512 screen prompt is a single repetitive 146-token prompt with a 256-token continuation — trivially predictable, hence 100% acceptance; (b) the 4K/8K fixtures are needle-in-filler with genuinely higher forward entropy, so unconditional next-token prediction saturates lower; (c) per-position acceptance decays monotonically with position at every context (§1.2/§1.3, DERIVED), so the deepest position absorbs the worst of the context effect.
- Measurement gap to close on the rig: **per-step acceptance arrays are NOT recorded** by the current harness (cumulative counters only; "do not call cumulative counters per-row acceptance evidence" [N:m4-4352prereg:49–51]). The qualification matrix (§4 R6) must add a per-position acceptance trace so k decisions are grounded, mirroring DSv4's DEV screen [A:32].

---

## 4. Remediation receipts (per blocker — exact command, frozen identity, expected output, pass/fail)

General rules: every receipt is a fresh preregistered attempt with **new immutable run/cache/evidence roots and a new port**; freeze + record `sha256` of every changed executable BEFORE first use; any stop blocks publication and stays scoped (no speed credit ever comes from a quarantine row). After any card reset, run a fresh four-rank collective + known-good generation canary before further GPU work [N:m3-8kprereg:130–131]. Retained frozen hashes usable as-is: launcher base `62b40c92…`, MTP3/8K wrapper `fbf4e826…`, one-request client `83e18abc…`, supervisor `94292d12…` [N:m3-8kprereg:99–106]; MTP4/4K launcher `b8fedb33…`, wrapper `bb05718f…`, quality suite `3350671d…`, exact-depth suite `8f162c1a…`, comparison harness `d590c63c…`, 4K fixture `c44fccba…` [N:m4-4352prereg:42–47]; sealed MTP0 4K baseline `a458747f6c…` [N:m4-4352prereg:49–51].

### R1 — [B1] MTP4@1K + MTP3@1K re-run with descendant-aware lifecycle (cheapest, clears two cells)
```bash
# Only the supervisor changes; model/source/stage/topology/request stay frozen:
#   vLLM 1372c62d, stage 2f829747, model bcd9f01, TP4/EP4 eager, max_len 1536,
#   MTP4 -> 29-block cache (341,266,432 B) [R:400]; MTP3 -> 25-block (294,195,200 B)
cp tools/supervise-<descendant-aware>.sh  tools/supervise-tp4-mtp4-1536-r2.sh
sha256sum tools/supervise-tp4-mtp4-1536-r2.sh > R1-supervisor.sha256   # freeze before use
# protocol: frozen p1024/o256 cache-zero; stop sentinel MUST reach the server group
```
- Contract: supervisor rc **0** after stop (old failure rc 143 = sentinel never forwarded [R:412–414]); postflight no listener/group/compile/RPC path; 4 cards < 43 MiB; no B70-addressed event.
- Expected: 2/2 exact p1024/o256 rows; perfect 204/204 acceptance; median ≥ 15.0 tok/s (diag on file: 13.326165 / 17.290937 [R:409–410]). Also add a per-position acceptance trace here (measurement-gap fix, §3.3).
- **Pass → MTP4@1K qualifies (same for MTP3@1K if its SIGTERM recurs after the port/identity check is audited). Fail → keep quarantined; attach logs to the first-request no-output family, not to MTP depth.**

### R2 — [B2] Same-runtime MTP0 parity re-baseline for 2K/8K (clears the row by re-baselining the yardstick)
```bash
# 1) boot current-source MTP0 on 1372c62d with the SAME fixed cache as each candidate arm:
#    8K: MTP1/2 32-block 376,569,856 B; MTP4 36-block 423,641,088 B [R:800]; 2K: 32-block
#    capture NEW same-runtime MTP0 authority {raw receipt SHA-256, output-token hash}.
#    Do NOT reuse 0efd150b…/2a8bfbb… — those are the 658965050-era authorities.
# 2) for k in {1,2,4} @8K and {2,3} @2K: exactly ONE p8192/o128 (resp. p2048/o128) request
#    hash-vs-NEW-authority. Bounds unchanged: 300 s worker gate, 900/910 s client, 2,700 s lifecycle.
```
- Contract: descendant-aware supervisor; frozen executable hashes above; cache-zero; iterate one boot per cell (each p8192 boot costs ~15–16 min TTFT + load — the 3-hour window fits the 8K row + 2K row with margin).
- Expected: completion + a definitive parity decision per k. Given MTP2/MTP4@8K already agree **with each other** byte-for-byte [§1.3], a same-runtime MTP0 either matches both (parity cleared — cells graduate to screened) or matches neither differently (engine/model divergence — upgrade classification and stop MTP-8K work). **Pass → then apply the rate gate: any k whose qualified 8K median ≥ 4.1 diag replaces MTP0@8K as the row target (MTP2 candidate).**

### R3 — [B3] MTP2@1K clean-host re-run (host gate)
```bash
# NO code change. Re-run the frozen MTP2@1K identity [R:486–492] on a quiet boot:
#   verify BEFORE start and AFTER teardown that the kernel journal has zero corrected
#   NVMe/root-port records at 0000:01:00.0 within the run window (frozen cutoff).
```
- Contract: pre-run journal check must be clean; a single corrected record after cutoff = auto-quarantine (the rule that caught the 11 records) [R:501–505].
- Expected: 2/2 exact p1024/o256, perfect 170/170 acceptance, decode median ≥ 11.0 tok/s (diag 10.682699 / 12.641866 [R:497–499]). **Pass → MTP2@1K qualifies (the best-observed 1K throughput on file). Fail → storage-link repair is upstream of any 1K MTP cell.**

### R4 — [B4] MTP3 thinking transfer re-run under fresh-server-per-batch discipline
```bash
# Split the 25-response battery across server sessions — the failure was session-length,
# not answer quality (19/19 correct, 20th request API-500) [R:121–130]:
#   session A: scout (4/4) + 7 grid seeds    session B: 7 grid seeds
#   session C: 7 grid seeds  (all official-thinking, reasoning_effort=xhigh, parser qwen3)
# Each session: fresh four-rank preflight; identity per N:m3-official-quality-prereg:34–47.
# Count API-500 / worker timeouts across ALL sessions; require 25/25 completes.
```
- Contract: unchanged quality gates (scout 4/4, grid 21/21, nonempty separated reasoning/final, normal stops, complete usage, zero cache) [N:m3-official-quality-prereg:88–93]; no timing rows authorized.
- **Pass → MTP3 thinking-qualified (content already 19/19-vs-MTP0). Fail with a NEW API-500 inside a short session → engine-stability blocker is request-scoped, not session-scoped; escalate to the same family as B5.**

### R5 — [B5] MTP4@4K material-treatment ladder (the big one; sub-receipts in order, stop at first pass)
```bash
# Frozen for every sub-step: vLLM 1372c62d, stage 2f829747, model bcd9f01, TP4/EP4 eager,
# max_len 4352, BLHNC, prefix/async off, batched cap 64, UVA 12.22 GiB/rank,
# enable_thinking=false, port NEW, roots NEW. Record sha256 of every changed launcher.
#
# R5a - cache-geometry: 29 -> 36 blocks (423,641,088 B: the exact allocation that admitted
#       9,504 tokens on the 8K MTP4 arm [R:800]). Rationale: MTP2@4K stalled at 21 blocks,
#       passed at 32; both 3904-computed stalls logged ~93% KV usage [R:564-590; D:m4-4352].
#       Expected: 3 salted p4096/o256 rows, hash == accepted 4K target (6949154f…), decode
#       median >= 18.0, zero resets, no shared-broadcast waits.
# R5b - if R5a stalls: halve max-batched/scheduled tokens 64 -> 32 (the scheduler-32 treatment
#       extended MTP2@16K 3,200 -> 5,440 computed [R:878-904]) AND 36 blocks.
# R5c - if R5a/b stall: engine diagnosis in the legacy-GDN draft path (patch 0018) and the
#       spec-group schema guard (0016); capture was already added - do not change gates,
#       do NOT raise VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS above the 300 s default.
```
- **Pass (R5a or R5b) → MTP4@4K is a candidate for gate 1 (§5); run the full qualification battery (§4 R6). Fail on R5c → MTP4@4K stays quarantined as engine-stability; k=3 stands at 4K and the 18 tok/s gate is moot.**
- Do not interpret a sub-step that stalls at 3,904 computed as "deadline too short" — that signature is prefill-boundary, and the MTP2 precedent shows geometry clears it.

### R6 — Clean qualification matrix run (graduates MTP3@4K screen → qualified, and MTP4@4K if R5 clears)
Per cell (MTP3@4K always; MTP4@4K only after R5): **3 fresh server boots**, each passing, in order:
1. **Identity gate** — verify the 5-part frozen identity (§0.1) + launcher SHA; fresh four-rank collective; four 12.22-GiB placement receipts; exact block count (MTP3: 25 / MTP4: R5-adjusted); capacity ≥ 4,352.
2. **Cache-zero & determinism** — exact 4,096-token needle with **zero cache reuse**; 16/16 fixed-set repeats hold ONE output hash; all 24 quality requests report zero cached/created-cache tokens.
3. **Formal row** — p4096/o128: 128 returned token IDs, length stop, valid 100-event/99-interval window, zero cached.
4. **3 service rows** — p4096/o256/c1, salts `context-r1/r2/r3`, no harness warmups; all three rows must return the **accepted target hash** (repeat-hash rule); report every row, never suppress variance.
5. **MTP/acceptance capture** — cumulative counters + NEW per-position acceptance array (measurement-gap fix, §3.3).
6. **Teardown** — descendant-aware supervisor rc 0; no listener/group/compile/RPC path; 4 cards < 43 MiB; bounded kernel-journal capture; zero corrected NVMe records in window (clean-host).
Median-of-medians = median of the three boot medians (each boot = median of its 3 rows). Same-boot Grade-C is NOT qualification; qualification requires the 3-boot replay.

---

## 5. Acceptance gates (falsifiable — every threshold explicit, with justification and failure condition)

**G1 — MTP4@4K qualifies (k=4 becomes the 4K target) iff ALL of:**
1. **median-of-medians decode (aft-1st-text, 3 rows × 3 boots) ≥ 18.0 tok/s.** Justification (DERIVED): at 512 the 4th-position marginal is +5.838 tok/s (MTP3 14.889 → MTP4 20.727, +39.2%) [§1.1]. Applying k=3's qualified 4K value 15.502 [§1.2], MTP4@4K must retain **at least half** of the 4th-token marginal to justify the switch: 15.502 + 2.92 ≈ 18.4 → round to ≥ 18.0. The full-ratio extrapolation (×1.392) would be ~21.6; half-marginal is the conservative bar because deep-position acceptance provably decays (8K pos3 = 34.1% [§1.3]) — a candidate under 18.0 is carrying dead-weight speculation.
2. **Exact-token parity** vs the eager MTP3 4K authority — all 3 rows hash ≡ accepted target (`6949154f…` formal, `a458747f…` sealed baseline); **zero divergence at any index**. (At 4K MTP1/MTP2/MTP3 all matched the sealed MTP0 authority 26/26 [R:622–624], so parity at 4K is an achievable, meaningful bar.)
3. **Zero engine resets across 20 consecutive p4096/o256 requests** (2 boots × 10), zero worker-response timeouts (no `RPC call to sample_tokens timed out`), zero shared-broadcast waits, every request completes with 128 IDs + length stop + zero cached tokens.
4. **Per-position acceptance floor** on the battery: pos0–2 ≥ 70%, pos3 ≥ 50% (else the cell re-quarantines as acceptance-collapse, not engine-wedge — keeps §3.3's decision variable explicit).
5. **Clean host** — zero corrected NVMe/root-port records in-window after the frozen cutoff.
**Failure condition:** any of G1.1–G1.5 fails → MTP4@4K remains quarantined; **k=3 stands at 4K**; do not re-run blind (§6).

**G2 — MTP3@4K graduates screen → qualified iff** median-of-medians ≥ 15.0 tok/s (keeps a 0.5 margin under the measured 15.502 while allowing cold-start variance) AND R6 per-boot gates pass AND a separately booted 3rd boot reproduces within ±9% of the first two AND zero resets across 20 consecutive requests AND clean-host. **Failure →** back to screened; no change to §3.1.

**G3 — 8K row.** Any MTP k re-qualifies iff R2 same-runtime parity passes AND the request completes under the 900 s inner bound with a durable receipt AND per-position acceptance ≥ the 4K analog (pos0 ≥ 90%, posN ≥ 50%) AND zero-reset teardown AND clean-host. **Until the first pass: 8K serves k=0 (MTP0, screened 3.98 formal).**

**G4 — Thinking transfer** qualifies iff 25/25 complete under R4 with zero API-500 / zero worker timeouts and ≥ 19 final-parity vs MTP0. **Failure →** unqualified; no timing credit; content-side findings are unchanged (no quality failure was ever observed).

**G5 — 1K/2K row.** A re-run qualifies iff it passes R1/R3 identity, exact transport, perfect-or-declining-by-position acceptance, and its **cell-specific blocker** (B1 teardown-0, B3 clean-host) is gone, plus decode median ≥ the cell's existing diagnostic floor where one exists (MTP4@1K ≥ 15.0; MTP2@1K ≥ 11.0). **Failure →** cell stays quarantined with its named blocker.

---

## 6. What NOT to do (hard constraints)

**Quarantined cells — do not re-run blind:**
- **MTP4@4K: no deadline-only retry** — raising `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` alone is explicitly unauthorized [N:m4-4352prereg:120–122]; requires the R5 material-treatment ladder.
- **No further MTP2@16K retries** (scheduler-32 treatment exhausted, supervisor rc 70, 8 card resets) [R:878–904]; **no MTP0@16K/24K/32K** until a material runtime treatment passes a fresh four-card gate (A3/A4 nondeterminism: 16,213-token request wrong on identical same-server repeat; sampling-RPC stop at 1,600 computed; 8 resets) [R:865–876].
- **No blind re-run of the 16K row or vision/graph lanes** — vision NOT established (30-cell contract all missing), graph NOT established (a1–a7 exhausted) [S:40; R:34–35,914–916]. Lane 2 is text MTP only.

**Protected anchors — never touch / never lower:**
- MTP0 `5.515783 tok/s` [N:a30:43], MTP4@512 `20.72717637199404` [R:833], MTP3@4K `15.50156510641242` [R:834], and the entire 512 grid (§1.1) plus the MTP3/4K 25-block receipt [D:m3-4352-a1]. Every Lane-2 pass "does not lower or replace" prior cells — that is the standing lab rule [e.g. R:122–130].

**Runtime identity:**
- Do not substitute kernel source `ad25aa9f…` for the loaded stage `2f829747…` [S:48]. Do not include patches 0011/0013 (diagnostics) on any timing/qual tree [S:49]. Patches 0019–0033 are candidates, not production [S:49]. Do not drift dependency versions (observed-only, non-installable) [S:50].
- Do not run MTP cells on the newer `797769b34` checkout (grouped-HC descendants) — that is A31's identity [N:a31:27–31].

**Host/environment:**
- **Never set `SYCL_CACHE_PERSISTENT=1`** on B70 (poisons cache, SEGV next boot) [lab plans:238–239].
- **Wedge watchdog mandatory:** every run must have a supervisor that (i) owns the exact PID/process-group and port/state/paths, (ii) enforces no-progress bounds by **external kill, not in-process alarm** — in-process SIGALRM cannot bound a stall in a non-yielding C path [edgequant-env-quirks, batch-rig]; this is the 180 s MTP2-reuse-deadlock class [A:93], (iii) greps the kernel journal for `xe 0000:(23|27|43|47):00.0.*(reset|fault|timeout|timed out|fatal|wedged|failed)` and fails the run on a hit [tools/supervise-*.sh:148–151], (iv) enforces memory floors (root ≥ 40 GiB, 64 GiB swapfile, mem floors — 4× 32 GiB cards + 51.2 GiB pinned UVA + graphs on a 128 GiB-class host) [S:19], and (v) posts a fresh four-rank collective + generation canary after any card reset.
- **Boot prerequisite:** the current boot (`c36480de-9150-4182-9888-08c85d2d9de4`) is rejected after the event-chain device-lost [N:a31:43–45]. No Lane-2 GPU work until an **attended fresh boot** passes the ordinary-XCCL affinity component, and A31 (M1-only eight-warp, frozen) either claims that boot's first full-model-load slot or runs separately [N:a31:45–55]. Lane-2 cells schedule either after A31 or on a second accepted boot — never on the rejected boot.
- **Quiet-host policy for clean-host gates:** corrected Samsung NVMe/root-port events are perennially seen and auto-quarantine any cell whose prereg declares the clean-host rule [S:76; R:501–505,820–821]. A corrected event is host context, not a B70 failure — but it IS a blocker for "qualified" wording.

**Methodology:**
- Do not pad the draft descriptor to a rounded M — DSv4's padded-M8 lost to exact-M7 (60.518 vs 64.661) [A:28]; keep the native MTP fixed descriptor and avoid any padded capture on the legacy-GDN draft path.
- Do not reuse draft positions across requests (the DSv4 MTP2 reuse deadlock: 0.5–2.2% acceptance then 180 s hang) [A:93].
- Do not treat cumulative MTP counters as per-row acceptance evidence [N:m4-4352prereg:49–51]; per-position acceptance arrays must come from the new trace (R1/R6).
- Do not grant speed/quality/deployment credit to any quarantine diagnostic rate; do not relabel cross-runtime parity as MTP-caused corruption [R:478–482].
- Do not serve k=4 at 4K or any MTP k at 8K until gates G1/G3 pass — §3.1 is the deployment answer, and it is falsifiable by exactly those gates.

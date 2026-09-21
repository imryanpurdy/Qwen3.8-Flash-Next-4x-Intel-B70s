# Tracker Search: 12-Concurrent Output Stall (EngineCore idle, wave completions, PleOffload duplicate signature)

Date: 2026-09-19 · System: vLLM v0.26.1rc1.dev1250+g76cfe1cd8, 4x Intel Arc Pro B70 XPU, TP=4,
hybrid GDN/linear-attention (Qwen3.8-Flash-Next class), V2 model runner, PLE n-gram CPU offload
(mmap'd host table, PleOffloadWorker process), MTP0. Symptom: fine at c=8 (~180 tok/s), step-function
collapse to ~30 tok/s at c=12, rounds lock to ~237s, APIServer shows 12 "running" with zero completions
for up to ~4-min windows, EngineCore idle at `_process_input_queue` (queue.get, core.py:1453), workers
idle in shm_broadcast dequeue (multiproc_executor.py:1037), completions arrive in waves, engine still
serves new small requests in 0.4s. Log: `PleOffloadWorker: Duplicate PLE request for dp_rank=0; skipping
duplicate` every ~4 min. Gauges unreliable in this build.

Method: ~40 GitHub search-API queries on `vllm-project/vllm` (issues + PRs, open+closed) plus full
bodies/comments fetch for ~25 shortlisted items. Results saved in `docs/gh_sweep_results.json`,
`docs/gh_details.json`, `docs/gh_phase2.json`, `docs/gh_phase3.json` (same dir).

---

## 1. Top candidate: PLE-offload connector vs async scheduling (custom feature, upstream PR)

**PR #53899 "Support PLE-Offload for Qwen3.8-Flash-Next"** (open, UNMERGED, paused for maintenance;
32 comments) — https://github.com/vllm-project/vllm/pull/53899 — is the upstream home of our custom
PLE n-gram offload. The whole feature is NOT on main; it lives in this PR (fork:
`peakcrosser7/vllm`, base PR #53896 "Support Qwen3.8-Flash-Next", same branch, plus UVA variant
PR #54371, mmap variant PR #54129, ROCm variant PR #57497). Because it is an unmerged branch, fixes
in that branch are **not** in any official release — a custom build can sit on pre-fix code.

Upstream thread documents **three distinct stalls** (jschmied's table in issue **#53960**,
https://github.com/vllm-project/vllm/issues/53960 - "VLLM_PLE_CPU_OFFLOAD deadlocks at kernel warmup"):

| # | Stall | Status |
|---|-------|--------|
| 1 | uniproc executor never spawns offload worker (TP=1) | fixed `95dc96d1d012` (only affects TP=1/uniproc) |
| 2 | **`PleOffloadConnector` single-slot event race under async scheduling / MRV2** — connector allocates ONE `_input_ready_event` + ONE `_d2h_done_event` for the whole connector and `_request_queue = queue.Queue(maxsize=1)` ("PLE rejects DBO, each forward consumes its output before the next launch, so one pending request is sufficient"). Under async scheduling `max_concurrent_batches == 2`: `_launch(#2)` re-records the shared event on the model stream behind forward #1 -> **deadlock** | fixed `4e8b849b8d97` ("fix PLE offload for async MRV2 scheduling", 2026-08-29) |
| 3 | V1-runner variant; offload thread idle on empty queue, main stuck in warmup forward — does **not** fit the shared-event race; submit path broke upstream | unresolved (V1 runner is self-disabled for this model family; #53896 "'Limit Qwen3.8-Flash-Next to model runner V2'") |

Also from #53899/#53960 threads, same feature:
- **One-step-lag under CUDA graphs** (jschmied, #53899 comment 2026-09-06): with `cudagraph_mode=PIECEWISE` (default) every forward consumes the *previous* step's PLE outputs; graph capture's dummy PLE signal keeps the semaphore "one step ahead for the life of the server". `cudagraph_mode=NONE` is correct; 11-line fix `PleOffloadConnector.prepare_forward` reset in `peakcrosser7/vllm#13`.
- **Offload-worker CUDA context invisible to the memory profile** (ssubbotin, #53899, 2026-09-17): engine over-commits GPU; OOM at 8 concurrent ~24k-token requests with `--gpu-memory-utilization 0.97` (https://github.com/vllm-project/vllm/issues/54905-adjacent; same finding).
- **`pidfd_getfd`/ptrace gates** (DONGRYEOLLEE1, jschmied, alansrobotlab2): the offload worker ptrace-attaches to its parent; Yama `ptrace_scope=1` or default seccomp = EPERM, sometimes after ~10 min of serving, engine init fails.
- **GPU-memory accounting** and **PP>1 rejection** for PLE checkpoints: #54709, #54722, #54431 ("Use ragged PLE IDs for CPU offload"), #54525 (PLE prefill memory amplification), #55375 (fused PLE conv strides).

Why it fits: our build is exactly V2 model runner + PLE offload + (per the async-scheduling default
below) async scheduling is ON. If our tree predates `4e8b849b8d97`, the single-slot connector race is live.
The 4-min periodicity + wave completions are consistent with a repeatedly re-armed/aborted in-flight
PLE round (the duplicate-PLE skip could be the fork's attempt to dodge the race and instead drop work).
**UNVERIFIED** — we cannot read the fork's connector code from here; the exact "Duplicate PLE request for
dp_rank=0" message is **not** in any upstream source/issue (searched `"duplicate PLE"`, `"PleOffload"`,
`dp_rank` — 0 matching hits). See §6.

## 2. Best structural match for the idle-signature: #40926 (open) — not PLE-specific

**[Bug] V1 engine + MTP + GLM-5.1 — workers hang under sustained traffic, sample_tokens RPC timeout,
EngineDeadError** — https://github.com/vllm-project/vllm/issues/40926 (open, 9 comments).

- `ccgibson` root-caused it (v0.20.0): **hang manifests in shm_broadcast.py `acquire_read`/`_spin_condition.wait` but the bug is in the caller — `multiproc_executor.py collective_rpc()` was refactored (FutureWrapper, v0.19.1→v0.20.0)**. Progressive stall: throughput 5924 → 2257 → **0.0** over 20s while 3 requests stay "running"; no CUDA error, no OOM, no worker death; KV at 37% (no pressure).
- **Reproduces with `speculative_config=None`** ("bug is not specific to MTP"), ~14 min after engine ready, same `step_counter=0`.
- More data points: `ARSblithe212` (2026-07-02): Qwen3-Next 27B + MTP, TP=2 — **async scheduling is auto-enabled even though not passed** (`--no-async-scheduling` helps); generation silently stalls at 0.0 tok/s. `leviGp` (2026-08-10): Qwen3.6-35B hybrid — py-spy of the exact livelock point. `nichdiekuh` (2026-08-21): TP=1, **both v0.26.0 and v0.27.1 affected**; **"only the MoE + linear-attention (GDN/Mamba) hybrids do [this]"** — dense Qwen3.6/3.8-27B never hung. `klapom` (2026-09-04): five negative controls, no repro (so not universal).

Why it fits: worker-side shm_broadcast spin during a collective is precisely what our py-spy shows
(`multiproc_executor.py:1037`); engine-side stall with requests stuck "running" matches; hybrid
linear-attention + MoE is the common denominator (our model class); still-open with 0.26/0.27 reports.
Why it may not fully fit: CPU-side py-spy frames can be **misleading** — `xexex7` (#53960, 2026-08-30)
found py-spy showed the OOM-container of a spinning GPU kernel: "py-spy always showed the *next*
lazily-compiled Triton kernel ... which is why the stacks looked contradictory" (their true hang was a
GPU spin in custom_all_reduce or PDL `_build_qsa_metadata_kernel` on sm120). On XPU the same caveat
should apply: **confirm with a GPU-side signal (e.g., xe/xpu utilization, strace of the worker syscalls)
before trusting "idle"**.

## 3. Hybrid multi-KV-group / chunked-prefill concurrency bugs (no spec decode)

- **[Bug] AsyncScheduler `num_output_placeholders` underflow with chunked prefill + concurrency (no spec decode, no preemption) — regression from 0.24.0** — https://github.com/vllm-project/vllm/issues/57562 (open, v0.29.0, Qwen3.6-35B-A3B-FP8 hybrid GDN on 1x H100):
  trigger = chunked prefill concurrent with other requests generating; model runner returns a *spurious sampled token*
  while the prompt is still prefilling (`computed 8448 < num_tokens 9119`), scrubbing placeholder accounting.
  **`--no-async-scheduling` → 24/24 requests complete, no underflow.** v0.24.0 is clean (0 underflows).
  Related: #48245 (stale-output rework in 0.27.0+), #35755/#50692 (same assert via /v1/realtime), #46424
  ("Mamba/GDN attention reclassifies a single-query-token prefill as a decode step; possibly related").
  Our v0.26.1rc1 sits inside the regression window (0.24.0 clean → 0.29.0 broken) — **UNVERIFIED** whether
  the runner-side change is already in our commit.
- **[Bug][Spec Decode] Hybrid GDN (Qwen3.5/Qwen3.8 27B) + MTP: scheduler runs only ~3 concurrent sequences at batch ≥ 4** — https://github.com/vllm-project/vllm/issues/55533 (open) + WIP diagnostics PR #55617:
  scheduler block accounting charges per-Mamba-KV-group (`1+k` blocks per group, `num_speculative_blocks =
  num_speculative_tokens` per Mamba layer); `allocate_slots` returns None → WAITING loop breaks → stable
  `floor((num_gpu_blocks−1)/(groups·(1+k)))` concurrency window, zero preemptions (Raymondlol's math).
  Related: #54076 (mamba group block size in align-mode chunk splitting), #55390 (annotate MTP draft KV
  groups positionally on hybrid grouping path), #37429 (hybrid KV sizing), #42960 (batch-invariance for GDN).
  Our MTP0 case does not need the 1+k term, but the multi-KV-group accounting path under concurrency is the
  same code; a hidden `+k`/align-mode block charge would cap scheduled concurrency silently.
- No upstream issue shows a pure "step cliff between c=8 and c=12" on non-hybrid stacks. The only
  throughput-degradation-at-concurrency-12 data points are hybrid-GDN (#57680, below) and the B70 XPU
  corruption at c=12 (#53480, below).

## 4. Exact-platform family: Intel Arc Pro B70 / XPU

- **[Bug][XPU] Silent persistent output corruption (endless "!" / token 0) under sustained concurrent decode on Arc Pro B70, W4A16 27B head_dim 256** — https://github.com/vllm-project/vllm/issues/53480 (open, 8 comments): 27B Qwen-family W4A16 on B70; ramp c=1,2,4,8,12 → **at c=12 all 36/36 responses garbage** (167 tok/s then 50.3s e2e), persistent until engine restart, zero dmesg; **the only clean stretch correlated with `--max-num-seqs 4`; every recurrence at width 12** (bryanvine). `faaany`-attributed 27B-only; **v0.29.0 XPU image + vllm_xpu_kernels 0.1.14.1 (vs 0.1.8.2) fixed it**: 4 rounds, 318 requests, 0 garbage at c=12 (bryanvine 2026-09-15). The 0.1.8.2 kernels "predate the v0.1.9 mixed-batch attention rework". Also: #41663 (B70 TP=2 GP fault/BCS reset), #57535 (XPU Qwen4Exp registry test skip).
- Caution: the symptom documented in #53480 is corruption, not stall; but it proves **c=12 on B70 with 27B-class hybrid XPU is past a documented failure boundary**, and points at the vllm_xpu_kernels / mixed-batch attention stack as the variable that fixed it. On our 0.26.1rc1 custom build, **verify which vllm_xpu_kernels is bundled** (testable locally).

## 5. "No available shared memory broadcast block" / rank-blocked signposts

- **[Bug] qwen3.8-flash-next-fp8: No available shared memory broadcast block found in 60 seconds.** — https://github.com/vllm-project/vllm/issues/54559 (open): same fork image (`vllm/vllm-openai:qwen38-flash-next`, `0.1.dev20073+g8e685d198`), Qwen3.8-Flash-Next FP8, TP=2 + EP, `VLLM_PLE_CPU_OFFLOAD=1`; sequence in log = **GPU OOM during `vllm serve` wrt offload-path over-commit, then** the shm_broadcast timeout — i.e., in this code family the broadcast wait is the *symptom* of a peer/rank that died or never finished init, not the root cause. Our py-spy "workers idle in shm_broadcast dequeue" should be read the same way (something upstream stopped producing broadcasts).
- **#57423 / #57635**: FlashInfer autotune cache hits only on rank 0 → engine launch deadlock (rank-1+ waits; fix #57635 persists autotune cache per rank). CUDA-only, launch-phase — listed for completeness (we autotune per-rank we can list as "if any first-call autotune/jit happens per rank under a new batch shape while another rank skips, a collective can block exactly at shm_broadcast" — **UNVERIFIED on XPU**).
- **#44185** — DP hang with MoE draft near max_model_len; #45800 — MNNVL rendezvous hang; #46121 — proposal: **model-free microbenchmark for the frontend↔EngineCore IPC path** (Msgpack + ZMQ send/recv) — the closest upstream thing to "suspect the ZMQ/shm output path"; it's a diagnostic, not a fix.

## 6. Frontend/API-server side: no exact match; the "Duplicate PLE request" log is fork-origin

- Query family "requests stuck running at frontend while engine idle", "AsyncLLM output handler deadlock",
  "zmq message lost", "VLLM_OUTPUT_PROCESS", "output processing stall" → **no upstream issue describes this
  exact split** (EngineCore idle at input queue + API server holding zombie-running requests + wave completions
  + small requests still served). Nearest: #17385 ("v1 AsyncLLM hangs with 2 successive batches"), #17972
  (server hangs after 1-2 requests, closed: config), #27194 (EngineDeadError at high-concurrency benchmark via
  `get_output_async` output_handler), #41834 PR (DeepSeek V4 Flash — unrelated), #36826 (streaming stalls +
  concurrent requests, **closed as user-side**: blocking work inside the user's own async SSE endpoint —
  worth re-checking that nothing in our gateway/middleware does sync work in the stream path).
- Env var `VLLM_OUTPUT_PROCESS*` does not exist (0 hits).
- The **`PleOffloadWorker: Duplicate PLE request for dp_rank=0; skipping duplicate`** message is not in
  upstream vLLM source or any issue/PR text. Given `dp_rank` naming, it comes from the fork's
  DP-aware PLE worker. **UNVERIFIED but it is the single most on-point custom signal**: if the fork's
  duplicate-skip drops a request's PLE input, the corresponding sequence can hang EngineCore-side
  (never finishing prefill/decode) while the frontend still lists it running — and a per-round retry
  would re-emit the log roughly every round (~4 min cadence matches the observed interval). Check the
  fork's `ple_offload/worker.py`/`connector.py` duplicate-skip logic and whether "skipping duplicate"
  leaves the GPU-side forward waiting on a signal that will never fire.

## 7. Testable workarounds / flags (ordered by expected value)

1. **`--no-async-scheduling`** — cheapest A/B; directly fixes the #57562 underflow family, is implicated in #40926 (auto-enabled, silent stall), and removes the `max_concurrent_batches==2` condition that breaks the PLE connector single-slot assumption (#53960/#53899). Caveat: #53726 commenters found `resolve-as-on` behavior — **verify it actually took effect** (`async_scheduler` in logs, `prepare_inputs_event` presence) because passing nothing still means ON in 0.26/0.27 era.
2. **`cudagraph_mode=NONE`** (or at least avoid PIECEWISE/FULL_DECODE_ONLY) — kills the documented PLE one-step-ahead semaphore bug (jschmied, #53899) if graphs are on; also the #53726-class workaround pattern (MTP proposer at NONE) though ours is MTP0.
3. **Verify the PLE connector in-tree against upstream fixes** `95dc96d1d012` (uniproc spawn — likely N/A at TP=4) and **`4e8b849b8d97`** ("fix PLE offload for async MRV2 scheduling": D2H copies on main model stream + per-batch events). If absent: backport, or set the connector queue/events per-batch. Search the fork for `queue.Queue(maxsize=1)` in `vllm/v1/ple_offload/connector.py`. Maybe also inspect `peakcrosser7/vllm#13` (semaphore reset) and `davidtai` TP1 startup barrier PR.
4. **Isolate the offload path**: run with `VLLM_PLE_CPU_OFFLOAD=0` (table in VRAM if it fits; or `VLLM_PLE_MMAP` — #54129, itself ~16% slower on ROCm per davetha) and compare the c=12 round times; also try `--distributed-executor-backend mp` explicit (TP=4 should already be mp, but confirms).
5. **Instrument before/while reproducing**: PR #49628 "[V1][Observability] opt-in diagnostics for stalled engine stages" (unmerged; describes exactly "EngineCore stays alive but stops making progress ... requests hang without evidence"), #55700 watchdog PR, #36130 request watchdog (aborts stuck requests), #46121 IPC microbenchmark. Meanwhile: VLLM_LOGGING_LEVEL=DEBUG on the PleOffloadWorker/EngineCore; count "Duplicate PLE" occurrences per 4-min window and correlate with the stall start; log timestamps of `_copy_cuda_inputs`/`_d2h_done_event` waits.
6. **CPU-side "idle" is not proof the GPU side is idle** (xexex7's lesson): during a stall capture xpu util / a second py-spy with `--duration` while also checking whether workers are inside a SYCL launch (`eager` submittor) rather than truly parked; `CUDA_LAUNCH_BLOCKING`-equivalent on XPU (`ZE_DEBUG`? `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE`? — UNVERIFIED; XPU has no direct equivalent documented here) can expose GPU spins.
7. **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** — runs EngineCore in the APIServer process, eliminating the EngineCore↔frontend ZMQ/shm hop entirely (if the fork honors it). This is the direct test of the "ZMQ message loss / async_llm.py output handler region" hypothesis. (Docs-backed env, exists in v1; **verify honored in fork**.)
8. **XPU stack refresh**: #53480 shows vllm_xpu_kernels 0.1.8.2 → 0.1.14.1 + vllm-xpu 0.29 image resolved c=12 corruption on the same card. Check bundled `vllm_xpu_kernels`/`triton_xpu`/`torchxpu` versions vs upstream XPU wheels; time-permitting, A/B the same workload on an up-to-date `vllm/vllm-openai-xpu` image (custom PLE offload is fork code, so only as a diagnostic split).
9. **Scheduler-side knobs**: `--max-num-seqs 8` (below the cliff — c=8 is already OK), `--max-num-batched-tokens` behavior around hybrid prefills (#16916-style serialization on hybrid), and the (experimental, CUDA-era) prefill admission limits tracked in #57413. Also confirm `mamba_cache_mode`/`mamba_ssm_cache_dtype` — align mode charging is what #55533/#55617 implicate.

## 8. Verdict / closest 3 (with why/why-not)

No upstream issue matches the whole signature. Priority order for local reproduction:

1. **#53960 / #53899 (PLE offload connector race under async scheduling, fix `4e8b849b8d97`)** — fits: same custom feature, same V2 runner, same async-scheduling interaction; our tree may predate the fix (feature is in an unmerged PR — release builds cannot carry fixes). Doesn't fit perfectly: reported as a hard hang, ours partially recovers in waves; and our stall shows CPU idle, not a busy GPU. **Test: `--no-async-scheduling`, grep connector.py for single-event/maxsize=1.**
2. **#40926 (multiproc_executor `collective_rpc`/FutureWrapper → shm_broadcast spin; hybrids only; no-MTP)** — fits: exact frame family (`multiproc_executor.py` shm_broadcast wait), engine-idle + requests-running signature, 0.26/0.27 still affected, hybrid MoE+linear-attention is the distinguishing factor (we are one). Doesn't fit: their traces end in `sample_tokens` RPC timeout/EngineDeadError (~30s), ours self-recovers after ~4 min; and ours is XPU (theirs CUDA, though the code is backend-neutral).
3. **#57562 (async-scheduler placeholder underflow, chunked prefill + concurrency, no spec decode, regression from 0.24.0)** — fits: our exact option set (no MTP, chunked prefill likely, hybrid GDN, 12-way concurrency), and `--no-async-scheduling` = 24/24 complete. Doesn't fit: it asserts-dies in EngineCore rather than stalling, and we lack sampled-token-during-prefill evidence; also our 0.26.1rc1 predates 0.29.0 where it was reproduced. The *family* (scheduler/runner accounting for hybrid + concurrency under async scheduling) is the best-supported common thread with #40926 and #53960.

Honest caveats: (a) we could not inspect the recipe fork; "Duplicate PLE request" and our exact behavior are outside upstream evidence — call the fork's duplicate-skip path **top suspect for custom code, UNVERIFIED upstream**. (b) #54559 and #57423 deal with launch-phase broadcast timeouts, listed as signpost evidence only. (c) No evidence of ZMQ message loss between EngineCore and APIServer was found upstream; the IPC-path suspicion remains untested upstream — use `VLLM_ENABLE_V1_MULTIPROCESSING=0` A/B instead of assumption.

## Sources (all GitHub, fetched 2026-09-19)
- PR #53899 (PLE-Offload; comments incl. jschmied 09-06 one-step-lag, ssubbotin 09-17 memory blind spot, DONGRYEOLLEE1/dolf3131/estrella159/xexex7/davidtai/peakcrosser7)
- Issue #53960 (warmup deadlock thread; jschmied 3-stall table; xexex7 custom-all-reduce/PDL caution; alansrobotlab2 ptrace)
- Issue #40926 (collective_rpc root cause; ARSblithe212 async-scheduling; nichdiekuh hybrid-only; klapom negatives)
- Issues #55533 + PR #55617; #57562; #57680 (0.26→0.29 throughput drop, inconclusive non-repro); #53726; #53480 (+ bryanvine 09-15 v0.29.0 fix); #41663; #57535
- Issues #54559, #57423/#57635, #44185, #46121, #49628, #55700, #36130, #17385, #17972, #27194, #36826 (closed user-side), #18431, #45388 (fixed by #44560, KV-connector strand), #42371
- Related PRs: #53896, #54371 (UVA), #54129 (mmap), #54431 (ragged PLE IDs), #54525, #54709, #54722, #55375, #54076, #55390, #37429, #42960, #51287 (disable async scheduling when VLLM_BATCH_INVARIANT=1)

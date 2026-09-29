# Measured results — 4x Intel Arc Pro B70, Qwen3.8-Flash-Next W4A16

All timestamps UTC. Platform: 4x Intel Arc Pro B70, container `qwen38-flash-next`, endpoint `:8022` (`qwen-256k`). Weights: `devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d`. Image: the build of this repo's `docker/Dockerfile` (measured reference digest `sha256:15a806fc7367a44f6ab66d42e9f1b237fbb7431f4e913eb9ea197505f8d8417a`).

---

## 1. MNS 32 is the verified operating point

- **32x600 sustained: 1,038.3 tok/s** — `scripts/soakfix.py`, agg = Σcompletion_tokens÷round_wall per round, r1 warmup discarded, sustained_agg = MEAN r2..rN. Gate was ≥ 900: **passed** (2026-09-25).
- MNS ladder for reference (same harness): n=2 95.3 / n=4 179.9 / n=8 335.2 / n=16 629.1 tok/s; 32 (1,038.3) is the knee value at MML 262144.

## 2. P2P lane closed — measured zero

- `CCL_TOPO_P2P_ACCESS=1`: **1,038.2 vs 1,038.3 tok/s (n=32x600, Δ=0.1 = noise). Dead lever.**
- Operational law learned on the way: the passthrough must default **UNSET**, never `0` — explicit `0` forces oneCCL down the fd-exchange path whose drmfd mint is broken on this platform
  (`ze_handle_manager.cpp:43 mem_to_ipc_handle: device_fd is invalid value` at worker-init all_reduce). Absent (= engine auto) and `1` both boot clean.

## 3. MTP speculation lane closed — measured negative

**Method:** reference-faithful implementation of `get_mtp_target_hidden_states` for Qwen4Exp — the drafter consumes the **wide hyper-connection residual** `[T, hc=4, n_embd]` (fork reference `qwen4_exp_mtp.py` L15–17, L26, L184–190; hook `qwen4_exp.py:1170`, buffer filled at 957–959). Fidelity gates all numbers: temp-0, token-for-token vs banked serial baseline on long output / multi-turn+prefix-cache / long-context.

**Three fork bugs fixed before the engine ever ran MTP on this arch** (all image-baked, grep-asserted):

| Bug | Site | Fix |
|---|---|---|
| 1. Unguarded vision attr | `llm_base_proposer.py:1405` reads `config.image_token_index`; Qwen4Exp has `image_token_id` | guarded-attr patch `experimental/patches/es-mtp-image-token.patch` |
| 2. Buffer allocation gate | `qwen4_exp.py:839–844` allocates `_mtp_hidden_buffer` only for `use_eagle()/uses_draft_model()`; resolved method `'mtp'` misses both → narrow 2560 flowed into 10240-wide drafter buffers | gate widened to recognize method `mtp`/`qwen4_exp_mtp` |
| 3. Hook on wrong class | Checkpoint `architectures=['Qwen4ExpForConditionalGeneration']` (multimodal wrapper); hook lived only on inner `Qwen4ExpForCausalLM`; runner getattr fell through → None | delegating hook on the wrapper; combined patch `experimental/patches/es-mtp-qwen4exp-mtp.patch` |

**Final blocker + resolution:** first generation request died in the fork's XPU gated-delta-net native op — `causal_conv1d does not support spec-decode and non-spec tokens in the same invocation; mutually exclusive`. Setting `disable_padded_drafter_batch=true` in the speculative config routed spec batches through the non-padded path and cleared it for single-stream (the op accepted the invocation).

**Measured results (single-stream, k=1, W4A16 + PLE weights):**

| Metric | Value | Baseline / note |
|---|---|---|
| Draft acceptance | **47.4%** (772/1628, k=1) | public Arc-range claims: 74–94% |
| Single-stream prose | **49.5 tok/s** (r2–r4: 49.5/49.5/49.6) | serial 52.5 → **MTP is 5.7% SLOWER** |
| Fidelity (temp-0, long output) | **DIRTY** — diverges from paragraph 1 | gate requires token-for-token |
| n=8 concurrency | **all requests errored** (spec-batch rejection) | spec decode unusable multi-tenant |

**Ceiling math (why the lever was capped regardless of implementation quality):** at ~30 ms per MTP composite step, 1,000/30 = 33.3 steps/s; throughput = steps/s × (1 + accept). Break-even vs 52.5 serial needs accept ≥ 57.5%; **even 90% acceptance tops out at 63.3 tok/s**. The 30 ms model is validated by the measurement: 33.3 × 1.474 = 49.1 predicted vs 49.5 measured. At the measured 47.4% acceptance the arithmetic guarantees a loss.

**Cause of the low acceptance: UNRESOLVED on current evidence.** Candidates not excluded: W4A16 quantization degrading the residual signal the draft head consumes, hook/semantics still imperfect vs the reference contract, draft-head training mismatch. Not attributed without evidence.

**Verdict:** spec decode stays **OFF** — doubly confirmed (accept-collapse-under-concurrency research 46–56%; measured 47.4% single-stream net-negative on these weights). Patches preserved in [`experimental/patches/es-mtp-image-token.patch`](../../experimental/patches/es-mtp-image-token.patch) and [`es-mtp-qwen4exp-mtp.patch`](../../experimental/patches/es-mtp-qwen4exp-mtp.patch); the A/B harness pattern is described above; the default build does not include them.

## 4. Single-stream 52.5 verified three independent ways

| Instrument | Result |
|---|---|
| Banked (host-local, `scripts/single-stream.py`, median of 19, N=20) | 52.5 tok/s |
| Fresh client over the network, banked harness (1200-tok essay, temp 0, r1 discarded) | **mean 52.55 / median 52.56** (r2–r4: 52.56/52.43/52.65), 0 running / 0 waiting throughout |
| vLLM `Engine 000` lines during a sustained 1200-tok stream | gen=53.3 and 52.4 on full 10 s decode ticks; client-side stream total 52.7 |

**Gauge artifact, named:** `Engine 000` lines are **10-second windowed averages**. Requests starting/ending mid-tick (the `gen=9.6`/`4.8` class), short replies that finish inside one tick, and the watchdog's 1-token liveness probes (`gen=0.1 prefill=5.3 run=0`, every 60 s) all read far below sustained decode. Reading rule: only ticks with `run ≥ 1` quote the decode rate; boundary ticks divide partial tokens by the full window.

## 5. Decode rate is flat across context — kills the "slow with big context" theory

Streaming sweep, idle engine, temp 0, decode-only rate (TTFT excluded):

| Prompt tokens | TTFT | decode tok/s |
|---|---|---|
| 31 | 0.04 s | 53.6 |
| 8,685 | 1.72 s | 52.7 |
| 17,355 | 2.03 s | 52.4 |
| 34,695 | 0.33 s (prefix-cache hit) | 51.5 |

Spread 4% from 31 to ~35K tokens — attention cost per step is not visibly bending in the operating range. Perceived slowness at long context is TTFT (1.7–2 s at 8–17K) plus the windowing artifact above, not decode rate.

## 6. Future work (not executed): v0.1.15 kernel bump — applicability split

From the vllm-xpu-kernels v0.1.15 release notes (wheel `0.1.15.4` on PyPI, abi3, no declared torch pin — runtime import check happens inside the window):

- **Applies to B70/Xe2:** fused top-k/top-p sampler with per-row RNG state (removes the fallback logged at startup: `topk_topp_sampler does not support per-request generators. Falling back to PyTorch-native implementation`); widened `act_and_mul` vectorization (every MoE layer, every step); MoE negative-expert-ID routing fix (we run EP). Xe2 BlockFP8 grouped-GEMM listed but our backbone is W4A16 → N/A.
- **Xe3P-only (headline, not our hardware):** Xe3P chunk-prefill + paged-decode attention kernels behind `VLLM_XPU_ENABLE_XE3P`; eager-path LayerNorm fusions (our decode runs captured graphs).
- **Expected, honestly:** mid-50s single-stream if the fused sampler saves 1–2 ms of the ~19 ms serial step (1000/52.5); aggregate likely unchanged. Plan: pre-build + wheel-import check → /health gate → fidelity A/B token-for-token vs banked serial base → n=1x4 (≥52.5) and n=32x4 (≥1,038.3) → rollback = previous image (2 min) on any gate failure. No spec decode, no quantization change.

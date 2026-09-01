# Lane 4 evidence pack — TTFT (188 s @ 4K): what the lab already knows

Compiled 2026-09-01 by EdgeQuant from /tmp/b70lab. Every claim cites file+line.
This pack feeds the Lane 4 spec; the spec writer must verify citations and may
dig deeper, but these numbers are the measured floor.

## 1. The smoking gun nobody has swept: chunk size = 64

Every Flash-Next launcher in the lab freezes `max_num_batched_tokens=64`:

- `experiments/qwen38-flash-next-fp8-b70/tools/launch-tp4-ep4-eager-mtp1-long-context-base.sh:346,384,441,476` — `max_num_seqs=1, max_num_batched_tokens=64`, asserted (`assert config.scheduler_config.max_num_batched_tokens == 64` at :384), passed `--max-num-batched-tokens 64` at :476
- Same 64 in `launch-tp4-ep4-eager-mtp0-vision-512-base.sh:385,429,525`, `run-tp4-mtp0-current-vision-a8-client.sh:112,255`, `run-tp4-mtp0-current-vision-a9-client.sh:112,255`, `run-tp4-mtp0-16512-sampler-native-a6.sh:53`, `run-tp4-mtp2-16512-scheduler32-client-a2.sh:84,85` (that one uses 32)

At 4096 prompt tokens / 64-token chunks → **64 serialized full-model scheduler
steps for one prefill**, each carrying the fixed per-step host overhead
(scheduling, PLE UVA gather, per-layer collectives at tiny token count).
The GLM-5.3 DGX-Spark precedent (Ryan's own box): `MAX_NUM_BATCHED_TOKENS=2048`
was required; 4096 measured worse; 8192 crashed the QSA indexer. Same knob,
same indexer class → the sweep is the first receipt, with an indexer-crash watch.
NOTE: **no chunk-size sweep exists in the lab data** — searched tools/ + notes/;
`max_num_batched_tokens` appears only at the frozen 64/32 values. This is an
untested axis, not a failed one.

## 2. A28 target-step profile (2026-08-30) — the only bucket-level timing

`experiments/qwen38-flash-next-fp8-b70/data/20260830-tp4-mtp0-a28-target-step-profile-result.json`
(status: report-only, profile capture positive; timing permanently ineligible
for speed credit; identity vllm_head `d14396e2…`, stage `2f829747`, MTP0, graph
off, max_model_len 4352, ple_only_synchronous_uva).

DECODE-cycle device buckets, mean ms per retained cycle across rank means:
- routed_shared_moe **26.084** ← largest non-collective
- dense_projection 10.031 raw / **7.575 clean** (rank-3 outlier audit, clean median 7.579)
- quantization_cast 4.185
- elementwise 2.943
- qsa 2.208
- other_noncollective 1.698 · moe_router 0.789 · gdn 0.409 · device_mem_op 0.163 · normalization 0.051 · **ple 0.0019**
- collective_allreduce_distorted 197.065 (Kineto-inflated — diagnostic only)

Per-cycle multiplicity: **97 BF16 allreduces**, 532 GEMMs, 96 fused_moe, 86 qsa
events, 12 qsa_sparse_splitk, 72 gdn events (36 causal_conv + 36 gated_delta_rule).

Lab's own interpretation fields (verbatim):
- `primary_bottleneck_class = collective_critical_path_and_cross_rank_arrival_imbalance`
- `strongest_concrete_noncollective_target = the production M1 routed-MoE path and dense projections`
- `not_supported_as_primary_speed_targets = ['GatedDeltaNet', 'QSA', 'PLE lookup']`

Caveats that matter for Lane 4: this profiled **decode cycles** (retained
execute_contexts), NOT the prefill phase. TTFT was never bucket-profiled —
A16/A17 were digest-stability traces (149 tensor digests, no timing credit:
`notes/2026-08-30-tp4-mtp0-4352-ple-only-a16-late-prefill-trace-result.md`,
"report-only path completed without a model fault", battery failed at exact-4K
repeat row 2, first diff at token 62, 66 positions differ).

## 3. Measured TTFT anchors (results/qwen38-flash-next-fp8-b70/README.md)

- MTP3 exact-4K: **TTFT 187.90 s** (three p4096/o256 rows, 15.502 tok/s after first text) — §"Exact 4K"
- MTP1 exact-4K: TTFT median **232.079 s**, formal p4096/o128 row **317.105 s** (README ~line 304) — NOTE the spread: same-ish config family, 188→317 s ⇒ TTFT is window/boot-sensitive, not a stable constant
- PE 8K formal cell: TTFT **386.5 s** (scaffold §2B)
- 512-ctx screens: TTFT 10.0–11.8 s (MTP2/3/4 rows above) — i.e. ~11 s even for a 317-token needle prefill ⇒ there is a large FIXED per-request/per-step overhead independent of prompt length
- Decode cycle arithmetic: 5.515783 tok/s MTP0 ⇒ ~181 ms/cycle; buckets above sum ≈ 46 ms non-collective ⇒ collectives/host gaps dominate decode too (consistent with the lab's primary_bottleneck_class)

## 4. QSA structure (from merged PR #53896 kernel source, /tmp/qsa_ops.py)

- `_qsa_mqa_paged_kernel` — paged MQA logits over visible blocks
- Logits workspace chunking: `_LOGITS_WORKSPACE_BYTES = 128*1024*1024`;
  `rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns*4,1))`
  (line 753) — a host-computed chunk loop `for row_start in range(0, rows, rows_per_chunk)`
  (line 761): **rows = prompt length here ⇒ prefill issues (rows / rows_per_chunk)
  sequential kernel launches with a device workspace alloc per chunk** — the
  8K/16K instability and indexer-crash class lives here
- `_TOPK_WORKSPACE_BYTES = 1 MiB`
- Line 873 comment: "Narrow tiles favor decode; wide tiles improve throughput for prefill"
- These kernels are Triton, merged upstream; the lab overlay may still route the
  checkpoint's QSA through patched equivalents — identity check is a Lane 4 receipt.

## 5. What Lane 4 must produce (evidence summary)

1. **Prefill-phase bucket profile** — extend the A28 machinery (Kineto rank
   traces + base-time normalization `(raw_anchor_ns - baseTimeNanoseconds)/1000`
   + bucket classifier) to a p4096 prefill step. The corrected analyzer exists
   (sha a4d4c54c…, 5/5 tests, `offline_contract_failure` block above) — reuse it,
   do NOT rewrite the unit conversion (the frozen summarizer bit exactly this).
2. **Chunk sweep** — 64 (baseline) → 512 → 2048 → 4096 `max_num_batched_tokens`
   with indexer-stability gate; expect the GLM-5.3 shape (optimum below max).
   Cache-zero, exact p4096 fixture, TTFT + output-hash parity each cell.
3. **Fixed-overhead decomposition** — the ~11 s TTFT at 317-token needles vs
   188 s at 4096 ⇒ linear fit TTFT = F + c·prompt; if F is tens of seconds,
   the fix is per-step host work (PLE sync, scheduler, D2H), not chunk size.
4. **Workspace-chunk count receipts** — rows_per_chunk at columns seen here,
   predicted launch counts at 4K/8K/16K, matching observed instability boundary.

## 6. Related negatives (do not re-run blind)

- A26/A27 async UVA prefetch + moe-warps8: endpoint-negative (`notes/2026-08-30-tp4-mtp0-a26-async-uva-endpoint-negative.md`, `...a27-moe-warps8-endpoint-negative.md`)
- Event-chain A1 combined clone-elision + same-queue XCCL: device-lost all 4 ranks (caused the current boot block)
- A30 hc-grouped-m1: −1.82% speed-gate fail
- A31 M1-only warps-8 MoE: preregistered, FROZEN, blocked on reboot — it is the
  rig's first scheduled MoE test and overlaps Lane 3, not Lane 4.

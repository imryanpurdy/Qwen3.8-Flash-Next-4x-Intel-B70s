# 2026-09-19 — Sync-class exposure ranking: 155 `.cpu()/.item()/.tolist()` hits in vLLM 0.26.1 XPU (post v3 + gdn:546 fix)

## 1. Scope

Input: `flashnext-scout/synscan_raw.txt` — 155 grep hits of `.cpu()/.item()/.tolist()` under
`vllm/v1/{worker/gpu,attention,ple_offload}`. Local source for context: `v3-src/model_runner.py`
(2176 lines), `v3-src/connector.py`. All other files are **not available locally** — rows are
triaged from the line text + naming convention (`_np`/`_cpu` = host mirror), and every row that
cannot be settled from that is marked NEEDS-FILE with the Sec.7 on-rig check that settles it.
**No rig touch, no edits, no commit — this doc only.**

Tally: CAPTURE: 30; BG-THREAD: 0; PER-STEP: 25; BOOT: 1; OOS: 88; DOC: 11; — live-stack capture-conditional: 30; per-step: 25;
background-thread: 0; OOS/doc: 100.

## 2. Failure model — what wedges vs what only stalls

Confirmed class A (capture): any device→host sync **inside** `build_for_cudagraph_capture` is
an L0 `appendUSMMemcpy` racing graph-capture command-buffer appends → wedges the command-list
manager (gdn_attn.py:546 on boot 4010612) or hard-crashes the device (`UR_RESULT_ERROR_DEVICE_LOST`).

Confirmed class B (foreign thread): any L0 op submitted from a thread other than the runner
(connector thread `copy_` → 27/27 dumps with `appendUSMMemcpy`; fixed by v3 host-staging, 0/170).

Class C (same-thread per-step): `.cpu()/.item()` on the runner thread before/after replay submit
is **serial** with the submit — it cannot self-contend. It is the frequency term (per-step D2H
tail) and the drift term (the codebase keeps the pattern alive for class A/B to reappear at).

**Triage rule used:** SAFE = host tensor (`_np`/`_cpu`/`from_numpy` provenance) → host math, no L0.
SUSPECT = device tensor + blocking sync (class A/B/C depending on phase). NEEDS-FILE = device-ness
or phase not decidable from the line; leaning noted in the row.

## 3. Reachability filter (what this rig actually binds)

Live: GDN attn builder (mamba_hybrid model state, 2 GDN refs in qwen3_5.py), the recurrent/mamba
attn group (**mamba_attn vs mamba2_attn binding unconfirmed — Sec.7 check #4**), flash_attn (MM
encoder **only** — model is text-only, so effectively dead), input-prep path (`attention/ops` +
`attention/backends/utils.py` shared helpers), sampler/output/async_utils, dp_utils per-step.
Capture is ON, MTP spec decode, prefix caching ON, chunked prefill ON, VLLM_USE_V2_MODEL_RUNNER=1.

Dead on this stack: cpu_attn, rocm_*, mla/*, flashinfer, turboquant*, flex_attention, amx,
vit/MM wrappers, mm_* / encoder_runner / lora (no MM), pooling_runner, encoder_decoder,
eagle/dflash/autoregressive speculators, short_conv_attn (**0 ShortConv refs**), batch-sharded
sampling (config-gated off; hits are SAFE anyway), LoRA paths.

## 4. Master triage (155/155 rows)

#### A. CAPTURE-class (executes inside build_for_cudagraph_capture / capture prewarm on the live stack, or is one bind/config flip from it)

`cond` = reachability conditional (backend bind / config). These are the rows that can reproduce the gdn_attn.py:546 wedge; every such line needs the Sec.7 py-spy confirmation before it is considered clean.

| # | loc | snippet | triage | group | note |
|---|---|---|---|---|---|
| 44 | `attention/backends/mamba_attn.py:292` | this_num_computed = num_computed_tokens_p_cpu[req_idx].item() | **NEEDS-FILE** | CAPTURE | *_p_cpu suffix => host mirror by convention (same lineage as query_start_loc_cpu used to fix gdn:546). If host: SAFE per-step+capture. If mislabelled device: per-req D2H every step + capture wedge. 1-min on-rig confirm (Sec.7). |
| 45 | `attention/backends/mamba_attn.py:294` | query_start_loc_p_cpu[req_idx + 1].item() | **NEEDS-FILE** | CAPTURE | same as :292. |
| 46 | `attention/backends/mamba_attn.py:295` | - query_start_loc_p_cpu[req_idx].item() | **NEEDS-FILE** | CAPTURE | same as :292. |
| 48 | `attention/backends/mamba_attn.py:468` | if torch.any(prefill_to_decode).item(): | **SUSPECT** | CAPTURE | RANK #1. prefill_to_decode = device bool mask (no _cpu suffix). Metadata build for the hybrid recurrent group; same build_attn_metadata chain that PROVED capture-reachable for gdn (546 wedge stack: mamba_hybrid.py:327 -> attn_utils.py:314 -> build_for_cudagraph_capture). If mamba group is in the capture loop -> next capture wedge; else per-step D2H on every mixed prefill/decode step (chunked prefill ON). |
| 49 | `attention/backends/mamba_attn.py:644` | write_pos_cpu.to(torch.int32).tolist(), | **SAFE** | CAPTURE | explicit _cpu host tensor; .to(int32) is host op. Safe even in capture. |
| 50 | `attention/backends/mamba_attn.py:649` | is_flush_cpu.tolist(), | **SAFE** | CAPTURE | host *_cpu. |
| 51 | `attention/backends/gdn_attn.py:183` | total_tokens = int(prefill_query_start_loc_cpu[-1].item()) | **SAFE** | CAPTURE | host mirror (prefill_query_start_loc_cpu). Runs in prepare_chunk_indices per-step AND during capture - host math, no L0. |
| 53 | `attention/backends/gdn_attn.py:236` | num_spec_decodes = spec_sequence_masks_cpu.sum().item() | **SAFE** | CAPTURE | host *_cpu mirror. |
| 54 | `attention/backends/gdn_attn.py:239` | or num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item() | **SAFE** | CAPTURE | host mirrors. |
| 55 | `attention/backends/gdn_attn.py:271` | num_decodes = (non_spec_query_lens_cpu == 1).sum().item() | **SAFE** | CAPTURE | host mirror. |
| 56 | `attention/backends/gdn_attn.py:273` | num_zero_len = (non_spec_query_lens_cpu == 0).sum().item() | **SAFE** | CAPTURE | host mirror. |
| 57 | `attention/backends/gdn_attn.py:277` | non_spec_query_lens_cpu.sum().item() - num_decode_tokens | **SAFE** | CAPTURE | host mirror. |
| 58 | `attention/backends/gdn_attn.py:280` | query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens | **SAFE** | CAPTURE | host mirror. |
| 59 | `attention/backends/gdn_attn.py:296` | query_start_loc_cpu[-1].item(), | **SAFE** | CAPTURE | host mirror (fix token for 546). |
| 60 | `attention/backends/gdn_attn.py:321` | output_size=query_start_loc_cpu[-1].item(), | **SAFE** | CAPTURE | host mirror. |
| 62 | `attention/backends/utils.py:734` | first_extend = is_prefill_or_extend.int().argmax(dim=-1).item() | **SUSPECT** | CAPTURE | RANK #2. split-decodes/prefills helper; device bools + device query_start_loc. Called by backend metadata builders (GDN/mamba builds on stack - caller confirm Sec.7). per-step D2H each mixed step; CAPTURE if any capture build calls it. |
| 63 | `attention/backends/utils.py:735` | first_prefill = is_prefill.int().argmax(dim=-1).item() | **SUSPECT** | CAPTURE | same family. |
| 64 | `attention/backends/utils.py:737` | num_decode_tokens = query_start_loc[first_extend].item() | **SUSPECT** | CAPTURE | device query_start_loc indexing -> D2H. |
| 65 | `attention/backends/utils.py:811` | if query_lens[0].item() > decode_threshold: | **SUSPECT** | CAPTURE | RANK #3. device query_lens; chunked-prefill threshold/decode-passthrough decision, per-step; capture-cond. Fix: gdn already keeps query_lens_cpu (277-280) - read that. |
| 66 | `attention/backends/utils.py:834` | first_prefill = is_prefill.int().argmax(dim=-1).item() | **SUSPECT** | CAPTURE | twin of :734 (second splitter variant). |
| 67 | `attention/backends/utils.py:837` | num_decode_tokens = query_start_loc[first_prefill].item() | **SUSPECT** | CAPTURE | twin of :737. |
| 68 | `attention/backends/utils.py:858` | assert torch.all(seq_lens_cpu <= workspace_size).item() | **SAFE** | CAPTURE | seq_lens_cpu named host; assert is host tape check. Confirm suffix on rig (cheap). |
| 69 | `attention/backends/utils.py:862` | chunk_total + (s := seq_lens_cpu[i].item()) <= workspace_size | **SAFE** | CAPTURE | host loop over seq_lens_cpu - proves utils.py builds _cpu mirrors for this path. |
| 70 | `attention/backends/utils.py:1055` | nums_dict[BLOCK_M]["tot"] = nums.sum().item() | **NEEDS-FILE** | CAPTURE | nums device-ness unknown from line; workspace/block-size selection - per-step; capture-cond. Identify function on rig. |
| 72 | `attention/backends/mamba2_attn.py:44` | assert int(query_start_loc[0].item()) == 0, "query_start_loc[0] must be 0" | **SUSPECT** | CAPTURE | RANK #5 candidate. Device query_start_loc. Reachability CONDITIONAL on which backend binds the hybrid recurrent group (mamba_attn vs mamba2_attn - bind confirm Sec.7). If bound: assert+per-req syncs per step and in capture. |
| 73 | `attention/backends/mamba2_attn.py:48` | starts = qsl64[:-1].tolist() | **SUSPECT** | CAPTURE | same binding-conditional; qsl64 device. |
| 74 | `attention/backends/mamba2_attn.py:49` | ends = qsl64[1:].tolist() | **SUSPECT** | CAPTURE | same. |
| 75 | `attention/backends/mamba2_attn.py:50` | total = int(qsl64[-1].item()) | **SUSPECT** | CAPTURE | same. |
| 144 | `attention/ops/common.py:108` | Lmax = int(lengths.max().item()) | **NEEDS-FILE** | CAPTURE | RANK #4. attention/ops/common.py = prepare-kernel layer ('triton kernels for prepare' per stack context); function unknown from line. lengths likely device (cu_seqlens-like). Per-step; capture-cond. Identify function + device on rig. |
| 145 | `attention/ops/common.py:208` | N = int(lengths.sum().item()) | **NEEDS-FILE** | CAPTURE | same file/function question as :108. |

#### B. BACKGROUND THREAD

**0 hits.** The only known background-thread hazard (PLE connector thread `copy_` at connector.py:315/344 -> L0 USM memcpy) is fixed by v3 host-staging (numpy copies, connector.py:334-341) and its L0-free status is proven (0/170 dumps). It leaves no `.cpu()/.item()/.tolist()` in the scanned tree (it uses `event.synchronize()` at connector.py:447 - a host-side wait, not an L0 submit; covered in Sec.8 as out-of-sweep).

| # | loc | snippet | triage | group | note |
|---|---|---|---|---|---|

#### C. PER-STEP SERVING (runner thread, post/pre-forward, outside capture)

Single-threaded with the replay submit, so these block the loop (latency) but do not race it (not the wedge, absent a second submitter). They are the **frequency** term of the failure model and each one is one config flip away from being capture-reachable.

| # | loc | snippet | triage | group | note |
|---|---|---|---|---|---|
| 6 | `worker/gpu/sample/sampler.py:162` | cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None | **SAFE** | PER-STEP | cu_num_logits_np is host numpy (built in prepare_inputs, model_runner.py:1245-1247 np.cumsum). Runner thread, post-forward. Adjudicated in Sec.6. |
| 7 | `worker/gpu/sample/output.py:114` | counts = self.counts.cpu().numpy()[sampled_rows] | **SUSPECT** | PER-STEP | SamplerOutput host materialization, runner thread per step, outside capture. Device-ness of counts unconfirmed -> real D2H if device. Adjudicated Sec.6. |
| 8 | `worker/gpu/sample/output.py:119` | self.packed_mask.cpu().numpy()[sampled_rows], | **SUSPECT** | PER-STEP | same call site family as :114. |
| 9 | `worker/gpu/sample/output.py:130` | ).tolist(), | **NEEDS-FILE** | PER-STEP | target of tolist unknown from line; likely host after 114-119. Confirm callers on rig. |
| 10 | `worker/gpu/sample/batch_shard.py:262` | local_req_ids = [input_batch.req_ids[i] for i in local_req_indices_np.tolist()] | **SAFE** | PER-STEP | _np host. Batch-sharded sampling is config-gated (off by default), but SAFE either way. |
| 11 | `worker/gpu/sample/batch_shard.py:285` | num_logits_per_rank=num_logits_per_rank_np.tolist(), | **SAFE** | PER-STEP | host numpy. |
| 12 | `worker/gpu/sample/batch_shard.py:589` | global_batch.cu_num_logits_np.tolist() if num_logits != num_reqs else None | **SAFE** | PER-STEP | CuNumLogits_np host numpy (InputBatch field, model_runner.py:1388). |
| 13 | `worker/gpu/model_states/mamba_hybrid.py:249` | max_query_len = input_batch.num_scheduled_tokens.max().item() | **SAFE** | PER-STEP | InputBatch.num_scheduled_tokens is np.ndarray (model_runner.py:1367 <- numpy from gather_batch_req_state) -> host max. Also runs during capture (dummy make_dummy) - still host. |
| 14 | `worker/gpu/model_states/mamba_hybrid.py:255` | max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item() | **SAFE** | PER-STEP | seq_lens_cpu_upper_bound = torch.from_numpy (model_runner.py:1347) -> host. |
| 15 | `worker/gpu/model_states/default.py:190` | max_query_len = input_batch.num_scheduled_tokens.max().item() | **SAFE** | PER-STEP | host numpy; default model_state not bound on mamba_hybrid rig (dead on this stack). |
| 16 | `worker/gpu/model_states/default.py:196` | max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item() | **SAFE** | PER-STEP | host; dead on hybrid rig. |
| 23 | `worker/gpu/dp_utils.py:46` | if torch.all(num_tokens_across_dp == 0).item(): | **NEEDS-FILE** | PER-STEP | dispatch_cg_and_sync_dp, runner thread, BEFORE replay submit (execute_model model_runner.py:1605-1615). Device iff dp_size>1 branch (all_gather). DP size on rig unconfirmed (TP4; DP likely 1). Same-thread pre-replay -> latency, not wedge. |
| 24 | `worker/gpu/dp_utils.py:52` | synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item())) | **NEEDS-FILE** | PER-STEP | same function/branch question. |
| 25 | `worker/gpu/dp_utils.py:67` | synced_num_tokens = int(num_tokens_across_dp.max().item()) | **NEEDS-FILE** | PER-STEP | same. |
| 26 | `worker/gpu/dp_utils.py:78` | if bool(torch.all(max_query_lens_across_dp != -1).item()): | **NEEDS-FILE** | PER-STEP | same. |
| 27 | `worker/gpu/dp_utils.py:79` | synced_max_query_len = int(max_query_lens_across_dp.max().item()) | **NEEDS-FILE** | PER-STEP | same. |
| 28 | `worker/gpu/spec_decode/rejection_sampler.py:134` | cu_num_generated_tokens = cu_num_logits_np.tolist() | **SAFE** | PER-STEP | host numpy (same _np family). RejectionSampler only if num_draft_tokens>0 - MTP path may bypass; SAFE either way. |
| 29 | `worker/gpu/spec_decode/rejection_sampler.py:252` | cu_num_logits_np.tolist() if expanded_logits else None | **SAFE** | PER-STEP | host numpy. |
| 30 | `worker/gpu/spec_decode/utils.py:48` | draft_token_ids = self.draft_tokens_np.tolist() | **SAFE** | PER-STEP | DraftTokensHandler host mirror (_np). Runner thread in sample_tokens. |
| 32 | `worker/gpu/spec_decode/multi_module_mtp/speculator.py:161` | max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item() | **SAFE** | PER-STEP | MTP speculator prepare; seq_lens_cpu_upper_bound host (model_runner.py:1347). Runs per step and inside speculator.capture() (model_runner.py:979) with dummy - host either way. |
| 35 | `worker/gpu/async_utils.py:174` | sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist() | **SAFE** | PER-STEP | AsyncOutput host materialization; invoked AFTER copy_event.synchronize() (stream-ordered D2H on output_copy_stream). Runner thread. Adjudicated Sec.6 - CONFIRMED safe-by-design. |
| 36 | `worker/gpu/async_utils.py:175` | num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist() | **SAFE** | PER-STEP | host (_np) mirror, same path. |
| 37 | `worker/gpu/async_utils.py:187` | zip(self.model_runner_output.req_ids, self.num_nans.tolist()) | **SAFE** | PER-STEP | num_nans materialized after the same event sync; same thread. |
| 38 | `worker/gpu/async_utils.py:198` | if self._has_fault is not None and self._has_fault.item(): | **SAFE** | PER-STEP | fault-path check, same thread, post-sync. |
| 39 | `worker/gpu/async_utils.py:203` | f"Mask: {mask.cpu().tolist()}" | **SAFE** | PER-STEP | error-reporting path only, runner thread; not hot. |

#### D. BOOT-ONLY (load / one-time checks)

| # | loc | snippet | triage | group | note |
|---|---|---|---|---|---|
| 31 | `worker/gpu/spec_decode/eagle/utils.py:24` | return torch.equal(w.cpu(), target.weight.cpu()) | **N/A** | BOOT | EAGLE SVD weight check; EAGLE not used (MTP = multi_module_mtp). load-time/boot-only anyway. |

#### E. OUT-OF-SCOPE (unreachable backend / feature on this stack)

Dead code for this rig, listed for completeness (Qwen3.5-Flash-Next: hybrid text, no MM/ViT, no LoRA, no pooling, no encoder-decoder, MTP=multi-module not EAGLE/dflash, TP4+EP4, no ROCm/MLA/flashinfer/cpu_attn/amx, 0 ShortConv layers).

| # | loc | snippet | triage | group | note |
|---|---|---|---|---|---|
| 1 | `worker/gpu/mm/rope.py:82` | pos = prefill_positions[i].tolist() | **N/A** | OOS | MM/rope: dead (Qwen3.5 text-only; no MM config). If MM ever enabled -> encoder capture class. |
| 2 | `worker/gpu/mm/encoder_runner.py:229` | exclude_embeddings = is_decode.tolist() | **N/A** | OOS | MM encoder capture path; dead today. encoder_runner.capture() runs inside capture_model (model_runner.py:958-959) -> would be CAPTURE class if MM on. |
| 3 | `worker/gpu/mm/encoder_runner.py:231` | query_start = num_computed_tokens.tolist() | **N/A** | OOS | same as above. |
| 4 | `worker/gpu/mm/encoder_runner.py:232` | query_end = (num_computed_tokens + num_scheduled_tokens).tolist() | **N/A** | OOS | same as above. |
| 5 | `worker/gpu/mm/lora.py:91` | index_mapping=tuple(connector_token_mapping.tolist()) | **N/A** | OOS | MM+LoRA path; dead today (no MM, no LoRA on stack). |
| 17 | `worker/gpu/model_states/mm_pruning.py:66` | req_idx_list = input_batch.idx_mapping_np.tolist() | **SAFE** | OOS | MM-only model state; dead on text-only stack (_np host anyway). |
| 18 | `worker/gpu/model_states/mm_pruning.py:67` | prefill_lens_list = input_batch.prefill_len_np.tolist() | **SAFE** | OOS | same. |
| 19 | `worker/gpu/model_states/mm_pruning.py:68` | num_computed_list = input_batch.num_computed_prefill_tokens_np.tolist() | **SAFE** | OOS | same. |
| 20 | `worker/gpu/model_states/mm_pruning.py:69` | num_scheduled_list = input_batch.num_scheduled_tokens.tolist() | **SAFE** | OOS | same. |
| 21 | `worker/gpu/model_states/encoder_decoder.py:133` | max_query_len = input_batch.num_scheduled_tokens.max().item() | **SAFE** | OOS | encoder-decoder models only; dead on decoder-only hybrid. |
| 22 | `worker/gpu/model_states/encoder_decoder.py:138` | max_seq_len = int(seq_lens_cpu_upper_bound[:num_reqs].max().item()) | **SAFE** | OOS | same. |
| 33 | `worker/gpu/spec_decode/dflash/speculator.py:347` | max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item() | **SAFE** | OOS | dflash speculator not used (MTP on stack). |
| 34 | `worker/gpu/spec_decode/autoregressive/speculator.py:232` | max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item() | **SAFE** | OOS | autoregressive (EAGLE-style) speculator not used. |
| 40 | `worker/gpu/pool/pooling_runner.py:102` | req_indices = input_batch.idx_mapping_np.tolist() | **SAFE** | OOS | pooling runner not instantiated (generate model); _np host anyway. |
| 41 | `attention/backends/cpu_attn.py:219` | total_block_num: int = block_nums.sum().item() | **N/A** | OOS | CPU attention backend; not selected on this stack. |
| 42 | `attention/backends/cpu_attn.py:220` | max_block_num = block_nums.max().item() | **N/A** | OOS | same. |
| 43 | `attention/backends/triton_attn.py:220` | suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len | **SUSPECT** | OOS | triton backend NOT selected on stack (hybrid uses GDN; flash_attn MM-only). WOULD be per-step device sync every prefill with prefix-cache hit. Adjudicated Sec.6. |
| 71 | `attention/backends/rocm_attn.py:136` | suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len | **N/A** | OOS | ROCm backend; unreachable on XPU. |
| 76 | `attention/backends/mla/rocm_aiter_mla.py:275` | if num_decode_tokens <= int(qo_len.sum().item()): | **N/A** | OOS | MLA/ROCm; unreachable. |
| 77 | `attention/backends/mla/rocm_aiter_mla.py:295` | first_zero = int(zero_positions[0].item()) | **N/A** | OOS | unreachable. |
| 78 | `attention/backends/mla/rocm_aiter_mla.py:646` | total_prefill_tokens = int(qo_indptr_cpu[-1].item()) | **N/A** | OOS | unreachable. |
| 79 | `attention/backends/mla/rocm_aiter_mla.py:656` | num_partial_tiles = int(self.fp8_ps_reduce_indptr[-1].item()) | **N/A** | OOS | unreachable. |
| 80 | `attention/backends/mla/rocm_aiter_mla.py:683` | max_qo_len = qo_len.max().item() | **N/A** | OOS | unreachable. |
| 82 | `attention/backends/mla/rocm_aiter_mla.py:1450` | total = int(new_indptr[-1].item()) | **N/A** | OOS | unreachable. |
| 83 | `attention/backends/mla/sparse_swa.py:243` | chunk_max_compressed = int(compressed_lens_cpu[chunk_start].item()) | **N/A** | OOS | unreachable. |
| 84 | `attention/backends/mla/sparse_swa.py:244` | chunk_max_gather = int(gather_lens_cpu[chunk_start].item()) | **N/A** | OOS | unreachable. |
| 85 | `attention/backends/mla/sparse_swa.py:250` | int(compressed_lens_cpu[chunk_end].item()), | **N/A** | OOS | unreachable. |
| 86 | `attention/backends/mla/sparse_swa.py:254` | int(gather_lens_cpu[chunk_end].item()), | **N/A** | OOS | unreachable. |
| 87 | `attention/backends/mla/flashattn_mla.py:200` | max_query_len = query_lens_cpu.max().item() | **N/A** | OOS | unreachable. |
| 88 | `attention/backends/mla/indexer.py:129` | q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item() | **N/A** | OOS | unreachable. |
| 89 | `attention/backends/mla/indexer.py:140` | chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item() | **N/A** | OOS | unreachable. |
| 90 | `attention/backends/mla/indexer.py:687` | min_decode_len = int(decode_lens_cpu.min().item()) | **N/A** | OOS | unreachable. |
| 91 | `attention/backends/mla/indexer.py:721` | actual_expanded = int(decode_lens_cpu.sum().item()) | **N/A** | OOS | unreachable. |
| 92 | `attention/backends/mla/indexer.py:803` | actual_expanded = int(decode_lens_cpu.sum().item()) | **N/A** | OOS | unreachable. |
| 93 | `attention/backends/mla/indexer.py:824` | actual_expanded = int(decode_lens_cpu.sum().item()) | **N/A** | OOS | unreachable. |
| 94 | `attention/backends/mla/indexer.py:962` | max_decode_len = int(decode_lens_cpu.max().item()) | **N/A** | OOS | unreachable. |
| 95 | `attention/backends/mla/indexer.py:1091` | total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item() | **N/A** | OOS | unreachable. |
| 96 | `attention/backends/mla/indexer.py:1120` | local_total_seq_lens = int(local_cu_seq_lens[-1].item()) | **N/A** | OOS | unreachable. |
| 97 | `attention/backends/mla/indexer.py:1121` | max_local_total_seq_lens = int(local_seq_lens.sum(dim=0).max().item()) | **N/A** | OOS | unreachable. |
| 98 | `attention/backends/mla/indexer.py:1128` | (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item() | **N/A** | OOS | unreachable. |
| 99 | `attention/backends/mla/indexer.py:1160` | token_start = query_start_loc_cpu[start_idx].item() | **N/A** | OOS | unreachable. |
| 100 | `attention/backends/mla/indexer.py:1166` | token_end = query_start_loc_cpu[end_idx].item() | **N/A** | OOS | unreachable. |
| 101 | `attention/backends/mla/flashmla_sparse.py:394` | decode_query_len = (query_start_loc_cpu[1] - query_start_loc_cpu[0]).item() | **N/A** | OOS | unreachable. |
| 102 | `attention/backends/mla/flashmla_sparse.py:467` | offset = prefill_workspace_starts_cpu[chunk_start].item() | **N/A** | OOS | unreachable. |
| 103 | `attention/backends/mla/flashmla_sparse.py:471` | token_start = query_start_loc_cpu[num_decodes + chunk_start].item() | **N/A** | OOS | unreachable. |
| 104 | `attention/backends/mla/flashmla_sparse.py:472` | token_end = query_start_loc_cpu[num_decodes + chunk_end].item() | **N/A** | OOS | unreachable. |
| 105 | `attention/backends/mla/flashmla.py:165` | max_query_len = query_lens_cpu.max().item() | **N/A** | OOS | unreachable. |
| 106 | `attention/backends/mla/cpu_mla.py:214` | if not valid_mask.all().item(): | **N/A** | OOS | unreachable (cpu MLA). |
| 107 | `attention/backends/mla/amx_mla.py:161` | prefill.max_len_extend = int(extend_seq_lens.max().item()) | **N/A** | OOS | unreachable (AMX). |
| 108 | `attention/backends/flex_attention.py:984` | common_attn_metadata.seq_lens_cpu_upper_bound.max().item() | **N/A** | OOS | flex_attention backend not used. |
| 109 | `attention/backends/rocm_aiter_fa.py:502` | common_attn_metadata.seq_lens.cpu() | **N/A** | OOS | ROCm; unreachable. |
| 110 | `attention/backends/rocm_aiter_fa.py:512` | max_query_len=query_lens_cpu[:num_decodes].max().item(), | **N/A** | OOS | unreachable. |
| 111 | `attention/backends/rocm_aiter_fa.py:523` | max_query_len=query_lens_for_prefill.max().item(), | **N/A** | OOS | unreachable. |
| 112 | `attention/backends/rocm_aiter_fa.py:524` | max_seq_len=seq_lens[num_decodes + num_extends :].max().item(), | **N/A** | OOS | unreachable. |
| 113 | `attention/backends/rocm_aiter_fa.py:561` | fetched_shape = cu_seq_lens[-1].item() | **N/A** | OOS | unreachable. |
| 114 | `attention/backends/rocm_aiter_fa.py:570` | max_seqlen_k = swa_seqlen_for_extend.max().item() | **N/A** | OOS | unreachable. |
| 115 | `attention/backends/rocm_aiter_fa.py:571` | total_tokens = cu_seq_lens[-1].item() | **N/A** | OOS | unreachable. |
| 116 | `attention/backends/rocm_aiter_fa.py:585` | num_chunks = cdiv(computed_kv_lens.max().item(), max_context_chunk) | **N/A** | OOS | unreachable. |
| 117 | `attention/backends/rocm_aiter_fa.py:607` | cu_seq_lens_cpu[:, -1].max().item() if num_chunks > 0 else 0 | **N/A** | OOS | unreachable. |
| 118 | `attention/backends/rocm_aiter_fa.py:622` | max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(), | **N/A** | OOS | unreachable. |
| 119 | `attention/backends/rocm_aiter_fa.py:624` | total_token_per_batch=cu_seq_lens_cpu[:, -1].tolist(), | **N/A** | OOS | unreachable. |
| 120 | `attention/backends/rocm_aiter_fa.py:639` | max_query_len=query_lens_for_extend.max().item(), | **N/A** | OOS | unreachable. |
| 121 | `attention/backends/rocm_aiter_fa.py:640` | max_seq_len=seq_lens[num_extends_slice].max().item(), | **N/A** | OOS | unreachable. |
| 125 | `attention/backends/short_conv_attn.py:228` | .item() | **SUSPECT** | OOS | device sync in ShortConv build; backend dead on this model (0 ShortConv refs). If a ShortConv model is ever served: per-step + capture class (twin of gdn). |
| 126 | `attention/backends/short_conv_attn.py:260` | if bool(candidate_mask.any().item()): | **SUSPECT** | OOS | device bool sync; dead today; capture/per-step if served. |
| 127 | `attention/backends/short_conv_attn.py:310` | num_spec_decodes = int(spec_sequence_masks_cpu.sum().item()) | **SAFE** | OOS | host mirror; dead backend. |
| 128 | `attention/backends/short_conv_attn.py:311` | num_decodes = int(decode_mask_cpu.sum().item()) | **SAFE** | OOS | host mirror; dead backend. |
| 129 | `attention/backends/short_conv_attn.py:312` | num_prefills = int(prefill_mask_cpu.sum().item()) | **SAFE** | OOS | host mirror; dead backend. |
| 130 | `attention/backends/short_conv_attn.py:314` | num_prefill_tokens = int(query_lens_cpu[prefill_mask_cpu].sum().item()) | **SAFE** | OOS | host mirror; dead backend. |
| 131 | `attention/backends/short_conv_attn.py:316` | query_lens_cpu[spec_sequence_masks_cpu].sum().item() | **SAFE** | OOS | host mirror; dead backend. |
| 132 | `attention/backends/short_conv_attn.py:321` | int(query_lens_cpu[prefill_mask_cpu].max().item()) | **SAFE** | OOS | host mirror; dead backend. |
| 133 | `attention/backends/short_conv_attn.py:552` | num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu() | **SUSPECT** | OOS | DISCREPANCY: tracked twin fix (query_start_loc_cpu) but scan still shows the raw .cpu() - fix may not have landed in scanned tree (gdn's 546 counterpart IS gone; only comment at 547 remains). Exposure 0 today (0 ShortConv layers); re-wedge if a ShortConv model is served. Verify Sec.7. |
| 134 | `attention/backends/flashinfer.py:1080` | q_len_per_req = int(nonzero.max().item()) if nonzero.numel() > 0 else 1 | **N/A** | OOS | flashinfer backend not used on stack. |
| 135 | `attention/backends/flashinfer.py:1081` | uniform = nonzero.numel() <= 1 or bool((nonzero == nonzero[0]).all().item()) | **N/A** | OOS | unreachable. |
| 136 | `attention/backends/flashinfer.py:1088` | q_lens = decode_q_lens.tolist() | **N/A** | OOS | unreachable. |
| 137 | `attention/backends/flashinfer.py:1566` | max_q_len_prefill = int(query_lens_prefill_cpu.max().item()) | **N/A** | OOS | unreachable. |
| 138 | `attention/backends/flashinfer.py:1984` | layer._o_scale_float = output_scale.cpu().item() | **N/A** | OOS | unreachable (quant scale read). |
| 140 | `attention/backends/turboquant_attn.py:765` | qsl = attn_metadata.query_start_loc_cpu.tolist() | **N/A** | OOS | unreachable backend. |
| 141 | `attention/backends/turboquant_attn.py:767` | qsl = query_start_loc.tolist() | **N/A** | OOS | unreachable; note the device-speed alternate (would be per-step sync if served). |
| 142 | `attention/backends/turboquant_attn.py:769` | seq_lens_list = attn_metadata.seq_lens_cpu.tolist() | **N/A** | OOS | unreachable. |
| 143 | `attention/backends/turboquant_attn.py:771` | seq_lens_list = attn_metadata.seq_lens.tolist() | **N/A** | OOS | unreachable. |
| 147 | `attention/ops/vit_attn_wrappers.py:59` | max_seqlen = max_seqlen.item() | **N/A** | OOS | MM encoder attention only (flash_attn used for ViT/MM only); model has no ViT. |
| 148 | `attention/ops/vit_attn_wrappers.py:173` | max_seqlen = max_seqlen.item() | **N/A** | OOS | same. |
| 149 | `attention/ops/vit_attn_wrappers.py:277` | lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist() | **N/A** | OOS | same. |
| 150 | `attention/ops/vit_attn_wrappers.py:356` | max_seqlen = max_seqlen.item() | **N/A** | OOS | same. |
| 151 | `attention/ops/rocm_aiter_mla_sparse.py:505` | seq_len = int(context_lens[i].item()) | **N/A** | OOS | ROCm ops; unreachable. |
| 152 | `attention/ops/rocm_aiter_mla_sparse.py:542` | context_len_i = int(context_len.item()) | **N/A** | OOS | unreachable. |
| 153 | `attention/ops/rocm_aiter_mla_sparse.py:555` | max_context_len = int(context_limit.max().item()) | **N/A** | OOS | unreachable. |

#### F. COMMENT / DOCUMENTATION lines (not executable syncs)

| # | loc | snippet | triage | group | note |
|---|---|---|---|---|---|
| 47 | `attention/backends/mamba_attn.py:336` | `compute_num_computed_tokens().cpu()` would force. | **COMMENT** | DOC | doc comment only; documents the hazard, not a sync. |
| 52 | `attention/backends/gdn_attn.py:192` | # GPU->CPU sync (.tolist()) in prepare_chunk_indices. | **COMMENT** | DOC | leftover comment claiming a .tolist() in prepare_chunk_indices; NO live .tolist() hit in gdn_attn in this sweep -> either historical (removed with 546 fix) or different spelling (.numpy()/.to('cpu')). Grep on rig (Sec.7). |
| 61 | `attention/backends/gdn_attn.py:547` | # (num_accepted_tokens - 1).cpu() issued a blocking D2H USM | **COMMENT** | DOC | documents the fixed 546 site; line 546 itself is gone (no .cpu() hit) -> fix landed in scanned tree. |
| 81 | `attention/backends/mla/rocm_aiter_mla.py:1214` | # in-forward .item() sync that would prevent CUDA Graph capture. | **COMMENT** | DOC | doc comment; unreachable file. |
| 122 | `attention/backends/rocm_aiter_fa.py:678` | skip split_decodes_prefills_and_extends() and avoid all .cpu() / | **COMMENT** | DOC | doc comment; unreachable file. |
| 123 | `attention/backends/rocm_aiter_fa.py:679` | .item() calls that would otherwise break CUDA graph capture. | **COMMENT** | DOC | doc comment; unreachable file. |
| 124 | `attention/backends/short_conv_attn.py:66` | # packing buffer without a device->host sync (``lengths.max().item()``). | **COMMENT** | DOC | doc comment; ShortConv backend - 0 layer refs in qwen3_5.py -> dead on this model. |
| 139 | `attention/backends/turboquant_attn.py:763` | # otherwise `.tolist()` on GPU tensors forces a synchronizing copy. | **COMMENT** | DOC | doc comment; turboquant backend not used. |
| 146 | `attention/ops/vit_attn_wrappers.py:6` | `.item()` in flash attention) | **COMMENT** | DOC | doc comment; MM/ViT only. |
| 154 | `attention/ops/turboquant_soa/triton_turboquant_unified_attention.py:1115` | # track it from the block table). We avoid calling seq_lens.max().item() | **COMMENT** | DOC | doc comment; unreachable (turboquant). |
| 155 | `attention/backend.py:467` | If a CPU copy is needed, use `seq_lens.cpu()` instead. | **COMMENT** | DOC | base backend docstring; documents the rule, not a sync. |

## 5. Top-5 most dangerous sites (phase × thread × frequency)

Ranking rule: capture-reachable + foreign-thread first; per-step by frequency. Every row below is
on a **live-stack** call path (or one backend-bind away from one).

**1. `mamba_attn.py:468` — `if torch.any(prefill_to_decode).item():` (SUSPECT, capture-cond)**
- Tensor: `prefill_to_decode` — device bool mask, no `_cpu` suffix; built in the recurrent-group
  metadata path from device seq/queries.
- Why live: the hybrid's attn metadata builder is `mamba_hybrid.py:327 prepare_attn →
  attn_utils.build_attn_metadata → backend build_for_cudagraph_capture` — this exact chain is the
  **proven** capture path (546 wedge stack). If the mamba/recurrent group is built in the same
  loop (it is at least bound per-step), this line executes inside MTP1 capture.
- Failure: replica of 546 (boot capture wedge, spin in `appendUSMMemcpy`) or `DEVICE_LOST` if it
  lands against a replay; per-step (chunked prefill co-schedules prefill+decode, so most steps) →
  silent per-step D2H tail.
- Fix (same shape as gdn 546): `prefill_to_decode` is derivable on host — `BatchReqState`
  already carries `is_prefilling_np`/`prefill_len_np`/`num_computed_prefill_tokens_np`
  (model_runner.py:1169-1177). Pass a host bool (or `torch.from_numpy` of the padded req-order
  mask) into the mamba metadata prep; drop the device `.any().item()` entirely.

**2. `attention/backends/utils.py:734-737` (and twin `834-837`) — split decodes/prefills helpers**
- Tensors: device `is_prefill_or_extend` (bool) + device `query_start_loc`; `argmax(...).item()`
  + `query_start_loc[first_extend].item()`.
- Why live: shared metadata helpers; the GDN/mamba builds on this stack sit in the same
  `build_attn_metadata` family that reached capture for gdn. **Caller unconfirmed (Sec.7 #2)** —
  if any live backend build calls these inside a capture build, this is the next wedge.
- Failure: per-step D2H on every mixed batch (chunked prefill) — silent; capture-cond — wedge.
- Fix: these split decisions are host-known (`query_start_loc_np` InputBatch field,
  model_runner.py:1373; `is_prefilling_np`). Compute `first_prefill/first_extend/num_decode_tokens`
  from the numpy mirrors; the file already builds `seq_lens_cpu` mirrors (see :858/:862).

**3. `attention/backends/utils.py:811` — `if query_lens[0].item() > decode_threshold:`**
- Tensor: device `query_lens` (per-step derived), no suffix.
- Why live: chunked-prefill threshold / decode-passthrough decision in the same helper file;
  per-step whenever a prefill chunk is scheduled; capture-cond.
- Failure: silent per-step D2H; wedge if reached in capture. Fix mirrors #2: `query_lens_cpu`
  already exists in the same flow (gdn keeps one at 277-280) — index the host copy, or `int()`
  a numpy value.

**4. `attention/ops/common.py:108` / `:208` — `int(lengths.max().item())` / `int(lengths.sum().item())`**
- Tensor: `lengths` — likely device (cu-seqlens family); **function identity unknown (NEEDS-FILE)**,
  and this file is the "triton kernels for prepare" layer the stack context names as live.
- Why live: prepare-path; per-step for every batch that goes through the helper; capture-cond if
  `prepare_inputs_to_capture` calls it (capture builds go through `cudagraph_utils.py:683` → the
  same prep functions).
- Failure: silent per-step D2H (prepare is the hottest host phase); wedge if capture-reachable.
- Fix: feed the host numpy (query_start_loc_np / num_scheduled_tokens_np) instead of a device
  `lengths`; both already exist per-batch.

**5. `mamba2_attn.py:44-50` — assert + `qsl64` tolist/ends/total (capture-cond, binding-conditional)**
- Tensors: device `query_start_loc` (`assert int(query_start_loc[0].item())==0`, `qsl64[...].tolist()`).
- Why conditional: if the hybrid recurrent group binds **Mamba2AttnBackend** rather than
  MambaAttnBackend, this is the per-step + capture-during-build sync family — if it runs in
  capture, wedge; if per-step, per-step D2H.
- Failure: `DEVICE_LOST`-prone (assert lands on a value pulled from a device that may be in
  replay); wedge if capture-reachable.
- Fix: read `query_start_loc[0]` and qsl from the host `query_start_loc_np` mirror; keep the
  assert on host.

**Verified-clean (no action):** gdn_attn.py:183-321 (`*_cpu` host mirrors, SAFE — the 546 fix
pattern), mamba_attn.py:292-295/644-649 (`*_cpu` host mirrors by name; one confirm), the whole
async_utils block (Sec.6), sampler/batch_shard/spec competitors (`_np` host).

## 6. Adjudications

**triton_attn.py:220 — `common_attn_metadata.seq_lens.cpu() - common_prefix_len` (could it fire
per decode step with prefix caching on? what calls build() with common_prefix_len > 0?)**
Verdict: **cannot fire on this stack today; per-step sync if it ever does.**
- `common_prefix_len > 0` is produced by `build_attn_metadata` when prefix caching is ON and a
  scheduled batch has cached (already computed) prefix tokens — i.e., a **prefill chunk with a
  prefix-cache hit**, or a decode-pass segment after one. Prefix caching IS on and chunked prefill
  IS on here, so such batches occur — but they are consumed by the **GDN** backend
  (`gdn_attn` prepares prefix state; its own `query_start_loc_cpu`/`seq_lens_cpu`-based counts at
  183/296/321). The triton backend is only built if `VLLM_ATTENTION_BACKEND`/backend selection
  picks `triton_attn` — not the case (hybrid → GDN builder; flash_attn MM-only).
- If the backend is ever flipped (or a code path routes hybrid metadata through
  `attention/backends/triton_attn.py`), line 220 becomes a runner-thread D2H per affected prefill
  step (silent tail), and — because triton builds also run under `for_capture` metadata prep — a
  class-A wedge candidate. Recommendation: leave as OOS, but note it as the **backend-select
  invariant**: GDN is the only live attention backend; any doc that introduces a second one must
  re-run this audit.
- Note `rocm_attn.py:136` is the identically-shaped twin (OOS, ROCm).

**async_utils.py:174-203 — confirm or refute the safe-by-design verdict.**
Verdict: **CONFIRMED safe-by-design, with three invariants that must keep holding.**
- Shape: `sample_tokens` builds `AsyncOutput(model_runner_output, sampler_output, ...,
  main_stream=self.main_stream, copy_stream=self.output_copy_stream)` (model_runner.py:1939-1947),
  with the explicit comment "Start async output copy here so that it can overlap with speculator
  proposal" (:1938) and "ensuring that `copy_event` is recorded before calling postprocess"
  (:1962-1965). So: D2H copies are issued on **`output_copy_stream`** (a dedicated side stream,
  created once at model_runner.py:208), ordered after the main stream via event waits; the host
  materialization at 174/175/187/198/203 is gated by `copy_event.synchronize()`.
- Why not the wedge: (1) **same thread** as every L0 submit — the engine/runner thread; (2) the
  copies are stream-ordered (event wait on main stream) so they do not race the device writers;
  (3) host access is behind the event sync, so no torn reads; (4) the next step's graph replay
  cannot start until the engine loop has consumed this output — no concurrent command-list append
  while the event wait spins.
- Refutation risks (each individually sufficient): (a) host materialization moved off the runner
  thread (e.g., into a PLE/sampler/pool thread) — then a blocking `event.synchronize()` from a
  foreign thread is exactly class B; (b) `max_concurrent_batches > 1` overlapping a next
  `execute_model` with a prior AsyncOutput wait — only safe if the engine still serializes
  execute on one thread (verify `max_concurrent_batches` on rig; `set_default_max_concurrency`,
  model_runner.py:217, exists); (c) line 203 `mask.cpu().tolist()` is in the fault-report path —
  keep it on the same thread.
- "Safe-by-design" here means *not a wedge*; it is still a per-step blocking wait on the D2H copy
  (silent latency) — acceptable, but the copy is the only remaining per-step L0 D2H family on the
  sample side.

**sampler.py:162 and sample/output.py:114-130 — which thread, which phase?**
- `sampler.py:162` (`cu_num_logits_np.tolist() if expanded_logits`): **host numpy** (built by
  `np.cumsum` in prepare_inputs, model_runner.py:1245-1247) → SAFE; runs on the **runner thread**
  inside `self.sampler(logits, input_batch)` (model_runner.py:1475), i.e., per-step in
  `sample_tokens`, never inside capture.
- `output.py:114-130` (`self.counts.cpu().numpy()` / `packed_mask.cpu().numpy()` / `.tolist()`):
  **runner thread, per-step, post-forward** — same `sample()`/`gather_sampler_output` call tree
  (:1475-1506, :1491), outside capture (sampling is never part of capture; `_dummy_sampler_run`
  at :858-870 is profile-run only). **Device-ness NEEDS-FILE** on `counts`/`packed_mask`: if they
  are device tensors (likely — sampler outputs), these are real per-step D2H syncs on the main
  thread → silent tail, not wedge (same-thread, pre-replay). They are the natural second wave of
  the async_utils stream-ordered copy pattern; if `SamplerOutput` host materialization can go
  through the `output_copy_stream` (or read the AsyncOutput mirror), it should — see Sec.7 #6.

**mamba_attn.py:292-295, 468, 644-649.**
- `292-295` (`num_computed_tokens_p_cpu[req_idx].item()`, `query_start_loc_p_cpu[...].item()`):
  **SAFE-by-name** (`*_p_cpu` = per-request host mirror — the same convention that produced
  `query_start_loc_cpu` for the gdn:546 fix and `seq_lens_cpu_upper_bound` via
  `torch.from_numpy`, model_runner.py:1347). Real host math. ONE confirm required because the
  worst-case mislabeling is severe: if these are device tensors, each `.item()` in the
  per-request loop is a D2H — thousands per step (and inside capture) → instant class A/C. The
  1-line print settle is in Sec.7 #1.
- `468` (`torch.any(prefill_to_decode).item()`): **SUSPECT — top rank** (Sec.5 #1). Device bool
  mask, no mirror in sight, and the same build chain that carried 546 into capture.
- `644-649` (`write_pos_cpu.to(torch.int32).tolist()`, `is_flush_cpu.tolist()`): **SAFE** —
  explicit `*_cpu` host tensors; `.to(int32)`/`.tolist()` are host ops even inside capture.

**short_conv_attn.py:552 discrepancy (raises, cheap to settle):** the tracked fix says the twin
was fixed from `(num_accepted_tokens - 1).cpu()`; the scanned tree **still shows the `.cpu()`**
at 552 while gdn's own 546 line is gone (only its historical comment remains at gdn:547). Either
the scan predates the twin patch or the twin patch landed incompletely. Exposure is 0 right now
(0 ShortConv refs in qwen3_5.py), so this is a **pre-serve gate**: before any ShortConv model is
served, grep and fix :552 — otherwise it is a guaranteed re-wedge at capture.

## 7. On-rig verification list (settle static vs dynamic; all read-only)

```bash
ENGINE=<engine-container>   # per your docker setup

# ---- (a) NORMAL DECODE BURST: 30 dumps, one file per (shot,pid), then grep per suspect ----
for i in $(seq 1 30); do
  for pid in $(docker exec $ENGINE pgrep -f 'vllm|worker'); do
    docker exec $ENGINE py-spy dump --native --pid $pid > /tmp/sp_${i}_${pid}.txt 2>/dev/null
  done
done
# per-suspect frame search (empty result = clean for that site):
grep -B2 -A4 'mamba_attn.py:468'                       /tmp/sp_*.txt          # #1
grep -B2 -A4 -E 'utils\.py:(734|735|737|811|834|837)'  /tmp/sp_*.txt          # #2,#3
grep -B2 -A4 -E '(common\.py:108|common\.py:208)'      /tmp/sp_*.txt          # #4
grep -B2 -A4 -E 'mamba2_attn\.py:(44|48|49|50)'        /tmp/sp_*.txt          # #5 (bind-cond)
grep -B2 -A4 -E 'mamba_attn\.py:(292|294|295|644|649)' /tmp/sp_*.txt          # should be mirror-only
grep -B2 -A4 -E 'async_utils\.py:(174|175|187|198|203)|output\.py:(114|119|130)|sampler\.py:162' /tmp/sp_*.txt  # expect only async_utils (safe path)
grep -B2 -A4 -E 'dp_utils\.py:(46|52|67|78|79)|triton_attn\.py:220|short_conv_attn\.py:552' /tmp/sp_*.txt
# frame histogram of the top of every dump (what else is hot during bursts):
grep -oE '[a-zA-Z_]+\.py:[0-9]+' /tmp/sp_*.txt | sort | uniq -c | sort -rn | head -25

# ---- (b) MTP1 BOOT CAPTURE: poll all workers 1s apart from capture start until done ----
# start MTP1 boot, then:
rm -f /tmp/capdumps.txt
while ! docker logs $ENGINE --since 2s 2>&1 | grep -q 'Graph capturing finished'; do
  for pid in $(docker exec $ENGINE pgrep -f 'vllm|worker'); do
    echo "=== pid $pid $(date +%T)" >> /tmp/capdumps.txt
    docker exec $ENGINE py-spy dump --native --pid $pid >> /tmp/capdumps.txt 2>/dev/null
  done; sleep 1
done
# which capture-time frames actually ran (the 546-shaped risk set):
grep -B2 -A4 -E 'build_for_cudagraph_capture|prepare_inputs_to_capture|prepare_attn' /tmp/capdumps.txt | head -120
grep -B2 -A4 -E 'mamba_attn\.py:468|utils\.py:(734|811|834)|common\.py:10[08]|mamba2_attn\.py:(44|48|49|50)' /tmp/capdumps.txt
# and confirm the fixed site never returns: NO gdn .cpu() frame, only the 547 comment:
grep -E 'gdn_attn\.py:5[0-9][0-9]' /tmp/capdumps.txt | sort | uniq -c

# ---- STATIC settles (run once, before/while debugging; no state change) ----
# 1) mamba_attn.py:292-295/468/644-649 -> which functions; device print (one run, main thread):
sed -n '280,300p;455,475p;635,655p' vllm/v1/attention/backends/mamba_attn.py
#     one-off: print(num_computed_tokens_p_cpu.device) in the build with MTP1 -> expect cpu
# 2) who calls the split helpers (does a live/capture build reach utils.py:734/811/834?):
grep -rn 'split_decodes_prefills_and_extends\|utils.py' vllm/v1/attention/backends/gdn_attn.py vllm/v1/attention/backends/mamba_attn.py vllm/v1/worker/gpu/model_states/mamba_hybrid.py | head -40
# 3) utils.py:1055 + ops/common.py:108/208 -> identify enclosing functions:
awk 'NR>=1000 && NR<=1060' vllm/v1/attention/backends/utils.py
awk 'NR>=95 && NR<=115; NR>=195 && NR<=215' vllm/v1/attention/ops/common.py
# 4) which mamba backend binds the recurrent group (mamba_attn vs mamba2_attn):
grep -rn 'MambaAttnBackend\|Mamba2AttnBackend\|attention_backend' vllm/v1/worker/gpu/model_states/mamba_hybrid.py vllm/v1/worker/gpu/attn_utils.py | head -30
# 5) short_conv twin fix state + gdn:192 comment target:
sed -n '146,200p' vllm/v1/attention/backends/gdn_attn.py; grep -n -E '\.cpu\(\)|\.item\(\)|\.tolist\(\)|\.numpy\(|to\(.cpu' vllm/v1/attention/backends/gdn_attn.py
sed -n '545,560p' vllm/v1/attention/backends/short_conv_attn.py
# 6) AsyncOutput host-materialization callers (threads) + concurrent batch count:
grep -rn 'sampled_token_ids\|copy_event\|get_outputs' vllm/v1/worker/gpu/*.py vllm/v1/engine/*.py | grep -v 'def ' | head -30
#     env: max_concurrent_batches (expect 1) and backend selection + dp_size from boot log:
#     grep -i 'attention backend\|data parallel\|dp_size' <boot.log> | head
# 7) triton backend indeed not selected (settles adjudication A):
#     grep -i 'Using attention backend\|attention_backend' <boot.log> | head
```

## 8. Residual gaps (do not carry as verified)

- **Sweep grammar.** The sweep only matched `.cpu()/.item()/.tolist()`. NOT covered: `.to("cpu")`,
  `.numpy()` on device, `event.synchronize()`, `stream.synchronize()`, `torch.*.synchronize()`,
  `wait_stream`, and `non_blocking` launchers. Noteworthy: connector.py:447 `d2h_done_event.synchronize()`
  is **host-side wait only** (no L0 submit) — harmless; but any `stream.synchronize()` from the
  PLE/EPLB threads would be a class-B hazard invisible to this scan. A second sweep with
  `synchronize|to\(['"]cpu['"]\)|\.numpy\(` is a 10-minute follow-up.
- **Host-mirror provenance.** The `*_cpu` mirrors (gdn, mamba, utils) are assumed host because the
  fix lineage (546) and model_runner.py:1347 confirm the pattern. The moment a mirror is populated
  by a per-step `.cpu()` elsewhere (uncaught by this sweep if spelled `.to("cpu")`), the SAFE rows
  in table A turn into class-C per-step syncs. Sec.7 #1/#5 covers this by reading the builders.
- **gdn_attn.py:192 comment** claims a `.tolist()` in `prepare_chunk_indices` — no live hit found
  in this sweep; could be a `.numpy()`/`.to("cpu")` or a stale comment. Sec.7 #5 settles it.
- **Threads not in the sweep:** EPLB `maybe_start_async_loop` (model_runner.py:507) is a live
  background thread on this stack; no `.cpu()/.item()/.tolist()` hits, but its device traffic is
  outside this audit — if wedges persist, add EPLB to the py-spy frame dict.

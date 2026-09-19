# GDN Capture-Path L0 Wedge — MTP1 Boot (boot 4010612, 2026-09-19 ~16:55Z)

## Summary
Second confirmed L0 command-list wedge, different call site. The MTP1 boot
(MTP_NUM_SPECULATIVE_TOKENS=1, stage-v24f) hung at kv_allocated/capture start
(+678s, first wedge since v3 landed). All four worker mains pinned 17+ minutes
(active, ~47% CPU each, sched_yield spin) inside
`ur_command_list_manager::appendUSMMemcpy` reached from the **main thread**:

```
sched_yield (libc.so.6)
  libze_intel_gpu.so.1.15.37833 frames
  ur_command_list_manager::appendUSMMemcpy (libur_adapter_level_zero_v2.so.0)
  build_for_cudagraph_capture (vllm/v1/attention/backends/gdn_attn.py:546)
  build_attn_metadata (vllm/v1/worker/gpu/attn_utils.py:314)
  prepare_attn (vllm/v1/worker/gpu/model_states/mamba_hybrid.py:327)
  prepare_inputs_to_capture (vllm/v1/worker/gpu/cudagraph_utils.py:683)
```

## The offending line
`gdn_attn.py:546`:
```python
num_accepted_tokens = torch.diff(m.query_start_loc)          # device tensor
num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()  # D2H USM memcpy
```
A device→host sync copy **mid-graph-capture**, through the same L0
command-list manager as the connector wedge. MTP0 never executes this path
(`build_for_cudagraph_capture` with MTP shapes differs) — consistent with
MTP0 boots being clean all day on the identical image.

## Relation to the connector fix (v3)
Same failure primitive — an L0 append spinning under contention during graph
capture/replay — at a different call site. v3 removed the connector thread's
L0 traffic entirely (0/170 mid-burst dumps contain appendUSMMemcpy; 15/15
clean bursts). It could not have fixed this site: it is the runner's own main
thread executing a `.cpu()` sync inside capture. It is the same *class* the
program has now reproduced in two independent places: any L0 USM memcpy
append racing graph-capture command-buffer appends can wedge the runtime.

## Evidence preserved
`~/fn-recipe-int4/.run/evidence/gdn-capture-wedge-4010612/` (native dumps:
worker_{319,345,402,433}.txt [NOTE: native dumps written after container stop
may be truncated — the 4 native dumps above were captured live in-session and
quoted verbatim in this doc; locals dump for worker 319 is complete],
worker_319_locals.txt). Observer wedge_obs_4010612.log.

## Fix direction (not yet implemented)
Replace the blocking `.cpu()` at gdn_attn.py:546 with a pre-computed host
value: during capture, `num_decode_draft_tokens_cpu` is a constant per MTP
config (num_speculative_tokens), and `num_accepted_tokens` is derivable on
host from the capture shape — no device sync needed at this point at all.
Candidate: compute both from `m.query_start_loc_cpu` when available, or pass
the capture constants directly. Same class of fix as v3: remove L0 from a
place it never needed to be.

## Campaign result unaffected
The 15/15 clean campaign and the 170-dump L0-absence proof were MTP0 on boot
3915112 — untouched by this finding. Boot provenance for the MTP acceptance
A/B is now: Boot A (MTP0) complete; Boot B (MTP1) blocked on this wedge.

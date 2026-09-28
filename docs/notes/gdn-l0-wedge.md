# GDN capture-path Level-Zero wedge — mechanism note

## Summary

A confirmed Level-Zero command-list wedge with a specific call site. A boot
with speculative decoding (MTP1, `MTP_NUM_SPECULATIVE_TOKENS=1`) hung at
kv_allocated/capture start (+678s). All four worker mains pinned 17+ minutes
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

A device→host sync copy **mid-graph-capture**, through the L0 command-list
manager. A boot without speculation (MTP0) never executes this path
(`build_for_cudagraph_capture` with MTP shapes differs) — consistent with
MTP0 boots being clean on the identical image.

## The failure class

Any L0 USM memcpy append racing graph-capture command-buffer appends can
wedge the runtime. This is the same *class* reproduced in two independent
places (a connector-thread variant and this runner-main-thread variant): an
L0 append spinning under contention during graph capture/replay. The
mitigation philosophy: remove L0 traffic from places it never needed to be.

## Fix direction (not yet implemented)

Replace the blocking `.cpu()` at `gdn_attn.py:546` with a pre-computed host
value: during capture, `num_decode_draft_tokens_cpu` is a constant per MTP
config (num_speculative_tokens), and `num_accepted_tokens` is derivable on
host from the capture shape — no device sync needed at this point at all.
Candidate: compute both from `m.query_start_loc_cpu` when available, or pass
the capture constants directly.

## Relation to the 2–6 h standing wedge

The README "Known limits" wedge is the broader runtime-level instability of
the Xe2 stack under sustained load (`ccs`/`bcs` engine reset signatures).
This note documents one precisely-localized instance of it on the
capture path. Both are recovered by a container restart; py-spy captures
before any restart are what made this mechanism visible.

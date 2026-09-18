# 2026-09-18 Graph-Mode Breakthrough (stage-v24c)

## Result
- Eager: 4.7-4.9 tok/s single-stream, 21.7 tok/s 8-way aggregate
- **stage-v24c (XPU graphs, FULL_DECODE_ONLY): 24.2-24.9 tok/s single-stream, 133 tok/s 8-way** (5-6x)
- Correctness intact: 17x23 -> 391; coherent long-form output

## Mechanism (measured, not theorized)
- Eager decode was CPU-launch-bound: all 4 worker main threads burned ~100% of one core each
  during decode (thrcensus: comm=VLLM::Worker_TP wchan=0); GPUs starved between ~460 kernel
  launches per step (~500us x 10 ops x 48 layers ~= the whole 205ms step).
- Batch-invariance proof: batch=1 and batch=8 both ran ~4.6 steps/s => fixed per-step cost,
  not bandwidth. This killed the sysmem-streaming theory (8x expert bytes would collapse steps).
- GDN Triton decode kernel timed standalone: 0.07ms/call x36 layers = 2.5ms/token = 1% — acquitted.
- PCIe acquitted earlier by direct measurement (H2D 26.9GB/s, DtoD 252GB/s VRAM-verified).

## Config that unlocked it
- `VLLM_XPU_ENABLE_XPU_GRAPH=1` (env) + `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
- `-O 0` (mode=NONE) intentionally KEPT: FULL_DECODE_ONLY does not require piecewise compilation
  -> no Dynamo/Inductor -> the 2026-08-28 a1-a7 compile-OOM quarantine does NOT apply.
- v24b guard checks (compilation.py:1385): FULL_DECODE_ONLY vs min CG support
  UNIFORM_SINGLE_TOKEN_DECODE — our GDN backend qualifies.
- startup line confirms: `cudagraph_mode: <CUDAGraphMode.FULL_DECODE_ONLY...>`,
  `mode: <CompilationMode.NONE: 0>`, "Capturing CUDA graphs (FULL): 100% 4/4".

## Patches (stage-v23 -> stage-v24c lineage)
- v24: ple_offload_layer.py:297 host-sync bypass under `torch.xpu.is_current_stream_capturing()`
  (first crash: "wait method cannot be used for an event associated with a command graph").
- v24b: removed v24 residue (3 dangling continuation lines -> IndentationError). LESSON: py_compile
  must run in the commit-verify gate BEFORE docker commit; v24 shipped a syntax error and launch51
  died in registry inspection. 
- v24c: capture path must return a DEVICE copy of `_gpu_output_buffer` (`.to(device,
  non_blocking=True)`) — the zero-copy view left downstream `F.linear` with mat2 on xpu:0 vs other
  on cpu ("Expected all tensors on same device"). Host-sync + sync-copy only in eager path.
- .env line 19 IMAGE pin drives start.sh; launch50 ran stage-v23 because IMAGE was never flipped.
  LESSON: verify `docker ps --format {{.Image}}` matches the intended image within 15s of launch.

## Env/config state (launch53 = current)
- IMAGE=stage-v24c; MAX_NUM_SEQS=8; MAX_NUM_BATCHED_TOKENS=2048; gpu-mem-util 0.75; -O 0
- KV cache: 127,078 tokens; "Total CPU offloaded parameters: 1.16" (GiB — small; PLE connector
  lane separate from UVA offload)
- CCL warning stands: "topology recognition shows PCIe connection" + ATL_TRANSPORT=ofi.
  Gloo 1-rank allreduce 0.03ms; CCL 4-rank microbench hung under torchrun (capture-incompatible?).
  Next lever: CCL transport/provider tuning for the remaining per-step collective latency.

## Next queue
1. FULL (prefill+decode) capture vs DECODE_ONLY: prefill is 1/48th of decode work in steady chat
   traffic, low priority.
2. CCL transport: try CCL_ATL_TRANSPORT=mpi / shm provider; measure step rate delta.
3. Re-bisect gpu-mem-util at 0.8-0.85 (graphs need capture pool; 0.75 was eager-era operating point).
4. max_num_seqs scaling study at graphs (8 -> 16/32): aggregate already 133 tok/s; batch-8 step
   rate still 4.6/s, so launch overhead per step is amortized — CPU may no longer be the wall.

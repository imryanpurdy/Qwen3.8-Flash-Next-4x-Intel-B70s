# 2026-09-18 Morning Addendum — Scaling Study + Config Sweep (launches 53-54)

## Numbers (stage-v24c)
| metric | value |
|---|---|
| single-stream | 24.2-24.9 tok/s (256-tok completions, 10.3-10.6s walls) |
| 8-way concurrent | 133.2 tok/s aggregate (1822 tok / 13.69s) |
| 16-way concurrent | 125-129 tok/s aggregate (3462/27.7s, 3542/27.4s) — saturated at 8-way |
| correctness | 17x23 -> 391; coherent prose |

## CCL allreduce microbench (4 ranks, 2MB bf16, xccl backend, subprocess launcher)
- default (ATL_TRANSPORT unset -> ofi warn): 0.173 ms/allreduce
- explicit CCL_ATL_TRANSPORT=ofi: 0.180 ms/allreduce
=> transport is NOT a lever (0.17ms x 2/layer x 48 = ~16.6ms/token accounted; graph replay
   already overlaps most of it). Gloo 1-rank 0.03ms. No config change warranted.

## GTT (live, stage-v24c at 0.75, serving idle)
dri/1: 76.14GB, dri/2: 76.14GB, dri/3: 82.64GB, dri/4: 76.14GB = ~311GB total GTT
(device 32GB x4 = 128GB; ~183GB beyond device = PLE mmap host-backed by design; DtoD 252GB/s
proves active working set is VRAM-resident. Sysmem-streaming theory stays dead.)

## gpu-mem-util 0.85 (launch54): KV 218,470 tokens (vs 127,078 @ 0.75)
- single-stream unchanged (24-25 tok/s); 16-way 129.4 tok/s (no aggregate gain)
- REVERTED to 0.75: KV headroom beyond working set buys nothing; 0.75 is the operating point.
- 0.85 anomaly: first 256-tok wall 29.6s then 7.7s/10.6s — warmup artifact, don't chase.

## Remaining wall analysis
16-way saturation at ~125-133 tok/s = ~7.8 steps/s batched (256 tok / 8 seqs / 2s... ~32 steps/s
per-batch-of-8 equivalent). Aggregate plateau 8->16 means scheduler/batch-size bound now, not
per-step CPU (graph replay removed that). Next levers, in order:
1. max_num_batched_tokens 2048 -> 4096+ (chunked prefill / larger decode batches)
2. max_num_seqs 8 -> 16 (matches measured 16-way demand)
3. Prefill FULL-graph capture (FULL vs FULL_DECODE_ONLY) for prompt-heavy traffic
4. Tensor-parallel allreduce volume: 2MB/layer is small; sequence-parallel not urgent.

## Ops notes
- CCL_ATL_TRANSPORT/CCL_PROCESS_LAUNCHER not set in container env; defaults are fine.
- memwatch not armed for launch54 (log only through launch53).
- All images committed with EP/CMD verified JSON; py_compile gate added after v24 IndentationError.

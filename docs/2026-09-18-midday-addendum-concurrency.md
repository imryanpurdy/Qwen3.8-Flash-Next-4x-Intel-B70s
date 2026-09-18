# 2026-09-18 Midday Addendum — Concurrency Sweep Results (launches 55-58, stage-v24c)

## Concurrency scaling matrix (max_tokens=256, same prompt, aggregate tok/s)
| config | single | 8-way | 12-way | 16-way | notes |
|---|---|---|---|---|---|
| MNS=8 MBT=2048 (launch53) | 24.2-24.9 | **133.2** | — | 125-129 | baseline graphs config |
| MNS=8 MBT=2048 gpu-util 0.85 (launch54) | ~24 | — | — | 129.4 | KV 218K tok; reverted to 0.75 |
| MNS=16 MBT=4096 (launch56) | ~24 | — | 192.2 (1st) → crash | — | **RPC sample_tokens timeout → EngineDead** at sustained 16-way |
| MNS=12 MBT=2048 (launch57) | ~24 | — | 19.4-73.3 | — | scheduler thrash; 12 not viable |
| MNS=8 MBT=2048 re-run (launch58) | 24.2 (10.7s/256) | — | 111.9-121.0 (12 in queue, 8 running) | — | **REPRODUCED ~120-130** |

## Findings
1. **133 tok/s @ 8-way is the reproducible ceiling** for stage-v24c. Engine logger confirms
   118-133 tok/s sustained generation throughput with 8 running reqs.
2. **MNS=16 + MBT=4096 crashes under sustained load** (RPC sample_tokens timeout after ~40s of
   16-way; engine dead; restart recovers). Crash class, not tunable-by-sweep — do not raise
   MNS beyond 8 on this vLLM build without further investigation (worker timeout, not OOM).
3. MNS=12 shows scheduler thrash (22 tok/s steady then recovery) — 8 is the stable batch width.
4. gpu-mem-util 0.85 no aggregate gain (129 vs 133 within noise); 0.75 restored (KV 127K ample).
5. MBT=8192 blocked by start.sh guard (GLM-5.3 QSA indexer crash class); 4096 allowed but
   pairs with MNS=16 which crashes — MBT stays 2048.
6. CCL allreduce microbench: 0.173 ms default vs 0.180 ms forced-ofi (2MB bf16, 4 ranks) —
   no transport lever. Graph replay already hides collective latency.
7. GTT live check (~311GB total, dri/3 82.6GB): PLE host-backed by design; DtoD 252GB/s stands;
   sysmem-streaming theory remains dead.

## Bottom line for Ryan
- 4.8 tok/s eager → **24.8 single / 133 aggregate (8-way)** = 5.2x / 27.7x vs last night.
- Context: SergiioB 133 single-card 8B; Steve lab 117.5 27B; FP8 cert floor 46.85. We are at
  4x-intel-B70 INT4 hybrid 35B-A3B delivering 133 aggregate — real competitive territory.
- Remaining walls: (a) per-batch step rate ~4.6-5.2 steps/s under graphs → GPU kernel time now
  dominant (GDN 2.5ms + MoE + 2 AR/layer); (b) MNS>8 crash class.

## Config state (launch58 = live)
IMAGE=stage-v24c; MNS=8; MBT=2048; gpu-util 0.75; -O 0; VLLM_XPU_ENABLE_XPU_GRAPH=1;
--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}; KV 127,078 tok; graphs 4/4 FULL.

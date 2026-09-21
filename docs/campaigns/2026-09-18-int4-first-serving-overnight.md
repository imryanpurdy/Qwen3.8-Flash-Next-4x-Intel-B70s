# 2026-09-18 Overnight Findings — First INT4 Serving + Bottleneck Isolation

## Milestone
- **03:51:00 — first INT4 serving engine on the rig** (stage-v22: PLE connector XPU event-gate fix
  `--enable-ep-weight-filter --quantization inc -O 0 --gpu-memory-utilization 0.75`, TP4/EP, MTP0 eager).
- Quant validated twice: 2+2→"4"; 17×23→"391" with clean decomposition.
- stage-v23: stripped per-token V22DIAG launch print (was on critical path; hypothesis failed — see below).

## Eliminations (measured, not reasoned)
| Suspect | Test | Result | Verdict |
|---|---|---|---|
| Per-token docker print | v23 strip, re-bench | 4.7→4.9 tok/s | **acquitted** |
| PCIe x1 Gen1 link | lspci LnkCap/LnkSta ×3 decodes + torch.xpu copy bench | **H2D 26.9 GB/s, D2H 8.3 GB/s, DtoD VRAM 252 GB/s** | **lspci misreports** (three independent decodes agreed and were wrong — same source, not independent confirmation) |
| GDN Triton decode fallback | standalone kernel bench, B=1 H=16 HV=48 K=V=128 | **0.07 ms/call × 36 layers = 2.5 ms/token ≈ 1% of budget** | **acquitted** |
| Host-memory expert streaming (bandwidth) | batch-invariance test: batch=1 vs batch=8 | **4.9 steps/s vs 4.6 batched steps/s — step rate invariant** | **acquitted** (8× tokens at constant step time = no per-token host round trip) |
| gloo CPU collectives | 1-rank all-reduce bench | 0.03 ms | **acquitted** |

## The live finding
- **~205 ms per decode step, batch-invariant.** Aggregate 21.7 tok/s at batch 8 (MBT=2048, MNS=8) vs 4.9 at batch 1.
- **Thread census during decode: every worker main thread (`VLLM::Worker_TP`, wchan=0) burns ~100% of one core;
  all aux threads asleep.** = CPU-side launch/dispatch saturation in eager mode. GPU starves between launches.
- oneCCL runs ATL/OFI (warned at init); microbench of xccl backend hangs out-of-context — not needed given census.
- **Config took**: MAX_NUM_SEQS=8 / MAX_NUM_BATCHED_TOKENS=2048 live in engine args; KV 127,078 tokens
  (slightly under 136K — MBT headroom; expected).

## The fix path (next launch)
- `VLLM_XPU_ENABLE_XPU_GRAPH=1` with `-O 0` (compile OFF — different path from the a1-a7 quarantine,
  which died in Dynamo/Inductor compile phase **before ever reaching capture**).
- Precedent on this exact box: **DSv4 TP4 PIECEWISE graphs = 80.8 tok/s** (results/README.md:30-46).
- Eager on this box has always been ~5 tok/s (FP8 27B MTP0: 5.52; INT4 35B: 4.9). The quarantine guarded
  compile-phase OOM, not capture-phase risk. H2 (GDN uncollectable) remains the open risk — watch capture logs.
- Fallback: revert to eager (stage-v23, serving-proven).

## Infra notes
- stage-v23 correct commit syntax (v22-proven): `docker commit --change 'ENTRYPOINT ["/opt/venv/bin/python3"]'
  --change 'CMD ["/opt/venv/bin/vllm","serve","Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"]' <container> stage-vNN`
  — always docker inspect the EP/CMD JSON after (first v23 commit produced EP=[] → dead container).
- launch45/46 sequence: stop.sh → arm memwatch → setsid nohup start.sh; acceptance = KV cache size line +
  Application startup complete + curl /v1/models.
- PCIe red herring documented for the record: LnkCap 0x0c11/LnkSta 0x0011 read as Gen1 x1 on all 4 cards
  and both switch down-ports — contradicted by direct DMA measurement. Config-space reads on this platform
  are untrustworthy; use transfer benchmarks.

# 2026-09-19 — PLE 82%-freeze extinction cause + watchdog rewrite design

## Extinction timeline (file mtimes inside image stage-v24c vs 09-17 freeze census)

| file | mtime | note |
|---|---|---|
| vllm/platforms/xpu.py | Sep 17 01:43 | BEFORE the freeze wave |
| vllm/v1/worker/gpu/model_runner.py | Sep 17 02:20 | BEFORE |
| vllm/model_executor/offloader/uva.py | Sep 17 12:19 | mid-wave (12 freezes 12:55→22:26) |
| vllm/v1/ple_offload/worker.py | Sep 17 22:37 | AFTER last freeze (22:26) |
| vllm/v1/ple_offload/ple_mmap_v18.py | Sep 17 23:02 | AFTER (image v18g built 23:02) |
| vllm/v1/ple_offload/connector.py | Sep 18 04:02 | AFTER |
| vllm/model_executor/layers/fused_moe/experts/xpu_moe.py | Sep 18 00:46 | AFTER |
| vllm/model_executor/layers/ple_offload_layer.py | Sep 18 08:02 | AFTER (v24c line) |

**Inference (marginal):** the freeze wave (12×82% shard-read stalls, 12:55–22:26 on 09-17) ended
in the worker.py + ple_mmap_v18.py window (22:37/23:02) — the v18 mmap rewrite generation. The
census freeze signature was the blocking open/read of ~10GiB shards at wildly varying cadence
(0.3→8s/it), i.e. I/O path behavior; v18 rewrote exactly that path. uva.py at 12:19 was mid-wave
so the UVA layer alone doesn't explain extinction. This is correlational (N=12 vs N=0 after,
confounded with image generations v16-v18 same evening) — NOT proof. The useful consequence:
**if the post-KV shm_broadcast mute shares a root with the v18-era PLE I/O design, it should have
survived the rewrite** (it did — wedgedAB/ABr/bisect1 are 09-18/19, all post-v18). So: two
separate classes; the mute is the live one; the 82% freeze is likely v18-extinct. Keep watching
for freeze recurrence as a v18-regression signal.

## Watchdog v2 rewrite — design (census-driven + Ryan's decision-logging order)

Defects to fix (all evidence-backed):
D1. `log_size` = byte-count via wc -c: tqdm carriage-return frames rewrite lines in place →
    file size can plateau while frames still advance → false "no growth" → kill of healthy boot.
D2. 3-probe × 60s stall limit (180s) sits INSIDE the healthy envelope: PLE shard cadence up to
    8.4s/it with 1-14s inter-frame gaps; KV->ready span 196-690s healthy.
D3. Device-count and signature paths bypass stall_count pacing (single-probe kill).
D4. No decision record: wedge logs show what happened, not why the watchdog decided.

New semantics:
1. Growth = server.log MTIME advanced < 45s ago (mtime, not size) OR /v1/models 200.
2. Phase-aware mute thresholds (parsed from boot_clock.jsonl + server.log markers):
   - pre-KV (PLE load/registration): no-mtime-kill at 300s (10x worst cadence)
   - post-KV (KV alloc done): no-mtime-kill at 900s (1.3x the 687s healthy ceiling), kill only
     if ALSO all 4 worker wchans blocked and captured
   - serving (health was 200 before): mute 180s + wchan capture, kill at 300s (serving stalls
     ARE fast-death: sample_tokens timeout observed at 300s)
3. NEVER kill on warning count. Warnings only feed the capture.
4. Decision log per kill AND per near-miss: {probe_ts, phase, mtime_age_s, size_delta, health,
   wchans[4], cpu%, trigger_fired, threshold_used, verdict} -> .run/wd-decisions.jsonl (the
   watchdog's own record = primary evidence for the next census, per Ryan).
5. Before every kill: force wchan/stack capture of all workers (observer-style) INTO the wedge log.
6. Preserve Xe2 signature fast-path (guc_exec_queue_timedout_job etc.) with stall_count pacing
   (D3 fix): require 2 consecutive probes with the signature.

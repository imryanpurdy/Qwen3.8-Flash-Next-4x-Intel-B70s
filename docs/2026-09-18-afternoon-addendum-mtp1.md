# 2026-09-18 Afternoon Addendum — MTP1 + Graphs Combined (launch59, stage-v24c)

## Result
| config | single-stream | 8-12 way aggregate |
|---|---|---|
| eager (launch45/46) | 4.8 | 21.7 @8 |
| graphs only (launch53/58) | 24.2-24.9 | 133 @8 / 120-123 @12 |
| **graphs + MTP1 (launch59)** | **35.7-38.1 (clean 256-tok reps)** | **120-123 @12 (16 reps) / 123.2 best** |

- Single-stream ladder vs Steve's FP8 lab: his MTP0 27.6 → MTP1 46.9; ours (INT4 hybrid, PLE
  host-backed lane) 24.8 → 37.2. Same ~1.5x MTP1 shape.
- Correctness: 17x23 → 391 intact; MTP acceptance 1.09-1.67 mean (healthy for draft=1).
- Speculator captured its own graphs ("Capturing model for speculator..."); startup clean.

## Why aggregate didn't rise with MTP1
MTP verify makes each step heavier (draft + verify + rollback); at 8-12 concurrent the batch
already fills the step, so accepted-length gain (1.2-1.4x) is offset by heavier steps. MTP pays
on single-stream latency-bound decode, not on saturated aggregate — matches expectation.

## Stability notes
- max_num_scheduled_tokens auto-clamped to 2048 by spec-dec settings (vllm.py:1942).
- Draft max_model_len clamped 262144 → 4352 (speculative.py:1578) — MTP lane is 4K-context
  bound in this build. Long-context target still needs the non-spec path (launch53 config).
- probe12.py must be docker-cp'd into FRESH containers (restart clears /tmp) — launch59 is a
  new container; keep /tmp/probe12.py upload in the launch checklist.

## Cumulative (same rig, 169GB INT4 checkpoint, PLE host-backed)
- 4.8 → 37.2 tok/s single (7.8x), 21.7 → 133 tok/s @8-way aggregate (6.1x) in ~12h.
- vs references: SergiioB 133 single-card 8B; Steve FP8 117.5/46.9; cert floor 46.85.
- Live config: stage-v24c, MNS=8, MBT=2048, gpu-util 0.75, -O 0, XPU graphs FULL_DECODE_ONLY,
  MTP_NUM_SPECULATIVE_TOKENS=1, KV 4352-clamped ctx.

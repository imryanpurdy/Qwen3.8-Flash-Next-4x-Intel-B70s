# Phase 3 workplan — pre-RAM / day-of / post-RAM

> Discipline: evidence before hypothesis before patch. Three prior
> "should-be-faster" paths on this rig measured slower (oneDNN, graphs,
> DEBUG instrumentation) — every hypothesis below lists the observation
> that would DISCONFIRM it before we write code for it.

## NOW (30GB RAM is sufficient — no box required for 1, box helpful for 2)

### 1. MTP4@4K stall investigation (Workstream A, phase 0)
The stall signature: tokens produced stop at 3,904/4,096 (96% of cap) with
MTP4, while MTP3 finishes 4,096 cleanly at the same context. Hypothesis
ladder, each with its disconfirming observation FIRST:

| # | Hypothesis | Disconfirmed if | Test |
|---|---|---|---|
| H1 | KV-block reservation ignores +k draft slots near cap → pool starves at 96% | scheduler logs show free blocks at stall; stall reproduces at 50% of cap with proportional truncation | stall-repro with `--max-model-len 8192` (stall should move to ~7,864/8,192 if cap-proportional) + capture scheduler block stats at stall |
| H2 | Persistent-scratch sizing (GDN spec scratch) is ctx-proportional and overflows | scratch alloc static in logs | compare scratch alloc lines MTP3@4K vs MTP4@4K |
| H3 | Verify kernel row-cap state corruption at 4 rows (2-row exact replay generalized, 4 not) | MTP4@512 loops fine for >4,096 produced tokens at tiny ctx | long-run MTP4@512: 20.727 cell, force max_tokens > 4096 — if it stalls at produced-token counts unrelated to ctx, H3 strengthens |
| H4 | QSA layer interacts with 4-row draft only at long ctx | MTP0@4K also stalls | control run MTP0@4K only if A stalls and B reproduces |

Evidence pack: `mtp4-stall-evidence.sh` (staged in flashnext-scout, moves to
jobe post-rebuild): synthetic 4K run with ignore_eos, dmesg/xpu-smi snapshot
at stall, memory state, then the H1–H4 control points. Rule from the rig:
idle-clock samples prove nothing — sample during the stall.

### 2. Read-only recon (no box needed)
- Pull `vllm-project/vllm-xpu-kernels` grouped-GEMM sources (the compiled
  `.so` we vendored came from this repo — study launcher/tile conventions
  for Workstream B while we wait).
- Read the lab's a1–a7 compile-OOM logs (already in recipe repo) and write
  the graphs-retrial runbook so day-of execution is mechanical.

## DAY OF REBUILD (before first vLLM boot)

1. Verify artifacts survived: `files/a367-kernel/` shas
   (`_xpu_C.abi3.so` = `593a7107…abc43b`) + `check-weights.sh` identity
   gate on re-download (`bcd9f01d…`).
2. Data volume: ext4 (not zfs), 96G swapfile, `HF_HOME=/data/hf`,
   `PREFLIGHT_DISK_GB` trimmed in `.env` (232GB NVMe cannot pass 200GB
   floor with 185GB weights — deliberate, documented).
3. First light: eager MTP1 + 4 selectors (lane5 doc) → 46.85 certified
   floor, quality gates `afffd211`/`1d833e5f`.
4. Then graphs retrial (PIECEWISE → FULL_DECODE_ONLY, watchdog on, bit-exact
   gate before adoption).

## POST-RAM SEQUENCE (once first light is green)

| Step | Work | Gate |
|---|---|---|
| 1 | Run `mtp4-stall-evidence.sh` (H1–H4) | stall reproduced + hypothesis selected by evidence, not plausibility |
| 2 | Patch per selected hypothesis; 4-row exact replay extension only if H3 wins | regenerated lossless battery (new hashes, NOT `afffd211` reuse), acceptance ≥ 799/852 ref |
| 3 | MTP4@4K e2e | ≥ 63 tok/s floor (Lane-2's own 512-ctx +34%) |
| 4 | Workstream B microbench (block-FP8 grouped GEMM shapes M=1/2/4) | ≥1.5× on routed-GEMM microbench before ANY integration |
| 5 | Integration behind a flag, eager fallback preserved | 7/7 quality battery, acceptance parity ±2%, e2e A/B on same client |

Standing rule (Ryan, 2026-09-14): we write what's missing. Kernel-starved
platform = highest-return custom work; every kernel lands on unclaimed
silicon. But the oneDNN lesson stands: measure before adopting, three
"obvious wins" measured slower.

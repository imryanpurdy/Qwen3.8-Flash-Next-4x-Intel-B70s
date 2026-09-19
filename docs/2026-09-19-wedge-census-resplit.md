# 2026-09-19 — Wedge census re-split (Ryan-ordered): watchdog kills healthy boots

## Verdict (41 episodes re-classified with timestamp-delta evidence)
- **~37% true wedges (15)**, ~54% killed-while-progressing (22), ~10% indeterminate (4).
- The ~92% native wedge rate COLLAPSES. The watchdog is the largest single boot-reliability
  problem — and it is our code, fixable without kernel work.

## True native wedge classes (only these survive)
1. **PLE shard-read freeze at 82% (14/17)** — 12 episodes, ALL on 09-17, none since. Cadence
   1-3s/it then 24-227s of total silence. Possibly extinct on the current stack (post-migration,
   v2x images). Watch for recurrence; do not assume.
2. **Post-KV shm_broadcast mute** — wedgedAB (18.8min), wedgedABr (17.4min), bisect1 (6.7min):
   KV alloc line, then only 60s-warning corpus to EOF, zero other lines. Still live (9/18-19).
   Healthy KV->ready span is 196-690s (preMTP0 214s, 2967807 621s, clean ~687s) — bisect1 was
   killed at 2.05x the reference recovery span; can't prove dead vs 10-min-slow from logs alone.

## Watchdog defects (both demonstrated)
1. **No-growth probe misfires during PLE tqdm**: frames advance 6→41-100% at ~8s/it, kill lands
   1-14s after the last frame (three boots killed AT 100% PLE-complete). Probe semantics read
   stale file size (tqdm carriage-return frames) or check at wrong cadence.
2. **3-broadcast-warning trigger fires on healthy compiling boots**: 2967807 triggered at +726s
   with workers at 93% CPU in Triton g++ compile; recovered to ready +877s. Count alone is not
   dispositive (handback §3 revision, now evidence-backed).

## Watchdog fix design (next boots, after MNS=12 boot completes)
- Treat "log line appended < 30s ago" as growth (mtime-based), not size-based.
- During PLE-load phase: no-growth threshold ≥ 120s (cadence 8s/it, frames 1-14s apart).
- Never kill on warning count; kill only on (mute ≥ 15min) AND (all workers' wchan blocked AND
  no CPU) — capture wchan/stack first (observer already does this).
- retry2-killed shows kills also land mid-weights-load with ZERO warnings — same fix covers it.

## Bonus datum for KV-expansion retry
wedgedAB KV-flag boot: `GPU KV cache size: 282,880 tokens` at 9.86 GiB — the flag allocates 2.2x
native (127,078 @ ~3.1 GiB effective) before that boot's post-KV mute. Flag mechanically works;
its wedge history is fully explained by the native post-KV mute class + watchdog behavior.

## Full evidence
Class counts, per-episode last-frame/cadence tables, exact lines: subagent transcript
~/.hermes/cache/delegation/live/deleg_02548e0c/task-0.log; local corpus
C:\Users\imrya\flashnext-scout\wedge-census\ (46 files).

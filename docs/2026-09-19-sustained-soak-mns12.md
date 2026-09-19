# 2026-09-19 — Sustained soak results: MNS=12 @ 8-way and 12-way (boot 3031727)

## Harness honesty
The soak script's timer arithmetic broke under heredoc quoting (ms field captured the boot
epoch). Aggregates below are reconstructed from output-file mtimes — first-file-per-round
boundaries, all files within a round land within 0.5s (simultaneous finish). The 8-way rounds
have clean boundaries (29.4s each); the 12-way round walls are read as boundary-to-boundary
deltas (±1-2s fuzz from spawn latency). A timer-fixed rerun is cheap and owed before any
percentage here becomes a cited fact (Ryan's correction on the 163±10 span touching 175).

## Results
### 8-way on MNS=12 (gate: agg>=200 + post_ok + zero timeouts)
- 4/4 rounds x 600 tok x 8 streams, zero errors, ~29.4s/round
- Sustained aggregate ~163 tok/s (mtime-recon, ±10) vs 175.2 burst reference
- post-soak: models=200, completions=200, 0 timeouts, 0 EngineDead
- 19,200 tokens served, engine healthy after. First sustained-concurrency evidence on this rig.

### 12-way on MNS=12 (gate amended by Ryan: STABILITY ONLY — post_ok + zero timeouts;
### aggregate = measurement, not gate)
- 4/4 rounds x 600 tok x 12 streams, zero errors, zero timeouts
- Round walls: 138.5 / 136.9 / 158.3 / ~157s → **aggregate 52.0 / 52.6 / 45.5 / ~46 tok/s**
- post-soak: models=200, 0 timeouts, 0 EngineDead — **STABILITY GATE PASSED at full width**
- 28,800 tokens served; KV usage 5.5% (129,689 pool)

## Findings
1. **The curve BENDS HARD at C=12 — 52 aggregate, not the 230-260 linear zone.** 12-way is
   3.1x slower per-stream than 8-way (4.3 vs 20.4 tok/s). The 8->12 aggregate scaling is
   NEGATIVE (-2.8x). Something serializes badly when MNS width is fully occupied: candidates
   = per-step PLE round-trip under wider batches, graph replay batch shapes, QSA paged-kernel
   path at >8 seqs, or scheduler overhead. This is the real MNS=12 answer the sweep wanted.
2. **Stability holds at both widths.** 48K tokens total across both soaks, zero stalls,
   zero deaths, engine healthy after each. The 02:21 stall did NOT reproduce at 8-way
   sustained or 12-way full-width on this boot. Load-dependence hypothesis: NOT confirmed
   by width alone — the MNS=8-config crash at 7 running reqs looks config-or-boot-specific
   (that run was on boot 2967807's config at MNS=8, this is MNS=12 on boot 3031727).
3. **Burst-vs-sustained sag (175→163 @8) is a HYPOTHESIS, not a fact** — 163±10 touches 175.
   Timer-fixed rerun owed before citing.
4. Round-3 12-way dip (45.5 vs 52.6) = single-round variance at heavy width; both rounds
   stable and error-free. Not conclusive of drift.

## Next (priority order)
1. Timer-fixed rerun of both soaks (cheap, kills the ±10 fuzz, settles the 7% sag question).
2. Diagnose the C=12 collapse: one variable at a time — compare per-step latency traces at
   8 vs 12 streams (engine counters exist: Avg generation throughput per request in logs).
3. MNS=16 NOT YET: at 52 agg the 12-way result makes 16's expected value unclear until the
   serialization mechanism is understood. Sweep pauses at 12 until the bend is explained.
4. Watchdog v2 implementation (design committed 6599e6e) — independent of the perf lane.
5. KV-expansion retry AFTER watchdog v2 is live (census shows its 3 failures were probably
   2 watchdog kills + 1 genuine mute; flag mechanically allocates 282,880 tokens).

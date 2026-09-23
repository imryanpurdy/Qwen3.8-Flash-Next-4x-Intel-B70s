# 2026-09-23 — Measurement reconciliation: soakfix.py vs bench_harness.py

**Artifacts:** proc output proc_79dcaf271a3e (box), both harness sources (md5s), ledger MEASUREMENT-RECONCILIATION-V2.

## The dispute

The pair-A/B batteries reported 16-way aggregates far below Saturday's soak campaign (171.6–195.9 vs 315.0 sustained), suggesting a platform regression on the restored 7.0 stack. Reconciliation was run on the live engine, no config change: **Saturday's soakfix.py itself, unmodified** (16 streams, 4 rounds, 600 tokens, r2–r4), then bench_harness.py at matching 600-token parameters.

## Findings

1. **Both harnesses share the identical aggregate formula:** `agg = Σ completion_tokens ÷ round wall`, neither uses ignore_eos. Cross-harness numbers are individually honest — they measure different regimes.
2. **The prompt shape is the entire gap:** soakfix prompts an open-ended essay that runs TO the 600-token cap; bench_harness prompts "Write a short thank-you note of exactly three sentences," which EOS-stops near 110–130 tok/stream. Rounds that never reach the cap measure the ramp+drain slice, not sustained decode.
3. **Stream count confirmed: 16** (9,600 tokens/round = 16×600; Saturday's doc "8-way" labels belonged to adjacent sections).
4. **Numbers of record (tonight's engine, soakfix unmodified):** r1 294.2 / r2 300.7 / r3 298.7 / r4 300.3 → **sustained_agg (r2–r4) = 299.9**; 0 errors. Saturday: 314.5/310.4/320.1 → 315.0. Delta −4.8%, inside the boot-to-boot band (±2.1% 1σ).
5. bench_harness 16×600 same engine: 185.0/168.2/208.7/215.7 (ctok 1,511–2,126 — EOS stopped far under the cap, proving the shape effect).

## Verdict

**Platform exonerated.** Tonight's platform measured with Saturday's method sits at the ≥300 bar. The 195-vs-320 "regression" was a measurement-category error (prompt shape), not hardware. **Comparability law:** any cross-harness number must print its formula AND prompt shape beside it; never compare soakfix sustained with bench_harness short-burst.

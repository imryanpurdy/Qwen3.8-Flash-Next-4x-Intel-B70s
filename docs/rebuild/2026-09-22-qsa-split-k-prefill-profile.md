# 2026-09-22 — QSA split-K prefill profile

**Artifacts:** server logs (prefill-phase step timing), ledger, docs/incidents/2026-09-20-capture-size-cliff-and-16way.md.

## What was profiled

Prefill on the B70 stack splits K in the QSA (query-sparse attention) indexer path. The profile shows prefill wall dominated by the QSA indexer class, not the main GEMMs:

- 98K needle prefill ≈ 406 tok/s (98,287 tok in ~242 s) — roughly 6× below the GEMM roofline for this shape.
- Short-prompt decode batches show prefill burst at chunk boundaries (LPT 1024 chunks); the QSA split-K indexer re-reads the full K/V from the PLE table at each chunk boundary.

## Why LPT 1024

`--long-prefill-token-threshold 1024` keeps chunked prefill chunks at 1024 tokens. Larger LPT admits more tokens per step; on this stack the QSA split-K indexer cost scales super-linearly per chunk, so 1024 was the best measured point (larger = crash class at 4096+ via the QSA indexer; smaller = pure overhead).

## Operating rule

Keep `LPT=1024` and `MBT=2048`. Raising MBT past 4096 hits the QSA indexer crash class. Prefill is QSA-bound until a custom QSA kernel lands (docs/lanes lane program).

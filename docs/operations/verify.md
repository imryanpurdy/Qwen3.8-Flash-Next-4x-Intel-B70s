# Verification contract — served line

Two tiers: **smoke** (after every restart/rollback, minutes) and **full gate**
(before any promotion or after engine/flag changes, ~1 h). All harnesses live
in the repo: `tests/verify.sh` orchestrates the full gate;
`scripts/toolcall.py`, `scripts/needle_probe.py`, `scripts/soakfix.py`,
`scripts/single-stream.py` are the harnesses. Prefix/recall scans live on the
host (`prefix_scan.py`, `recall100k.py`) and are run against the lane port.

**Comparability law** (from `tests/verify.sh`): every number names its
harness, aggregate formula, and prompt shape; rows are never comparable across
harnesses. Medians with spread, never best. Tool calls are compared
**structurally** (function name + argument JSON as parsed objects), never by
raw text.

## Smoke tier (post-restart / post-rollback)

1. **Models endpoint** — `GET /v1/models` answers 200 and lists the served
   name (`qwen-256k`).
2. **1-token smoke** — one chat completion, `max_tokens: 1`, HTTP 200 with a
   non-empty `choices[0]`. Proves the engine actually generates (a wedged
   engine can keep `/v1/models` answering — same probe the watchdog uses).
3. **Strict tool-call check** — `scripts/toolcall.py` at small count: an
   exact-args JSON tool call must come back with the right function name and
   the arguments parsing to the expected object (structural compare). This is
   the fast parser/quality canary — a sampler or parser regression shows up
   here first.
4. `<lane>/start.sh status` green (container + API + watchdog).

## Full gate — `tests/verify.sh`

`./tests/verify.sh` from the deploy directory. Requires the engine running and
`.env` present. Fails loudly; exit code = number of failed gates; transcript
in `.run/verify.out`. Gates (all must PASS):

| # | Gate | Harness | Pass condition |
|---|---|---|---|
| 1 | Tool calls | `scripts/toolcall.py --count 20` | 20/20 structural EQUIV + multi-tool `CORRECT_PICK` + nested-args `PASS` |
| 2 | 97K needle | `scripts/needle_probe.py` (engine-calibrated via `usage.prompt_tokens`) | `CORRECT=YES` + `SIZE_OK=YES` (≥97,000 **engine-confirmed** tokens), salted, temp 0 |
| 3 | Sustained (GATE metric) | `scripts/soakfix.py --n 32 --max-tokens 600 --rounds 15` | MEDIAN r2..r15 within 10% of 1,038.3 tok/s (band 934.5–1142.1) **and** soak clean (0 errors, post-check OK); r1 discarded as warmup |
| 4 | Single-stream | `scripts/single-stream.py` (N=20, first discarded) | median of 19 ≥ 40 tok/s (validated band 52.3–52.8) |
| 5 | Boot receipt | container logs | ≥1 `Application startup complete` line |

## Optional long-context scans (promotion evidence)

Run off the serving port (quiet engine) when an engine/flag change could touch
long-context state:

- **Shared-prefix scan — `prefix_scan.py`, run r3.** One ~99K-token shared
  prefix (seeded records with unguessable ref codes) × 8 concurrent requests
  with distinct lookup/quote/anomaly tasks, cold-salt and warm-salt phases,
  plus a T=0 logprob drift test (cold-vs-warm against a cold-vs-cold floor).
  Gate: (a) zero degenerate outputs (run-length ≥8 or periodic loop ≥48
  tokens), (b) warm incorrect/misquote/false-anomaly ≤ cold + 2, (c) median
  cold-vs-warm drift ≤ max(1.5× cold-vs-cold floor, 0.06) with every warm
  request confirmed as a cache hit on `/metrics`, ≥3 repetitions, (d) no
  warm-only scan anomalies. **What PASS looks like at promotion: drift
  0.0/0.0** (cold-vs-warm 0.0 against a 0.0 cold-vs-cold floor — a cache hit
  restores the cold path's state exactly; the pre-fix failure signature was
  0.25–0.36 drift with warm-only anomalies).
- **≤100K recall — `recall100k.py`.** Two full documents (~60K and ~95K
  tokens), 50 paired exact-value lookups each (100 total) at random positions,
  concurrency 8, thinking off, temp 0.8/top_p 0.95; wrong values and
  non-answers reported separately. **What PASS looks like at promotion:
  99/100** correct exact values.

## Promotion gate summary (what "verified" meant on 2026-10-04)

recall **99/100** · batch **8/8** · soak **3112/3112, 0 errors** · prefix scan
r3 **PASS (drift 0.0/0.0)** · full `tests/verify.sh` gates green. See the
STATUS promotion record.

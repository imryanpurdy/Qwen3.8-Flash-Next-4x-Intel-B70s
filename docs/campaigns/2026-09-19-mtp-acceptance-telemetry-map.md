# MTP Acceptance Telemetry Map — vLLM 0.26.1 lineage (2026-09-19)

Purpose: locate every spec-decode acceptance metric vLLM can emit, its exact
log format, and the cheapest way to get an accepted-drafts-per-step series from
the rig's server.log for the prereg check
(docs/campaigns/2026-09-19-mtp-acceptance-check-prereg.md).

## Bottom line

YES — the 0.26.1-lineage engine logs an acceptance line on the stats interval:

```
SpecDecoding metrics: Mean acceptance length: %.2f, Accepted throughput: %.2f tokens/s,
Drafted throughput: %.2f tokens/s, Accepted: %d tokens, Drafted: %d tokens,
Per-position acceptance rate: %s, Avg Draft acceptance rate: %.1f%%
```

Emitted by `vllm.v1.spec_decode.metrics.SpecDecodingLogging.log()` through the
engine stat logger, every `VLLM_LOG_STATS_INTERVAL` seconds (default 10.0),
whenever spec decode is enabled and `--disable-log-stats` is not passed
(default off). Identical format at tag `v0.26.1rc0` (closest released tag; see
Provenance) and at current `main`. Two additional channels exist:
Prometheus counters on `/metrics`, and per-request metrics in the API response
(`--per-request-spec-decode-metrics`, default `none`).

The counts are made in the **scheduler** from per-request sampled output token
ids (sched.py:2033-2053), i.e. after verification and after the runner's
optimistic-accept correction — so the metric is runner-agnostic and is valid
for the MRV2 (VLLM_USE_V2_MODEL_RUNNER=1) era and post-v3 alike.

## Rig build provenance (read before trusting any line number)

- Rig version string: `0.26.1rc1.dev1250+g76cfe1cd8` (stage-v24f campaign,
  docs/campaigns/2026-09-19-connector-v3-fix-and-smoke.md).
- vllm-project/vllm has no tag `v0.26.1` (tags: `v0.26.1rc0`, `v0.26.0`,
  `v0.26.0rc1`); commit `g76cfe1cd8` returns 404 via GitHub API for
  vllm-project/vllm and intel/vllm => the wheel is a fork/dev build.
- The file layout matches current upstream main exactly
  (`vllm/v1/worker/gpu/spec_decode/{mtp,multi_module_mtp,adaptive_verification,...}`
  all exist upstream), so upstream main is the right reference source; where a
  released tag matters I cite `v0.26.1rc0`.
- Caution: the fork's `model_runner.py`/`connector.py` (audited snapshot
  flashnext-scout/audit-src, flashnext-scout/v3-src) are patched locally; the
  fork's scheduler/metrics files were NOT in the snapshot. On-rig confirmation
  that the line exists (grep -c) comes first (§5 Q1).

## 1. Channel A — stats-interval log line (server.log)

### Format (exact)
`vllm/v1/spec_decode/metrics.py:120-136` (main) — `log_fn(...)` with
`"SpecDecoding metrics: Mean acceptance length: %.2f, Accepted throughput:
%.2f tokens/s, Drafted throughput: %.2f tokens/s, Accepted: %d tokens,
Drafted: %d tokens, Per-position acceptance rate: %s, Avg Draft acceptance
rate: %.1f%%"`. Byte-identical at v0.26.1rc0 (rc0 metrics.py:121-137).

Example line (illustrative, per-position length = num_speculative_tokens = 3):
```
INFO 09-19 14:22:31 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 2.40, Accepted throughput: 143.2 tokens/s, Drafted throughput: 197.6 tokens/s, Accepted: 1211 tokens, Drafted: 1231 tokens, Per-position acceptance rate: 0.813, 0.622, 0.307, Avg Draft acceptance rate: 76.8%
```
Note: the record's `[file:line]` is the `logger.info` call inside
`metrics.py` (the bound `log_fn`), so expect `[metrics.py:...]`, not
`[loggers.py:...]` — grep on `SpecDecoding metrics` regardless of module.

### Semantics
- Aggregated over one logging interval (reset after each log, metrics.py:137),
  i.e. a rolling ~10 s aggregate, not per-step and not per-request.
- `Mean acceptance length` includes the bonus token: `1 + accepted/drafts`
  (metrics.py:113-114); includes bonus => compare across boots with fixed k=3.
- `Per-position acceptance rate` vector has `num_speculative_tokens` entries
  (metrics.py:116-118); with MTP (k=3) expect 3 values.
- `Drafted` = proposed draft tokens (before grammar invalidation, which we do
  not use), `Accepted` = accepted draft tokens (excludes bonus token).
- Single `SpecDecodingStats` accumulator per engine per interval
  (SpecDecodingStats.new, metrics.py:24-49); the line is logged once per
  interval per engine.

### Emission point / cadence
- `LoggingStatLogger` holds `self.spec_decoding_logging` (main
  loggers.py:122); `record()` feeds it from
  `scheduler_stats.spec_decoding_stats` (loggers.py:229-230; only when not
  None); `log()` calls `spec_decoding_logging.log(log_fn)` after the
  throughput line (loggers.py:325). Level: `logger.info`, downgraded to
  `logger.debug` when the engine is idle (loggers.py:272).
- Trigger: sync `LLMEngine.step()` → `do_log_stats_with_interval()` every
  `envs.VLLM_LOG_STATS_INTERVAL` (llm_engine.py:338, 410-418;
  envs.py:48 default **10.0 s**, envs.py:857-859). `vllm serve` uses the
  AsyncLLM path: same `LoggingStatLogger`, `record()` per EngineCore output
  (async_llm.py:851-856), `log()` via `AsyncLLMEngine.do_log_stats()`
  (async_llm.py:1047-1049). The periodic trigger for the async path is
  exposed but its exact caller was not traced in the snapshot — treat
  "~every 10 s during traffic" as expected-but-unverified; presence in
  server.log is decided by the grep on-rig.
- Gate 1 (spec decode): `SpecDecodingLogging` only observes when
  `scheduler_stats.spec_decoding_stats is not None`, which is only populated
  when the scheduler schedules draft tokens (sched.py:2033-2045). With spec
  decode fully off (MTP_NUM_SPECULATIVE_TOKENS=0), **no line is emitted**
  (see §4 caveat).
- Gate 2 (stats logging): `--disable-log-stats` default `False`
  (arg_utils.py:581 main / rc0:535); AsyncLLM: `log_stats = not
  disable_log_stats` (async_llm.py:275). Default logger is also skipped only
  if the root logger is below INFO (loggers.py:1271-1275).

## 2. Channel B — Prometheus counters (/metrics)

`vllm/v1/spec_decode/metrics.py:177-264` (`SpecDecodingProm`), wired in
`PrometheusStatLogger` (loggers.py:458, 470-478). Counter names (exposed
with the usual `_total` suffix):
- `vllm:spec_decode_num_drafts` (default docs + code line 229)
- `vllm:spec_decode_num_draft_tokens` (line 230)
- `vllm:spec_decode_num_accepted_tokens` (line 231)
- `vllm:spec_decode_num_accepted_tokens_per_pos{position="..."}` (lines
  254-264; one series per draft position, 0..k-1)

Observed per scheduler stats (metrics.py:266-281). Not gated by
`--disable-log-stats` (PrometheusStatLogger is always appended, loggers.py
1278). Acceptance rate via PromQL (metrics.py:180-195):
`rate(vllm:spec_decode_num_accepted_tokens_total[$i]) / rate(vllm:spec_decode_num_draft_tokens_total[$i])`,
mean acceptance length `1 + rate(accepted)/rate(drafts)`.

## 3. Channel C — per-request metrics in the API response

- Flag: `--per-request-spec-decode-metrics {none,summary,detailed}`, default
  `none` (arg_utils.py:694-695, 1550-1551; config default observability.py:48;
  docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/).
- Response shape: `usage.metrics.speculative_decoding` =
  `{mean_acceptance_length, draft_acceptance_rate, acceptance_histogram,
  num_spec_steps, num_accepted_draft_tokens, num_draft_tokens,
  num_spec_tokens}`; `detailed` adds `per_step_accepted`,
  `per_step_drafted` (per verify step, ordered) — stats.py:304-344;
  docs page (field table). `acceptance_histogram[j]` = # steps accepting
  exactly j draft tokens (length k+1).
- Where computed: scheduler per verify step (sched.py:2051-2063 via
  `RequestSpecDecodeMetrics`, accumulated on the request, stats.py:304-344),
  attached to EngineCoreOutput (sched.py:2224) and surfaced by the API
  layer. Requires client-side capture — NOT in server.log. No collection
  cost when `none` (docs page).

## 4. Will stage-v24f rig logs contain the line? (pre-check answer)

Likely yes, with caveats:
1. `--disable-log-stats` default False and nothing in the workspace launch
   scripts disables it → the per-interval block ("Engine 000: Avg prompt
   throughput: ...") should already be in server.log; if those lines are
   present, the `SpecDecoding metrics` line will be too (same `log()` call,
   loggers.py:325).
2. Required: spec decode ACTIVE. If the recipe sets
   MTP_NUM_SPECULATIVE_TOKENS=0 by removing/wiping speculative_config, the
   stats accumulator is never created and the line never appears — the
   absence is itself the "spec decode off" signal for boot A. If the A/B
   harness instead keeps config with num_spec_tokens=0, drafts are never
   scheduled and the line will also be absent/empty — verify §5 Q4.
3. Idle intervals (no traffic between bursts) downgrade the line to DEBUG
   (loggers.py:272). The 8-way soak keeps the engine non-idle, but any
   "post-campaign, idle pillar" log region needs `-i` grep or DEBUG capture.
4. Per-position vector length = configured k (3 on the rig). A torn/stale
   n-gram era should show depressed `Accepted`/`Avg Draft acceptance rate`,
   which is exactly the prereg's log-forensics signal (monotone step change
   across the V2 cutover and the v3 cutover).
5. Advisory: the fork may renumber/patch `sched.py`/`metrics.py`; the
   predicted line's format is a hard assertion to check FIRST (`grep -c
   "SpecDecoding metrics"`).

## 5. Grep/parse recipe (part b)

One-liner (presence + raw lines):
```bash
grep -nE "SpecDecoding metrics" server.log
```

Python: extract per-interval acceptance time series to CSV
(parse the vLLM log prefix `INFO MM-DD HH:MM:SS`; year taken from the
architecture of your log rotation / today — pass `--year` if needed):

```python
import re, csv, sys
from datetime import datetime

LINE = re.compile(
    r'INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[[^\]]+\] '
    r'SpecDecoding metrics: Mean acceptance length: ([\d.]+), '
    r'Accepted throughput: ([\d.]+) tokens/s, Drafted throughput: ([\d.]+) tokens/s, '
    r'Accepted: (\d+) tokens, Drafted: (\d+) tokens, '
    r'Per-position acceptance rate: (.+), Avg Draft acceptance rate: ([\d.]+)%')

rows = []
for path in sys.argv[1:] or ["server.log"]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%m-%d %H:%M:%S").replace(year=2026)
            rows.append({
                "ts": ts.isoformat(),
                "mean_accept_len": float(m.group(2)),
                "accepted_tps": float(m.group(3)),
                "drafted_tps": float(m.group(4)),
                "accepted": int(m.group(5)),
                "drafted": int(m.group(6)),
                "per_pos": m.group(7).replace(", ", ","),
                "draft_accept_rate_pct": float(m.group(8)),
            })

with open("spec_series.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else
                       ["ts","mean_accept_len","accepted_tps","drafted_tps",
                        "accepted","drafted","per_pos","draft_accept_rate_pct"])
    if rows: w.writeheader(); [w.writerow(r) for r in rows]
print(f"{len(rows)} spec-decode lines -> spec_series.csv")
```

Notes for the forensics leg of the prereg:
- Segment by boot: grab `Initializing a V1 LLM engine` / `Engine 000:` line
  offsets to slice the series at the V2-cutover and v3-cutover boundaries.
- The interval counter is cumulative per interval, so accepted-vs-drafted
  across boots is directly comparable (same k, same prompt set); only
  per-boot totals of `Accepted`, not throughput, are the acceptance signal.
- If you need per-step resolution (to see step-level acceptance or to
  correlate 2026-09-19 stall captures): Channel C `detailed` on a few
  requests, or /metrics counters (rate() at 1s scrape) — both without a
  code patch.

## 6. Open questions for the on-rig verification run

1. **Line present?** `grep -c "SpecDecoding metrics" server.log` on the
   current boot (after some traffic). If 0 with traffic and spec decode on
   → fork removed/patch-moved it; fall back to /metrics counters
   (`curl -s localhost:PORT/metrics | grep spec_decode`) or
   `--per-request-spec-decode-metrics detailed`.
2. **Stats gate:** confirm the serve commandline — does it pass
   `--disable-log-stats`? (and what `VLLM_LOG_STATS_INTERVAL` env? default
   10 s.)
3. **Boot A semantics:** what exactly does MTP_NUM_SPECULATIVE_TOKENS=0 do —
   `speculative_config=None` (no line, no counters) vs 0-token config? Look
   at the launch command / `Initializing a V1 LLM engine` config dump.
4. **DP/engines:** TP4+EP4 with data_parallel_size=1 → 1 engine → 1 line per
   interval. Confirm no `Engine NNN:` variants (N>0) appear.
5. **Runner independence sanity check:** for MRV2, verify the returned
   `generated_token_ids` lengths feed the scheduler (fork optimistic-accept
   correction, audit-src/gpu_model_runner.py:1492-1516, 1694-1710) — quick
   check: `Accepted` ≤ `Drafted` always and per-pos vector ≤ 1.0 in samples
   from both eras.
6. **Fork drift on scheduler:** the fork's scheduler snapshot was not in the
   workspace (audit-src has connector.py, gpu_model_runner.py, utils.py
   only) — if Q1 fails, diff fork `vllm/v1/core/sched/scheduler.py` +
   `vllm/v1/spec_decode/metrics.py` against upstream v0.26.1rc0 before
   patching anything.

## References

- vLLM docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/
- vLLM docs (API, PromQL): https://docs.vllm.ai/en/stable/api/vllm/v1/spec_decode/metrics/
- Upstream files cited (tag v0.26.1rc0 or main, fetched 2026-09-19 via
  raw.githubusercontent.com): vllm/v1/spec_decode/metrics.py,
  vllm/v1/metrics/loggers.py, vllm/v1/metrics/stats.py,
  vllm/v1/core/sched/scheduler.py, vllm/v1/engine/{llm_engine,async_llm,core}.py,
  vllm/engine/arg_utils.py, vllm/config/observability.py, vllm/envs.py.
- Local snapshots: flashnext-scout/v3-src/{model_runner,connector}.py,
  flashnext-scout/audit-src/{gpu_model_runner,utils,connector}.py.

# 2026-09-23 — Side-lane quick-look, tool-structure verdict, MNS-16 stage, prod restore

Phase record for the side-lane (their stack) quick-look + corrections cycle. Harnesses live in
`C:/Users/imrya/flashnext-recipe-sidelane/` (untracked dir); box copies under `/home/bonz/`.
Outputs: `/home/bonz/.run/sidelane-quicklook/`. Ledger: `/home/bonz/.run/boot-ledger.txt`.

## Corrections cycle (user)

1. **MNS 16 never happened** — quick-look soakfix ran the whole day at engine MNS 4
   (`--max-num-seqs 4` in their launcher's Cmd); 16-way runs just queued. Both original
   soakfix numbers are 4-concurrent capacity.
2. **Tool-call fidelity must be structural** — parse `tool_calls` both sides, compare function
   name + arguments as JSON objects, not text; paste one raw side-lane call for Hermes-format check.

## Tool-call structural verdict (reverses the text-based read)

Side lane: clean canonical calls, 4/4 runs — `get_weather({"location": "Paris"})`,
finish_reason `tool_calls`, `arguments` is a JSON **string**, standard OpenAI shape
(Hermes-compatible; raw message JSON banked in session transcript).

Banked production reference: **`{"locationlocation": "Paris"}` in 3 of 4 runs** (tool1_RUN1,
tool2_RUN1, tool2_RUN2; tool1_RUN2 clean) — valid JSON, doubled key, prod-side defect.
The 4 fidelity DIVERGEs were comparator artifacts (tool rows have empty content → text
agreement 0/0 forced fail). Fix: `sidelane-fidelity.py` now compares tool rows structurally
(`tc_struct`: name + canonical args JSON; verdicts `TOOL_MATCH`/`TOOL_DIFF`; tool rows exempt
from the text-agreement gate). md5 `69311df8d89ada8a89352cd2c667d09c` (local = box).

## Long gates (MNS 4 boot, PC-ON) — all CORRECT

- Salted ~97K ×3: 98,211 engine ptok, CORRECT=YES ×3, TTFT 35.30/35.44/35.36 s.
- ~250K needle: 250,700 engine ptok, CORRECT=YES, TTFT 147.25 s (MML 262144 holds).
- Full fidelity 18/18: **long32k AGREE 1.0000, long80k AGREE 1.0000** — the dense-QSA
  "divergence appears at long context" prediction did NOT materialize. 7 DIVERGE total =
  4 tool comparator artifacts + 2 code1 (real behavioral: side runs long, prod stops) +
  1 reason1_RUN1 (temp-0 + PC split; run2 matches). Verdict stays
  DIFFERENT_MODEL_VARIANT (Ryan's call), divergence evidence now purely short-row.

## MNS-16 stage (reboot: `--max-num-seqs 16` as ONLY change, reconstructed from
docker inspect: same image/devices/binds/shm/port/env/Cmd otherwise)

- soakfix 16×600: sustained_agg **626.3** (r2–r4, 610.9–632.6), per_stream ~39.1, 0 errs
- soakfix 8×600: sustained_agg **338.3** (335.5–339.3), per_stream ~42.3, 0 errs
- single-stream: median19 **52.5** (52.3–52.8), 19/19 OK
- vs MNS 4 (181.7 / 181.1 / 52.3): 16-way **×3.45**, 8-way **×1.87**, 1-way flat. MNS 16 is
  the right operating point for aggregate; per-stream rates unchanged.

## Production restored (22:36–22:52Z)

es-lane stopped 22:36:59Z; `docker start qwen38-flash-next` (untouched digest);
args receipt 81920/qwen3.8-flash-next/TP4+EP/PC via vllm defaults; startup ~22:48Z;
receipt CONTENT `391` FINISH=stop (37 ctok; the 16-token window truncates in reasoning
phase = known parser shape). **Watchdog correction:** restore script's nohup guessed
`/home/bonz/wedge-watchdog.sh` (doesn't exist) — real one is
`/home/bonz/acceptance-v2/wedge-watchdog.sh`, launched + verified (PID 106523, gen-probe,
restart×3 armed). Ledger corrected.

## Watchdog-path gotcha (for future restore scripts)

`acceptance-v2/wedge-watchdog.sh`, NOT `/home/bonz/wedge-watchdog.sh`. Verify with
`ps aux | grep acceptance-v2/wedge-watchdog` after any restore.

## Open

- Fix-bundle gates (both flags=1, MML 98304, PC-ON; Gates A/B/C + reason1 canary, census
  every gate) — images ready, cards now free.
- /data 92% used / 18 G free — report-only standing state.
- Authenticated fresh-host pull test; 22:47 worker-attach stall OPEN; full ceiling gate deferred.

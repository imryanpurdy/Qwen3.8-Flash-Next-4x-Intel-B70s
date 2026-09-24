# es-lane watchdog (side-lane wedge watchdog)

`scripts/wedge-watchdog-eslane.sh` adapts the production wedge watchdog for the
side-lane stack. Diff discipline vs `wedge-watchdog.sh` (production): every
changed line is one of — container name, port, served model id, log source,
restart command, boot-grace comment, or a comment. Xe2 Level-Zero wedge
signatures are unchanged (same silicon, same Level Zero stack).

## Differences from the production watchdog

| Aspect | Production | es-lane |
|---|---|---|
| Container | `qwen38-flash-next` | `es-lane` |
| Port | 8021 | 8022 |
| Served model id | `qwen3.8-flash-next` | `qwen-256k` |
| Log source | repo `.run/server.log` (host tee follower) | `docker logs es-lane` (bounded: size=last 2000 lines, tail=last 200) |
| Restart command | `start.sh --launch` (repo copy) | `ESLANE_RESTART_CMD` (default `start.sh --launch`) |
| Boot time | ~8-9 min | ~4-5 min |
| State dir | `.run/` | `.run/eslane/` (`watchdog-eslane.log`, `wedge-<ts>.log`) |

Liveness is unchanged: **gen-probe** — a 1-token chat completion proves the
executor advances (a wedged engine can hold `/v1/models` up). Capture-first:
py-spy python-frame dumps of TP workers + EngineCore BEFORE any restart.
Cold-load guard unchanged: no stall counting until `/v1/models` answers once,
so the shorter boot needs no separate grace constant. Double opt-out unchanged
(`WEDGE_WATCHDOG_DISABLE=1` AND `PREFLIGHT_SKIPPED=1`); max 3 restarts, then
gives up loudly.

## Deploy-location gotcha (bit us 2026-09-23, ledger-corrected)

Deploy directories carry their own watchdog copy (acceptance-v2 pattern:
`/home/bonz/acceptance-v2/wedge-watchdog.sh`). Restore scripts MUST use the
deploy-dir path. `/home/bonz/wedge-watchdog.sh` does **not** exist — a restore
script that guesses it fails silently (nohup launches, file missing, instant
death, watchdog absent until someone checks). Always verify with
`ps aux | grep -E "wedge-watchdog"` after any restore.

## Restart-path test

`tests/watchdog-restart-test-eslane.sh` — same contract as the production
test: engine serving → simulated wedge (kill vLLM inside the container) →
watchdog detects via gen-probe → py-spy capture → restart → `/v1/models`
answers again → PASS line asserts it was the watchdog's restart path.
Run AFTER verify.sh; the es-lane watchdog must be running first.

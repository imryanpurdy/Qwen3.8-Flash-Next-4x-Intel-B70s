# Wedge watchdog

`scripts/wedge-watchdog.sh` watches the serving engine for the Xe2 Level-Zero
wedge (mechanism + captures: [`../docs/notes/gdn-l0-wedge.md`](../docs/notes/gdn-l0-wedge.md)).

| Aspect | Value |
|---|---|
| Container | `qwen38-flash-next` (`CONTAINER_NAME` in `.env`) |
| Port | 8022 (`PORT` in `.env`) |
| Served model id | `qwen-256k` (`SERVED_MODEL_NAME` in `.env`) |
| Log source | `docker logs $CONTAINER_NAME` (bounded: size=last 2000 lines, tail=last 200) |
| Restart command | `start.sh --launch` (launch-only; skips the XPU gate) |
| Boot time | ~4–5 min |
| State dir | `.run/` (`watchdog.log`, `wedge-<ts>.log`, `watchdog.pid`) |

Liveness is a **gen-probe**: a 1-token chat completion proves the executor
advances (a wedged engine can hold `/v1/models` up). Capture-first: py-spy
python-frame dumps of TP workers + EngineCore BEFORE any restart, so a wedge
is always diagnosed even when the restart fixes it.

Cold-load guard: no stall counting until `/v1/models` answers once, so the
~4–5 min boot needs no separate grace constant.

Disabling requires the double opt-out: `WEDGE_WATCHDOG_DISABLE=1` **and**
`start.sh --no-preflight`. Max `WEDGE_WATCHDOG_RETRIES` (default 3) restarts,
then it gives up loudly.

## Restart-path test

`tests/watchdog-restart-test.sh`: engine serving → simulated wedge (kill vLLM
inside the container) → watchdog detects via gen-probe → py-spy capture →
restart → `/v1/models` answers again → PASS line asserts it was the
watchdog's restart path. Run AFTER `tests/verify.sh`; the watchdog must be
running first.

# Scripts

All scripts run **on the host**, not inside the container, unless stated.
Production lifecycle entrypoint is `start.sh` (this directory). The devan-fork
rollback launcher lives at `../rollback/devan-fork/start.sh` (see
`docs/operations/rollback.md`).

| Script | Purpose |
|---|---|
| `start.sh` | Production (Lumnus lane) entrypoint: preflight → weights tree check → pre-boot XPU gate → launch → ready gate; spawns the wedge watchdog. Flags: `--launch --no-preflight --dry-run --replace` (`--replace` = explicit permission to stop a running container); subcommands `status/logs/stop/restart` |
| `stop.sh` | Graceful stop: watchdog TERM first, then the container (order matters) |
| `../check-weights.sh` | W4A16 snapshot presence + pinned-rev identity check; hard-fail on wrong/missing (wrong-weights guard) |
| `wedge-watchdog.sh` | Xe2 Level-Zero wedge watchdog: liveness gen-probe, py-spy capture first, restarts ≤ `WEDGE_WATCHDOG_RETRIES`, gives up loudly. Mandatory; start.sh refuses without it |
| `needle_probe.py` | Engine-calibrated long-context needle harness (v2.1): self-correcting prompt sizing, salted retrieval, CORRECT/SIZE_OK verdicts; served model via `NEEDLE_MODEL_ID` (default `qwen-256k`) |
| `soakfix.py` | Sustained multi-stream soak (the 32×600 gate): N concurrent streams × 4 rounds, r1 warmup discarded |
| `single-stream.py` | Single-stream latency: median of 19 sequential runs (first discarded as cold) |
| `toolcall.py` | Tool-calling battery: structural equivalence on function name + argument JSON objects |
| `host-setup.sh` | One-time host provisioning (kernel, firmware, grub, Docker limits); idempotent; **reboots** |

Never commit `.env` or `.run/`.

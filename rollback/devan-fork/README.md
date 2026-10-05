# Rollback / devan-fork kit

Pre-overhaul serving line: devan-carlin `vllm@xpu-qwen4exp` (a69fba21) on the
digest-pinned es-lane image — the rollback lane of `docs/operations/rollback.md`,
mirroring the deployed last-known-good launcher `~/mns32-lane/` on the rig
(devan fork + **wtdcode AWQ** checkpoint, **MNS 32**, fp8 KV — the lane that
measured the production baseline, 1,015 tok/s @32). The older MNS-4 launcher
(`~/es-lane-launch/start-qwen-256k-vllm.sh`, devan W4A16) is a historical
footnote, not the rollback of record.

- `./start.sh` / `./stop.sh` — same design rules as `scripts/start.sh`
  (validate → preflight → weights gate → XPU gate → image gate → launch →
  READY gate; mandatory watchdog; stop/restart/status/logs subcommands).
- `cp .env.example .env` first — the kit's env is its OWN (es-lane knobs);
  the repo-root `.env` belongs to the production lane. `.env` and `.run/`
  here stay out of git.
- Reuses the repo's `docker/gate.py` and `scripts/wedge-watchdog.sh`
  (resolved via the repo root); restarts itself, never the production lane.

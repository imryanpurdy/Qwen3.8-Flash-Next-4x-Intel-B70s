# Scripts

All scripts in this repo run **on the host** (the rig), not inside the container, unless
stated. Lifecycle entrypoint is `../start.sh`; the header comment of each script names the
deploy-contract rule (docs/lanes/deploy-kit-contract.md) it implements.

## Root (deploy kit)

| Script | Purpose | Contract / doc |
|---|---|---|
| `../build-image.sh` | Builds local image `qwen38-flash-next-xpu:4xb70`: pinned base (`intel/llm-scaler-vllm:0.21.0-b1`), pinned toolchain, 16-patch vLLM overlay, QSA kernels, certified kernel stage `2f829747` (sha-verified). Refuses without overlay artifacts | `files/overlay/README.md` |
| `../start.sh` | One entrypoint: preflight → weights (auto-skip/download) → `check-weights.sh` identity gate → wedge watchdog → `docker run` (`/dev/dri` passthrough) → `/v1/models` ready-poll. Flags: `--no-download --no-launch --launch --no-preflight`; serialized by a lockfile | `docs/lanes/deploy-kit-contract.md` D2 |
| `../stop.sh` | Graceful stop: watchdog TERM first, then container (order matters — contract D3) | D3 |
| `../wedge-watchdog.sh` | Xe2 Level-Zero wedge watchdog: captures `.run/wedge-<ts>.log`, restarts ≤`WEDGE_WATCHDOG_RETRIES`, gives up loudly. Mandatory; start.sh refuses without it | D3 |
| `../check-weights.sh` | FP8 snapshot presence + frozen identity hash check; hard-fail on wrong/missing (wrong-weights guard) | D6 / `docs/lanes/00-shared-context.md` |
| `../.env.sample` | Template for `.env` (gitignored); annotated defaults = measured lab identity | D4 |

## scripts/

| Script | Purpose |
|---|---|
| `rebuild-verify.sh` | **L1–L5 post-rebuild verification ladder as one gate script** (`PORT= MODEL= MNS= BURSTS= CONC=` env). L1 readiness, L2 generation probes, L3 soak/stall-spy, L4 rate check, L5 summary. Runbook §4: [`../docs/rebuild/2026-09-19-platform-rebuild-runbook.md`](../docs/rebuild/2026-09-19-platform-rebuild-runbook.md). Litmus: L3 loop/dumper caps + gate floor ≥300 s; `STALLSPY_*` capped |
| `dl-int4.sh` | Stage the INT4 W4A16 AutoRound snapshot (`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`) with HOST python3 (`huggingface_hub.snapshot_download`) — no docker, no worker SSH |
| `enumerate-pre-wipe.sh` | Rig inventory enumeration before the OS wipe (part of rebuild preflight) |
| `mtp4-stall-evidence.sh` | Synthetic 4K MTP4 run for the stall hypothesis ladder (H1–H4 controls; post-rebuild) |
| `soakfix.py` | Host-side soak client (used by rebuild-verify L3; drop-in `/tmp/soakfix.py` when URLs match) |
| `patch-capsizes.py` | Capture-size cliff root-cause/patch helper (2026-09-20) |
| `gh-sweep/` | GitHub tracker-sweep tooling that produced `../docs/gh_*.json` (offline research; not part of serving) — `gh_search.py`, `gh_fetch.py`, `gh_phase2.py`, `gh_sweep.log` |

## Adding a script

Operational/deployable scripts live here (or root if the launcher depends on them);
one-off campaign probes used the rig get a one-line row in this table. Never commit
`.env`, `.run/`, or overlay `*.patch`s.

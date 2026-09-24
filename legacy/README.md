# Legacy kit — the sparse-QSA production stack (v1)

This directory keeps the **previous production stack** deployable as the
rollback target after the side-lane stack became the deploy of record
(2026-09-23). Run everything from `legacy/` with its own `.env`
(`cp legacy/.env.example legacy/.env`).

## What this stack is
Sparse-QSA W4A16 serving, TP4+EP, MML **98304** (98K validated; ~130K dies
mid-prefill; 170K DEVICE_LOST), MNS 16, MTP 0, graph mode eager + decode
graphs, kv-cache-bytes pinned, t120 PLE-staging image. The full validated
line, numbers, and evidence live in `legacy/README-full.md` (the old README,
preserved) and `legacy/verify.sh` (carries its own numbers + harness/formula
headers per the comparability law).

## Why it went legacy
- **Prefix-cache crash**: cached-state corruption class (P1/U1); production
  hit malformed tool-call args (`locationlocation`) and restarts under load.
- **Interim ceiling 81K**: the interim ceiling is 81920 ctx — half the
  side-lane's 262144.

## When you'd still use it
- **Exact sparse-attention fidelity**: any workload needing the sparse QSA
  attention behavior of the v1 stack (the side lane is the dense-full-context
  QSA variant — a *different model variant*; see
  `docs/evidence/2026-09-23-qsa-dense-verdict.md`).
- Fresh-host pulls without building: this image is on GHCR
  (`ghcr.io/imryanpurdy/qwen38-flash-next-b70@sha256:0ca85985…66fc2`,
  private package — `docker login ghcr.io` first). The side-lane image is
  local-build-only until its GHCR push.

## Contents
| File | Role |
|---|---|
| `start.sh` | v1 launch/validation/readiness/watchdog kit (unchanged, digest-pinned image) |
| `.env.example` | v1 validated line (MML 98304, kv-bytes pin, EXTRA_DOCKER_ARGS contract) |
| `verify.sh` | v1 acceptance battery (186 checks; numbers + harness/formula headers) |
| `wedge-watchdog.sh` | v1 wedge watchdog (container qwen38-flash-next, :8021) |
| `watchdog-restart-test.sh` | v1 watchdog restart-path test |
| `README-full.md` | The old README, preserved verbatim (runbook + evidence links) |

## One-line rollback
**Roll back:** from the repo root run `cd legacy && cp .env.example .env && ./start.sh`
(v1 stack: pinned GHCR digest, port 8021, `qwen3.8-flash-next`) — ~8–9 min to READY.

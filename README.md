# Qwen3.8-Flash-Next on 4x Intel Arc Pro B70

Serving kit for **Qwen3.8-Flash-Next INT4 (W4A16-AutoRound)** on **4x Intel Arc Pro B70 32GB, TP4+EP4** — the measured, gated, 2026-09-22/23 production line: MTP0, 98K context, decode graphs (FULL_DECODE_ONLY, capture list to 32), qwen3_xml/qwen3 tool/reasoning parsers, exact KV pool pin, PLE n-gram host offload, t120 PLE-staging image.

**Repo = the deliverable.** Clone → `cp .env.example .env` → `./start.sh` → serves. Every number below comes from `tests/verify.sh` output on the pinned image; docs cite primary artifacts (boot ledger, server logs, md5s, raw battery output) in `docs/`.

## Hardware

| Component | Requirement |
|---|---|
| GPU | 4x Intel Arc Pro B70 32GB (Xe2 / Battlemage) |
| Host RAM | ≥100 GiB available (PLE table pins ~51 GiB + 4x ~31 GiB device loads; 128 GB installed) |
| Swap | ≥64 GiB ON |
| Kernel | 6.17.0-1010-intel (platform of record; `scripts/host-setup.sh`) |
| GuC firmware | 70.65 — sha256 `70d74627e395…67bb` (linux-firmware fb0889c0, 377,664 B) |
| IOMMU | off (grub `iommu=off`) |
| Userspace | intel-omix 0.4 (DLE 2026.1: oneCCL 2022.1.2, level-zero) |
| Docker | nofile ulimit ≥ 1M (LimitNOFILE=infinity) |
| Weights | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`, ~186 GB tree, identity-gated by `check-weights.sh` |

## Measured numbers

Every row names its harness, aggregate formula, and prompt shape. **Rows are not comparable across harnesses** (2026-09-23 reconciliation — `docs/rebuild/2026-09-23-measurement-reconciliation-soakfix-vs-bench-harness.md`). This table is produced by `tests/verify.sh`; acceptance run of record 2026-09-23: fresh clone `1efffe9` off GitHub, pinned image digest `sha256:0ca85985…66fc2`, kernel 6.17.0-1010-intel (iommu=off, GuC 70.65.0):

| Metric | Value | Harness / formula / prompt |
|---|---|---|
| **16×600 sustained (PRODUCTION)** | **299.4 tok/s** (r2–r4 of record; r1 warmup 150.8; spread 150.8–306.5) | soakfix.py · Σcompletion_tokens÷wall, sustained=mean(r2–r4) · open-ended essay prompt, runs TO the 600 cap |
| 8×600 sustained (PRODUCTION) | 175.6 tok/s | soakfix.py · same formula · same prompt |
| 16×320 short-burst (SECONDARY) | 164.469 tok/s median (BEST 195.242 / WORST 152.157) | bench_harness.py burst · same formula · thank-you-note prompt, EOS-stops ~110–130 tok/stream (ramp+drain slice, NOT comparable to soakfix) |
| Single-stream (N=20, first discarded) | 27.2 median (min 22.9, max 27.4) | inline verify.sh · ctok÷wall, median of 19 · essay request, 600 tok |
| Tool calls | 5/5 well-formed (gate) | 5 sequential /v1/chat/completions, get_weather tool, temp 0 |
| 98K needle | **PASS @ 97,754 engine-confirmed tokens** — CORRECT=YES (code `QRX-88-SHELDON`), FINISH_REASON=stop, STAGING_NEW=0, TTFT 262.8 s | needle_probe.py v2.1 (engine-calibrated, sizes to the tokenizer via usage.prompt_tokens), temp 0 |
| Context ceiling | 98,304 — 130K dies mid-prefill; 170K DEVICE_LOST (error-20) | needle probes 2026-09-22 (ledger B4T-NEEDLE170/130); 98,179 also served clean on 6.17 (2026-09-23) |
| KV pool | 671,232 tokens; 6.83× concurrency @ 98K | server log 2026-09-23 boot |
| 98K needle prefill | ~372 tok/s (97,754 tok, 262.8 s wall) | verify.sh run of record 2026-09-23 |
| Watchdog restart-path | PASS — wedge → watchdog relaunch via `--launch` → serving again | tests/watchdog-restart-test.sh, 2026-09-23 |

Historical cross-checks (same harness, platform of record): Saturday 2026-09-20 soakfix r2–r4 = 314.5/310.4/320.1; tonight's 299.4 is inside the boot-to-boot band of the 320 boot (±2.1% 1σ, n≈20).

## Quick start

```bash
git clone https://github.com/imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s
cd Qwen3.8-Flash-Next-4x-Intel-B70s
scripts/host-setup.sh          # once per host; sudo; REBOOT after (kernel+iommu+GuC)
docker login ghcr.io           # the image package is private
cp .env.example .env           # the production v1 line; every knob commented
./start.sh                     # preflight → weights → pull pinned image → launch → READY gate
./tests/verify.sh              # gates + the numbers table (production metric first)
```

`./start.sh` also supports `start|stop|restart|status|logs`; `--launch` is the watchdog's restart path. `./start.sh stop` = watchdog first, then the container.

## Knobs

All in `.env.example`, each commented there. The ones that bite:

- `MAX_MODEL_LEN=98304` — measured ceiling; start.sh refuses >98304.
- `MTP_NUM_SPECULATIVE_TOKENS=0` — MTP1 corrupts output on this image; non-zero refused.
- `MAX_NUM_SEQS=16` + `CAP_SIZES_LIST=1,2,3,4,5,6,7,8,12,16,24,32` — decode graphs must cover MNS (uncaptured width = eager cliff); transitional sizes 9–11/13–15/17–23 fault at capture.
- `MAX_NUM_BATCHED_TOKENS=2048` — >4096 = crash class (QSA indexer).
- `EXTRA_VLLM_ARGS` — `--kv-cache-memory-bytes 9494279680` (exact pool; removing re-derives it and taxes every decode step via PLE gather), parsers, `-O 0`, gpu-util 0.75, `--long-prefill-token-threshold 1024`.
- `EXTRA_DOCKER_ARGS` — v1 container set incl. `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` **PERMANENT** (oneCCL #212 stale-IPC-handle hang; no upstream fix in 2021.x/2022.x; measured cost undetectable, n=4). Do not strip.
- `WEDGE_WATCHDOG_*` — mandatory watchdog; disable requires the double opt-out (`WEDGE_WATCHDOG_DISABLE=1` **and** `--no-preflight`).
- `PREFLIGHT_*` floors — 4 XPUs, RAM, swap, **kernel**, **GuC hash**, **iommu=off**, disk.

## Known limits

- **98K context ceiling** — 130K dies mid-prefill, 170K DEVICE_LOST (error-20) at PLE staging; timing-based, not position/overflow (ledger CEILING-CORRECTIONS). Detail: `docs/rebuild/2026-09-22-device-lost-130k-170k.md`.
- **MTP1 unusable on this image** — temp-0 A/B: MTP1-miss / MTP0-clean (`docs/rebuild/2026-09-22-mtp1-corruption-temp0-diff.md`).
- **Xe2 Level-Zero wedge** every 2–6 h under load — watchdog captures py-spy + logs, restarts ≤3×; in-flight requests lost on restart.
- **Kernel 7.0.0-31 (HWE)**: serves fine and matches sustained numbers, but 98K KV is not reachable there (v1 OOM ×3 — NEO host-GTT mirror; `docs/rebuild/2026-09-22-neo-host-gtt-mirror-on-7.0.md`). Platform of record is 6.17.
- **CCL flag** must stay (`EXTRA_DOCKER_ARGS`) — removal risks the oneCCL #212 permanent allreduce hang.
- **Private GHCR package** — `docker login ghcr.io` before first pull.

## Troubleshooting

- **Preflight: GuC hash mismatch** → `scripts/host-setup.sh` (installs + hash-verifies 70.65; reboot).
- **Preflight: kernel not recognized** → same script installs 6.17.0-1010-intel; reboot; `uname -r` must show it.
- **Pull denied / image not found** → `docker login ghcr.io`; check `IMAGE` digest in `.env`.
- **READY GATE FAIL (no `Graph capturing finished` lines)** → graphs didn't capture; serving would fall back to eager (cliff). Inspect `.run/server.log` around capture; keep the v24h2 `CAP_SIZES_LIST`.
- **First answer garbled after a manual restart** → PLE cold-table page-in tore embeddings; fire any tool-call request first (start.sh does this automatically; the connector now tolerates 120 s staging).
- **Sudden 0 tok/s, `engine_reset` in journal** → wedge; watchdog handles it; check `.run/wedge-*.log` for captures; in-flight lost.
- **Wrong/failed tool calls** → parsers must stay in `EXTRA_VLLM_ARGS` (`--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`).
- **16-way number far below expectations** → are you reading soakfix sustained (production) or bench_harness short-burst (secondary)? Same formula, different regime — `docs/rebuild/2026-09-23-measurement-reconciliation-soakfix-vs-bench-harness.md`.
- **Port change** → set `PORT` AND `-p 8021:8021` inside `EXTRA_DOCKER_ARGS`; both must match.
- **`git describe: no-git`** in manifest → harmless; kit runs fine outside a git checkout.

Campaign and incident history: `docs/` (one doc per event, from primary artifacts). License: [LICENSE](LICENSE).

# Qwen3.8-Flash-Next on 4× Intel Arc Pro B70 (vLLM XPU, TP4+EP4)

Deploy recipe + runbook for serving **Qwen3.8-Flash-Next** (125B-main / 6B-active MoE,
48-layer GDN+QSA hybrid attention, PLE n-gram table host-offloaded) on **four Intel Arc
Pro B70 32 GB** with vLLM XPU: **TP4+EP4**, INT4 **W4A16 AutoRound** (primary target) /
FP8 (lab baseline), MTP speculative decode, XPU graph capture, wedge watchdog.

> **STATUS — RECIPE NOT FINALIZED.** The rig is mid-OS rebuild (OMIX 0.4 project,
> pending; see [`docs/rebuild/`](docs/rebuild/)). The FP8 line is the measured lab
> baseline (campaign 2026-08-28 → 09-19); the **INT4 W4A16 line is the finalized serving
> target** and is validated only to first-serving (2026-09-18) plus staging. Nothing on
> this box has been re-verified since the 2026-09-20 pre-wipe preflight. Clone it, read
> [`docs/README.md`](docs/README.md), and treat every number here as lab-measured, not
> promised.

---

## What you need

| Item | Requirement |
|---|---|
| GPUs | 4× Intel Arc Pro B70 32 GB (Xe2), all 4 visible to the container preflight gate |
| Host RAM | **≥128 GB ECC** (post-rebuild floor; PLE pins ~51.2 GiB + engine + loader). Preflight gate fails below 100 GB |
| Swap | ≥64 GiB, left ON (compile-time headroom; 64 GiB swap is a floor, NOT the fix for graph-compile OOM) |
| Weights volume | ≥200 GB free (checkpoint tree = 185.56 GB / 131 shards). Post-rebuild: NVMe formatted ext4/xfs, mounted at `/data` |
| OS disk | ≥40 GB free |
| OS | Ubuntu 24.04 LTS + HWE kernel 6.17 (**OMIX 0.3/0.4 pin**) — kernel 7.0 + PPA UMD mixes are the permanent-wedge correlate |
| Docker | docker CLI + daemon on the host; `--group-add video/render` handled by the launcher |
| Network | access to HuggingFace (weights), GitHub (patches/kernel stage), and expected GHCR/registry read for the base image |
| HF token | optional in `.env` (only if the checkpoint download needs auth); `.env` is gitignored |

Full BOM: [`docs/rebuild/2026-09-19-omix-target-bom.md`](docs/rebuild/2026-09-19-omix-target-bom.md) · hardware-transition recipe: [`docs/rebuild/phase3-hardware-upgrade.md`](docs/rebuild/phase3-hardware-upgrade.md).

## Quick start (clone → serve)

```bash
git clone https://github.com/imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s.git
cd Qwen3.8-Flash-Next-4x-Intel-B70s

cp .env.sample .env        # edit: PORT / HF_TOKEN / (INT4 track: MODEL_ID + rev, see below)
./build-image.sh           # local IMAGE from Dockerfile: pinned base + toolchain + 16-patch overlay + QSA kernels
./start.sh                 # preflight → weights → check-weights identity gate → watchdog → docker run → /v1/models
curl http://localhost:8000/v1/models
```

`./start.sh` does, in order: (1) preflight (4×XPU, RAM/swap/disk floors, hard fail with
actionable errors; `--no-preflight` records `PREFLIGHT_SKIPPED=1`), (2) weights auto-skip
when a sane HF snapshot exists, else download, (3) `check-weights.sh` identity gate on the
frozen hash — hard fail, (4) mandatory wedge watchdog, (5) `docker run` with `/dev/dri`
passthrough, (6) readiness poll on `/v1/models` (185 GB tree; minutes). Stop:
`./stop.sh` (watchdog TERM first, then container).

**Two checkpoint tracks:**

| Track | Weights | Runbook / repo identity | Status |
|---|---|---|---|
| **INT4 (target)** | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897` | [`scripts/dl-int4.sh`](scripts/dl-int4.sh) stages to `HF_HOME` (host-side, no container); serving identity + campaign: [`docs/campaigns/2026-09-18-int4-first-serving-overnight.md`](docs/campaigns/2026-09-18-int4-first-serving-overnight.md) | first serving 2026-09-18; graph-capture + MTP target documented; **final validation pending rebuild** |
| FP8 (baseline) | `Qwen/Qwen3.8-Flash-Next-FP8` (hash `bcd9f01d…`, frozen in `check-weights.sh`) | `cp .env.sample .env` (defaults = FP8 identity, TP4+EP4, MTP3, eager) | measured lab baseline; this is what `start.sh`/`build-image.sh` validate today |

## What runs (server identity)

| Layer | Value |
|---|---|
| API | vLLM OpenAI-compatible on the host, port 8000 (`.env PORT`) |
| Model id | `qwen3.8-flash-next` (`SERVED_MODEL_NAME`) |
| Weights | 185.56 GB FP8 tree / or INT4 W4A16 AutoRound snapshot — identity hard-gated by `check-weights.sh` |
| Image | local `qwen38-flash-next-xpu:4xb70` from [Dockerfile](Dockerfile): `intel/llm-scaler-vllm:0.21.0-b1` (pinned digest) + py 3.12.3 + torch 2.11.0+xpu + triton-xpu 3.7.0 + overlay 0001–0018 + QSA kernels + kernel stage `2f829747` (sha-verified, two-part prerelease) |
| Parallelism | TP4 + EP4 (TP must divide the 2 KV heads → TP ∈ {2,4,8}; TP6 is impossible) |
| Spec decode | MTP k=3 @ 4K (target); k=4 @ 4K **quarantined** (3,904/4,096 stall) |
| PLE | CPU offload 12.25 GB/rank ≈ 51.2 GiB pinned host RAM (`VLLM_PLE_CPU_OFFLOAD` gate) |
| KV | `MAX_MODEL_LEN=4352` lab identity; 8K contexts NOT MTP-qualified |
| Graphs | `GRAPH_MODE=eager` is the only value `start.sh` accepts today (FP8 a1–a7 compile-OOM quarantine); INT4 target uses `VLLM_XPU_ENABLE_XPU_GRAPH=1` with `-O 0` — guard lift is part of the rebuild work (see [`docs/campaigns/2026-09-18-graph-mode-breakthrough.md`](docs/campaigns/2026-09-18-graph-mode-breakthrough.md)) |
| Scheduler | `MAX_NUM_SEQS=1`, `MAX_NUM_BATCHED_TOKENS=64` frozen lab baseline — Lane-4 sweep 512/2048/4096 pending |

## Verify it came up

`./start.sh` prints these greps against `.run/server.log` with the EXPECTED values:

| Check | log pattern | EXPECTED (lab anchor) |
|---|---|---|
| KV pool | `KV cache size` | MTP3-4K authority geometry **294,195,200 B (25 blocks)**; actual scales with `MAX_MODEL_LEN` |
| PLE placement | `uva` / `offload` / `pinned` | **12.22 GiB/rank (13,117,911,040 B) × 4 ≈ 51.2 GiB** pinned host RAM (`cpu_offload_gb=12.25`) |
| Served model | served model name | your `SERVED_MODEL_NAME` answering on `/v1/models` |
| Graph status | `eager` | **eager** — no graph flags passed (graphs quarantined-negative a1–a7) |

First-request sanity: `curl http://localhost:8000/v1/models` lists the served name.
After the OS rebuild, the full gate is [`scripts/rebuild-verify.sh`](scripts/rebuild-verify.sh)
(L1–L5 ladder — runbook [`docs/rebuild/2026-09-19-platform-rebuild-runbook.md`](docs/rebuild/2026-09-19-platform-rebuild-runbook.md) §4).

## Scripts

| Script | What it does |
|---|---|
| `build-image.sh` | builds the local image (refuses without overlay artifacts; see `files/overlay/README.md`) |
| `start.sh` | single lifecycle entrypoint: preflight → weights → identity gate → watchdog → launch → ready-poll (`--no-download/--no-launch/--launch/--no-preflight`) |
| `stop.sh` | graceful stop (watchdog TERM first) |
| `check-weights.sh` | FP8 snapshot presence + frozen-identity check (wrong-weights guard) |
| `wedge-watchdog.sh` | Xe2 Level-Zero wedge watchdog (mandatory; refs in start.sh) |
| [`scripts/rebuild-verify.sh`](scripts/rebuild-verify.sh) | L1–L5 post-rebuild verification ladder as one gate script (runbook §4) |
| [`scripts/dl-int4.sh`](scripts/dl-int4.sh) | stage the INT4 W4A16 AutoRound snapshot (host python3 + huggingface_hub, pinned rev) |
| `scripts/soakfix.py`, `scripts/mtp4-stall-evidence.sh`, `scripts/patch-capsizes.py`, `scripts/enumerate-pre-wipe.sh` | campaign/measurement helpers |
| `scripts/gh-sweep/` | GitHub tracker-sweep research tooling (offline research, not part of serving) |

## Documentation

**[`docs/README.md`](docs/README.md)** is the index. Entry points by job:

- **Understand the box / deploy contract** → [`docs/lanes/deploy-kit-contract.md`](docs/lanes/deploy-kit-contract.md), [`docs/lanes/00-shared-context.md`](docs/lanes/00-shared-context.md)
- **Why it's slow / what's next** → lane specs 1–6 in [`docs/lanes/`](docs/lanes/) (graphs quarantine, MTP qualification, block-FP8 GEMM, TTFT sweep, A367 exact-GDN, custom kernel program)
- **The OS rebuild** → [`docs/rebuild/`](docs/rebuild/) (runbook, BOM, chainload staging, preflight audit)
- **What blew up and why** → [`docs/incidents/`](docs/incidents/) (stall/wedge forensics with captures)
- **What we changed and measured** → [`docs/campaigns/`](docs/campaigns/) (PLE v3 campaign, MTP acceptance, addenda)
- **Source material** → [`docs/evidence/`](docs/evidence/) (cookbook digest, DSv4 80 tok/s reference lane, qsa_ops.py extract)

## Measured anchors (lab campaign, immutable receipts — not targets)

| Cell | tok/s | TTFT |
|---|---|---|
| MTP0 @512 (protected anchor) | 5.515783 | — |
| MTP1 @512 | 9.372254368 | ~10–12 s |
| MTP2 @512 | 11.895061403 | ~11 s |
| MTP3 @512 | 14.888789794 | ~11 s |
| **MTP3 @4K (preferred cell)** | **15.502** | **187.9 s** |
| MTP4 @512 (screen only) | 20.727 | ~11 s |
| MTP0 @8K (screened) | 3.980 | 386.5 s |
| MTP4 @4K | **quarantined** (3,904/4,096 stall, 4-card resets) | — |
| INT4 first serving (stage-v22, MTP0 eager) | 4.9 → 21.7 aggregate @ batch 8 | ~205 ms/step (batch-invariant) |

Reference: the same box class with a fully-built stack ran DeepSeek-V4 at 80 tok/s
(PIECEWISE graphs + sparse-FP8 + sharded speculation) — the roofline proof, not a claim
for this model. Every number above carries receipts in
[`docs/lanes/evidence-lane4-ttft.md`](docs/lanes/evidence-lane4-ttft.md).

**Troubleshooting** — a wedge shows as engine-reset/`guc_exec_queue_timedout_job` in the
kernel journal; the watchdog captures `.run/wedge-<ts>.log`, restarts up to
`WEDGE_WATCHDOG_RETRIES=3` times, then gives up loudly (in-flight requests are lost on
restart). Compile-phase OOM / graphs: see [`docs/lanes/lane1-piecewise-graphs.md`](docs/lanes/lane1-piecewise-graphs.md)
(root/swap floors and a clean boot are mandatory; 64 GiB swap is NOT the fix). Slow first
token: see [`docs/lanes/lane4-ttft-chunk-sweep.md`](docs/lanes/lane4-ttft-chunk-sweep.md).

## Do not

- **`TENSOR_PARALLEL_SIZE=6`** — 2 KV heads do not divide by 6; only 2/4/8.
- **Serve MTP k=4 @ 4K** — quarantined: engine stall at 3,904/4,096 + resets.
- **Run `./start.sh` without the watchdog** — Xe2 Level-Zero wedge kills the job every 2–6 h under load; start.sh refuses unless you take the double opt-out and its red banner.
- **Raise `MAX_NUM_BATCHED_TOKENS` / `MAX_NUM_SEQS` blind** — frozen lab baselines; 8192 crashed the QSA indexer class; Lane-4 sweep is pending.
- **Flip `GRAPH_MODE` to anything but `eager`** — the `start.sh` guard exists because a1–a7 died in compile-phase OOM; the INT4 path is a separate, documented experiment.
- **Quote unmeasured numbers as targets** — spec targets stay marked as targets; only measured anchors get quoted.
- **Commit `.env`** (gitignored; contains secrets).

## License

See [LICENSE](LICENSE). This recipe is the operator's working record of a lab campaign —
error-correct, don't celebrate. Full index: [`docs/README.md`](docs/README.md).

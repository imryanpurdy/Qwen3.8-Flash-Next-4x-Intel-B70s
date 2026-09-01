# Qwen3.8-Flash-Next-FP8 on 4× Intel Arc Pro B70

First public recipe for serving **Qwen/Qwen3.8-Flash-Next-FP8** (125B-main / 6B-active MoE, 48-layer GDN+QSA hybrid attention, PLE n-gram table host-offloaded) on four Intel Arc Pro B70 32GB GPUs — vLLM **TP4+EP4** (TP must divide the 2 KV heads: 2, 4, or 8 only; 6 is impossible).

Checkpoint identity (frozen): `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` — verified by `check-weights.sh`, hard-fail on mismatch.

## Status today (measured, eager — lab campaign 2026-08-28 → 09-01)

| Config | tok/s | TTFT |
|---|---|---|
| MTP0 eager decode | 5.515783 | — |
| MTP1 @512 ctx | 9.372254368 | ~10–12 s |
| MTP2 @512 ctx | 11.895061403 | ~11 s |
| MTP3 @512 ctx | 14.888789794 | ~11 s |
| **MTP3 @4K** | **15.502** | **187.9 s** |
| MTP4 @512 ctx | 20.727 | ~11 s |

MTP3 is the preferred 4K recipe. Reference point: the same-class fully-built stack (graphs + sparse-FP8 + sharded speculation) ran DeepSeek-V4 at **80 tok/s** on this box class — that is the roofline proof, not a claim for this model.

## The four lanes to serving speed (`docs/lanes/`)

**Lane 1 — PIECEWISE graphs** (`lane1-piecewise-graphs.md`). Currently quarantined-negative (attempts a1–a7, 2026-08-28). RCA from the lab's own receipts: host-RAM exhaustion during post-load Dynamo/Inductor compile — death *before any capture ran* (≥57 GiB swap free at OOM ⇒ non-swappable demand; more swap is explicitly not the fix). Highest-leverage item; DSv4's 80 tok/s depended on graphs for both target and draft. Remediation plan P0–P6, reboot gate → H1 fix → GDN component → QSA determinism → capture ATP → full-model.

**Lane 2 — MTP qualification** (`lane2-mtp-qualification.md`). The 4B MTP head drafts fine (MTP3/MTP4 screens above) — it just isn't qualified. Five blockers ranked cheapest-first: harness lifecycle → cross-runtime parity (8K/2K vs legacy authority) → host gate (11 NVMe records) → thinking-session API 500 → engine stability (MTP4@4K 3,904/4,096 stall + 4-card resets). Includes per-k × per-context evidence tables, every cell cited.

**Lane 3 — Native XPU block-FP8 GEMM** (`lane3-block-fp8-gemm.md`). The checkpoint is FP8 running on fallbacks: dense linears = oneDNN W8A8 (`torch.ops._xpu_C.fp8_gemm`) with per-linear dynamic activation quant; routed MoE = Triton fp8_w8a8; native grouped block-FP8 is dormant upstream. Integration plan + preregistered speed/parity gates; positioned as follow-on to the rig's frozen A31 MoE test (collision rule honored, A30 −1.82% cited as known negative).

**Lane 4 — TTFT (188 s @ 4K)** (`lane4-ttft-chunk-sweep.md`). Smoking gun: **`max_num_batched_tokens=64` is frozen in every lab launcher and was never swept** — a 4K prefill = 64 serialized full-model scheduler steps. E1: prefill-phase bucket profile (reuses the corrected A28 analyzer, sha `a4d4c54c…`, 5/5 tests). E2: chunk sweep 64→512→2048→4096 with indexer-stability gate (GLM-5.3 precedent on the same QSA indexer class: 2048 required, 4096 worse, 8192 crashed). E3: fixed-overhead decomposition — linear fit TTFT = F + c·prompt; ~11 s at 317-token needles says a large fixed term exists.

## Deploy kit (contract: `docs/lanes/deploy-kit-contract.md`)

Docker-first, Mia-style clone-and-run: `cp .env.sample .env → edit → ./build-image.sh → ./start.sh → verify`. Kit scripts: `start.sh · stop.sh · check-weights.sh · wedge-watchdog.sh · build-image.sh · .env.sample · Dockerfile · files/`. Key decisions:

- **Docker-first** (decided 2026-09-01): image = vLLM XPU runtime base + pinned toolchain + overlay series 0001–0010,0012,0014–0018 + QSA kernels + certified kernel stage; `/dev/dri` passthrough with device-cgroup rules + `--group-add video/render`; host HF cache bind-mounted; weights persist outside the container.
- **Wedge watchdog mandatory**: the Xe2 Level-Zero wedge kills the job every 2–6 h under load — start.sh refuses to launch without it (double opt-out to disable: `WEDGE_WATCHDOG_DISABLE=1` **and** `--no-preflight`).
- **Preflight gate**: exactly 4×XPU visible, host RAM ≥100 GiB, swap ≥64 GiB ON, ≥200 GiB free on the weights mount (checkpoint tree = 185.56 GB), ≥40 GiB root free — fail with actionable errors; `--no-preflight` records `PREFLIGHT_SKIPPED=1` loudly.
- **Receipts**: every launch writes `.run/manifest.json` (env hash, git describe, .env minus secrets, epoch); only measured anchors are ever quoted, spec targets stay marked as targets.

### Quick Start (3 steps)

1. **Copy + edit the config** — `cp .env.sample .env`, set `HF_TOKEN` (or rely on the host HF cache mount), confirm `MAX_MODEL_LEN` and `PORT`. The defaults ARE the measured lab identity: `TENSOR_PARALLEL_SIZE=4` (TP∈{2,4,8}; 2 KV heads, TP6 impossible), `ENABLE_EXPERT_PARALLEL=true` (EP4), `MTP_NUM_SPECULATIVE_TOKENS=3` (the qualified 4K serving target), `MAX_NUM_BATCHED_TOKENS=64` (frozen lab baseline — Lane-4 sweep pending), `PLE_CPU_OFFLOAD_GB=12.25` (≈51.2 GiB pinned host), `GRAPH_MODE=eager` (graphs quarantined a1–a7).
2. **Build the image** — `./build-image.sh` builds the local `IMAGE` tag (`qwen38-flash-next-xpu:4xb70`) from `Dockerfile`: vLLM XPU runtime base + pinned toolchain (py 3.12.13, torch 2.11.0+xpu, triton-xpu 3.7.0, transformers 5.10.2) + the 16-patch vLLM overlay + QSA kernels + the certified kernel stage `2f829747`. **Fails fast** while `files/overlay/` artifacts are lab-staged, not vendored (see `files/overlay/README.md` for provenance), and while the unfrozen `BASE_IMAGE` / `RUNTIME_STAGE_URL` / `RUNTIME_STAGE_SHA256` pins are unfilled.
3. **Start** — `./start.sh`. PREFLIGHT (non-skippable) → weights auto-skip / HF download → `check-weights.sh` identity gate (frozen hash hard-fail) → mandatory wedge watchdog → `docker run` with `/dev/dri` passthrough → readiness poll (`/v1/models`, up to 15 min — a 185.56 GB tree takes minutes to load) → verification greps below.

Stop: `./stop.sh` (graceful — watchdog TERM first, then container). Flags: `--no-download` `--no-launch` `--launch` `--no-preflight` (Mia semantics + preflight opt-out). Never run without the watchdog unless you accept the double opt-out and its red banner.

### Verify it came up

`./start.sh` prints these greps against `.run/server.log` with the EXPECTED values:

| Check | log pattern | EXPECTED (lab anchor) |
|---|---|---|
| KV pool | `KV cache size` | MTP3-4K authority geometry **294,195,200 B (25 blocks)**; actual scales with `MAX_MODEL_LEN` |
| PLE placement | `uva` / `offload` / `pinned` | **12.22 GiB/rank (13,117,911,040 B) × 4 ≈ 51.2 GiB** pinned host RAM (`cpu_offload_gb=12.25`) |
| Served model | served model name | your `SERVED_MODEL_NAME` answering on `/v1/models` |
| Graph status | `eager` | **eager** — no graph flags passed (graphs quarantined-negative a1–a7) |

First-request sanity: `curl http://localhost:8000/v1/models` lists the served name. Measured anchors for this section (same cells as the Status table above — protected, never implied as deploy targets):

| Cell | tok/s | TTFT |
|---|---|---|
| MTP0 @512 (protected anchor) | 5.515783 | — |
| **MTP3 @4K (preferred cell)** | **15.502** | **187.9 s** |
| MTP4 @512 (screen) | 20.727 | ~11 s |
| MTP0 @8K (screened) | 3.980 | 386.5 s |

**Troubleshooting** — wedge watchdog behaviour: a wedge shows as engine-reset/`guc_exec_queue_timedout_job` in the kernel journal; the watchdog captures `.run/wedge-<ts>.log`, restarts up to `WEDGE_WATCHDOG_RETRIES=3` times, then gives up loudly — in-flight requests are lost on restart. OOM-during-compile / graphs: see `docs/lanes/lane1` (host-RAM pressure during post-load Dynamo/Inductor compile — root/swap floors and a clean boot are mandatory; 64 GiB swap is NOT the fix). Slow first token (TTFT 187.9 s @4K): see `docs/lanes/lane4` — `MAX_NUM_BATCHED_TOKENS=64` is the untested-axis smoking gun; the 512/2048/4096 sweep is pending.

## Evidence (`docs/evidence/`)

Lab lane maps used to build the specs: lab tree map, DSv4 80-tok/s reference lane, B70 cookbook digest (SergiioB), Flash-Next scaffold, Qwen2.7B reference lane, and `qsa_ops.py` — the merged Triton QSA kernel extract (vLLM PR #53896). The Lane-4 evidence pack (`docs/lanes/evidence-lane4-ttft.md`) carries file+line citations for every number above.

**Valid until it's on the rig:** every number here is from the lab campaign's immutable receipts. First thing the rig does is re-verify the anchors — cache-zero, cold prompts, median-of-medians.

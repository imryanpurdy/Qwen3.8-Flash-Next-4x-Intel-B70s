# Qwen3.8-Flash-Next on 4x Intel Arc Pro B70

Serving kit for **Qwen3.8-Flash-Next W4A16** on **4x Intel Arc Pro B70 32GB, TP4+EP**: vLLM fork `devan-carlin/vllm@xpu-qwen4exp` (a69fba21) built on `intel/omix:0.4.0-devel-ubuntu24.04`, **262144 context**, decode graphs, kv fp8, qwen3/qwen3_xml parsers.

**Model variant, stated plainly:** this serves the **dense-full-context QSA variant** (indexer weights dropped) — a **different model variant** from the sparse-QSA checkpoints. It is not a tuning of another engine. Fidelity verdict: `DIFFERENT_MODEL_VARIANT_SHORTS_ONLY` — long-context rows (32K, 80K) agree at 1.0000 against the sparse stack; all divergence is short-row behavioral (2 real code-path diffs) or comparator artifact. See [`docs/notes/dense-qsa-model-variant.md`](docs/notes/dense-qsa-model-variant.md).

## What it is

Everything needed to reproduce the measured stack on a clean Ubuntu 24.04 host with 4x B70:

- `scripts/host-setup.sh` — one-time host provisioning (driver stack, kernel, firmware, Docker limits)
- `docker/Dockerfile` — the serving image (vLLM fork + pinned toolchain; build fixes baked in)
- `.env.example` — the verified launch line, every knob annotated
- `start.sh` / `stop.sh` — preflight → weights identity gate → pre-boot XPU gate → launch → ready-poll; mandatory wedge watchdog
- `scripts/` — measurement harnesses (`soakfix.py`, `single-stream.py`, `needle-probe.py`, `toolcall.py`) and the wedge watchdog
- `tests/verify.sh` — the acceptance gate that produced the numbers below
- `docs/notes/` — the measured-results record and the two design verdicts (dense-variant fidelity, Level-Zero wedge mechanism)

## Prerequisites

`scripts/host-setup.sh` installs packages from two sources that are not enabled on a stock Ubuntu 24.04 install. Enable both first:

- **Intel's GPU software repository** (provides `intel-omix` 0.4 and the Level Zero userspace): follow Intel's client GPU installation guide at <https://dgpu-docs.intel.com>.
- **The package source for kernel `6.17.0-1010-intel`** (Intel's Ubuntu kernel packages). The script stops with a clear error if it can't find the package.

Also required: Docker Engine, with your user in the `docker` and `render` groups.

## Hardware

| Component | Requirement |
|---|---|
| GPU | 4x Intel Arc Pro B70 32GB (Xe2 / Battlemage), oneAPI-capable |
| Host RAM | ≥100 GiB free |
| Swap | ≥64 GiB on |
| Disk | ~200 GB for weights + PLE table |
| OS | Ubuntu 24.04, kernel 6.17.0-1010-intel, Intel OMIX 0.4 userspace (`scripts/host-setup.sh` provisions and verifies all of it) |

## Results

Every row names its harness, aggregate formula, and prompt shape. **Rows are not comparable across harnesses.** All numbers were measured on the image this Dockerfile builds, weights rev `40b8f18d`, engine port 8022 (`qwen-256k`).

| Metric | Value | Harness / formula / prompt |
|---|---|---|
| **32x600 sustained (GATE)** | **1,038.3 tok/s** (sustained_agg = MEAN r2..rN; gate ≥ 900 passed) | soakfix.py · agg = Σcompletion_tokens÷round_wall per round; r1 warmup discarded · open-ended essay prompt, runs TO the 600 cap |
| MNS ladder (reference) | n=2 95.3 · n=4 179.9 · n=8 335.2 · n=16 629.1 tok/s | same harness, same formula |
| Single-stream (N=20, first discarded) | **52.5 median (52.3–52.8)** — re-verified three independent ways: fresh cross-network client 52.55 mean / 52.56 median, and the engine's own `Engine 000` lines 52.4–53.3 during a sustained 1200-tok stream | single-stream.py · ctok÷wall, median of 19 · essay request, 600 tok |
| Single-stream vs context (decode-only) | flat: 53.6 @ 31 tok · 52.7 @ 8.7K · 52.4 @ 17.4K · 51.5 @ 34.7K (4% spread; TTFT excluded) | streaming sweep, temp 0, idle engine |
| Tool calls | **20/20 structural EQUIV** + multi-tool CORRECT_PICK + nested-args PASS | toolcall.py · compare = function name + argument JSON as OBJECTS (never raw text), temp 0 |
| 97K needle | **PASS @ 98,211 engine-confirmed tokens**, CORRECT, ×3 salted (TTFT 35.3–35.4 s) | needle-probe.py (engine-calibrated via usage.prompt_tokens), salted, temp 0 |
| 250K needle | **PASS @ 250,700 tokens**, TTFT 147.3 s (MML 262144 genuinely holds) | needle-probe.py, temp 0 |
| MTP speculative decode (k=1) | **CLOSED — measured negative**: 49.5 tok/s (−5.7%), accept 47.4%, fidelity DIRTY, n=8 errors; ceiling ≈ 63 tok/s even at 90% accept; unusable multi-tenant. Patches kept in [`experimental/patches/`](experimental/patches/) | A/B vs serial, same image, temp 0 — [`docs/notes/measured-results.md`](docs/notes/measured-results.md) |
| Pre-boot XPU gate | TRITON_XPU_GATE=PASS — triton vector-add compiles + exact result | docker/gate.py, mandatory before every model boot |
| Watchdog restart-path | test delivered: tests/watchdog-restart-test.sh | wedge watchdog (8022/`qwen-256k`), py-spy capture first |

## Quick start

```bash
git clone https://github.com/imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s
cd Qwen3.8-Flash-Next-4x-Intel-B70s

# 1. Host platform (CHANGES THE HOST — see warning below; REBOOT required)
sudo ./scripts/host-setup.sh

# 2. Weights (~180 GB: shards + PLE table) into a plain directory
#    (--local-dir gives real files; a Hugging Face cache snapshot is symlinks,
#     which break inside the container mount)
python3 -m pip install -U huggingface_hub
hf download devan-carlin/Qwen3.8-Flash-Next-W4A16 \
    --revision 40b8f18df4d4a32cb6e687a51c78207e5e438522 \
    --local-dir /data/Qwen3.8-Flash-Next-W4A16

# 3. Config + image
cp .env.example .env                 # set MODEL_PATH and PLE_TABLE_PATH to the directory above
./check-weights.sh                   # presence + family + size of the downloaded tree
docker build -t qwen38-flash-next:local docker/

# 4. Launch (preflight → weights gate → XPU gate → engine → ready-poll → watchdog)
./start.sh

# 5. Verify the measured numbers reproduce on your host
./tests/verify.sh
```

`./start.sh` also supports `start|stop|restart|status|logs`; `--launch` is the watchdog's restart path (skips the XPU gate). `./start.sh stop` = watchdog first, then the container.

> **`scripts/host-setup.sh` CHANGES THE HOST** (kernel, firmware, grub, Docker limits) and **reboots**. Read it before running. Only run it on a dedicated box.

## Weights

| What | Where |
|---|---|
| Checkpoint (17 shards, ~77 GB) + `ple_table_qwen4exp.pt` (~102 GB) | [`devan-carlin/Qwen3.8-Flash-Next-W4A16`](https://huggingface.co/devan-carlin/Qwen3.8-Flash-Next-W4A16) @ rev **`40b8f18df4d4a32cb6e687a51c78207e5e438522`** — one `hf download --local-dir` fetches both; command in Quick start |
| Identity gate | `check-weights.sh` — checks presence, model family and size of the downloaded tree (the pinned revision is in your download command) |
| Model license | **Qwen Community License 1.0** (see License below) |

The `ple_table_qwen4exp.pt` PLE table is part of the same pinned revision — no separate source or generation step. It must stay inside `MODEL_PATH`; the container reads it from there.

## Configuration

All knobs live in `.env` (`cp .env.example .env`), each annotated there. The ones that bite:

- `MAX_NUM_SEQS=32` — the verified operating point (1,038.3 tok/s at 32x600); the ladder above is the reference curve.
- `OVERRIDE_GENERATION_CONFIG` — sampler pin (temp 0.7 / top_p 0.80 / top_k 20 / presence_penalty 1.5) ships **intentionally**; measurement harnesses neutralize samplers per-request. Don't change the pin when comparing against the table above.
- `WEDGE_WATCHDOG_*` — mandatory watchdog (`scripts/wedge-watchdog.sh`); disable requires the double opt-out (`WEDGE_WATCHDOG_DISABLE=1` **and** `--no-preflight`).
- `XPU_GATE_DISABLE=0` — the pre-boot triton gate; same double-opt-out discipline.
- Optional MTP speculation: `SPECULATIVE_CONFIG` (default OFF — see the Results row for why).

## Known limits

- **2–6 h Level-Zero wedge under sustained load** — Xe2 driver-level wedge (`ccs`/`bcs` engine reset signatures); only a container restart recovers; in-flight requests are lost. The watchdog detects and restarts automatically. One precisely localized instance (a graph-capture hang with speculative decoding on) is documented in [`docs/notes/gdn-l0-wedge.md`](docs/notes/gdn-l0-wedge.md); the load-time wedge itself is recovered by the watchdog, not root-caused. Mitigation baked into `scripts/host-setup.sh`: xe GuC job timeout raised to 10000 ms (driver cap; default 5000).
- **Dense-attention model variant** — this is the dense-full-context QSA checkpoint, not the sparse-QSA variant; short-row behavioral diffs are real and characterized (2 code-path diffs); long-context fidelity agrees. See the note above.
- **Above 262144 context is untested** — `start.sh` hard-fails; the 250,700-token needle passes at the ceiling.
- **No authentication** — the API listens on all interfaces at `PORT` with no key. Bind it to localhost, put it behind an authenticating proxy, or add vLLM's `--api-key`.
- **`MAX_NUM_SEQS` > 32 hard-fails** (KV-cache math knee at MML 262144); 17–31 warn as untested.

## Troubleshooting

- **Preflight FAIL (XPUs/RAM/swap/kernel/GuC/iommu)** → run `sudo ./scripts/host-setup.sh`, reboot, re-run. `start.sh` prints exactly which floor failed.
- **XPU GATE FAIL** → triton/JIT gap in the image; fix via `docker/Dockerfile`.
- **Image not present locally** → build it: `docker build -t qwen38-flash-next:local docker/` (the `IMAGE` pin in `.env` is the exact measured build; a fresh host builds its own and points `IMAGE` at the local tag).
- **READY GATE FAIL** → `docker logs qwen38-flash-next`; graphs capture inside torch.compile (~135 s) before `Application startup complete`.
- **Idle-looking "gen=0.1 prefill=5.3" log lines** → engine log lines are 10-second windowed averages; the watchdog's 1-token liveness probes appear as near-zero ticks. Sustained decode reads ~52–53 on any tick a generation actually fills.
- **Watchdog** → state + captures in `.run/`; design notes in [`scripts/watchdog.md`](scripts/watchdog.md).

## Credits

- **devan-carlin** — the vLLM fork [`xpu-qwen4exp`](https://github.com/devan-carlin/vllm) and the W4A16 weights ([HF repo](https://huggingface.co/devan-carlin/Qwen3.8-Flash-Next-W4A16))
- **Intel** — the [`omix`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) base image and the XPU software stack
- **Qwen** — the [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) model, under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)

## License

This repo's kit (scripts, Dockerfile, docs): MIT — see [LICENSE](LICENSE).
The model weights are governed by the **[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)**; read it before use.

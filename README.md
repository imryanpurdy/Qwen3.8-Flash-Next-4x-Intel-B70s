# Qwen3.8-Flash-Next on 4x Intel Arc Pro B70 (side-lane stack)

Serving kit for **Qwen3.8-Flash-Next W4A16** on **4x Intel Arc Pro B70 32GB, TP4+EP** — the side-lane stack, deploy of record 2026-09-23: vLLM fork `devan-carlin/vllm@xpu-qwen4exp` (a69fba21) built on `intel/omix:0.4.0-devel-ubuntu24.04`, **262144 context**, MNS 16, decode graphs, kv fp8, qwen3/qwen3_xml parsers.

**Model variant, stated plainly:** this serves the **dense-full-context QSA variant** (indexer weights dropped) — a **different model variant** from the prior sparse-QSA production stack. It is not a tuning of the old engine. Fidelity evidence: 18-row harness comparison — long-context rows (32K, 80K) agree at 1.0000, all divergence is short-row behavioral (2 real code-path diffs) or comparator artifact; verdict `DIFFERENT_MODEL_VARIANT_SHORTS_ONLY` (`docs/evidence/2026-09-23-qsa-dense-verdict.md`, `docs/evidence/2026-09-23-sidelane-quicklook-mns16.md`).

**Repo = the deliverable.** Clone → `cp .env.example .env` → `./start.sh` → serves. Every number below names its harness, formula, and prompt shape; docs cite primary artifacts in `docs/`.

**Rollback:** the previous production stack stays deployable in [`legacy/`](legacy/README.md) — one line: `cd legacy && cp .env.example .env && ./start.sh` (~8–9 min to READY).

## Hardware

| Component | Requirement |
|---|---|
| GPU | 4x Intel Arc Pro B70 32GB (Xe2 / Battlemage) |
| Host RAM | ≥100 GiB available (128 GB installed) |
| Swap | ≥64 GiB ON |
| Kernel | 6.17.0-1010-intel (platform of record; `scripts/host-setup.sh`) |
| GuC firmware | 70.65 — sha256 `70d74627e395…67bb` (linux-firmware fb0889c0) |
| IOMMU | off (grub `iommu=off`) |
| Userspace | intel-omix 0.4 (the image builds FROM it; `scripts/host-setup.sh` provisions the host) |
| Docker | nofile ulimit ≥ 1M (LimitNOFILE=infinity) |
| Weights | `/data/hf-devan/Qwen3.8-Flash-Next-W4A16` (+ `ple_table_qwen4exp.pt`), rev of record `40b8f18d` |

## Measured numbers

Every row names its harness, aggregate formula, and prompt shape. **Rows are not comparable across harnesses.** Promotion gates of record 2026-09-23, image `es-lane@sha256:15a806fc7367…8417a`, MNS 16:

| Metric | Value | Harness / formula / prompt |
|---|---|---|
| **16×600 sustained (GATE)** | **median r2–r15 = 634.1 tok/s · mean r2–r15 = 596.0 tok/s** (promotion soak, n=14 — both stats include the mid-soak injection round, agg 109.7, whose 87.5 s wall is why mean < median; ex-injection rounds span 592.2–645.3) · MNS-16 battery of record 626.3 = harness sustained_agg (MEAN r2–r4, spread 610.9–632.6) | sidelane-soakfix.py · agg = Σcompletion_tokens÷round_wall per round; r1 warmup discarded · open-ended essay prompt, runs TO the 600 cap |
| 8×600 sustained | 338.3 tok/s (spread 335.5–339.3) | sidelane-soakfix.py · same formula · same prompt |
| Single-stream (N=20, first discarded) | 52.5 median (52.3–52.8) | sidelane-single-stream.py · ctok÷wall, median of 19 · essay request, 600 tok |
| Tool calls | **20/20 structural EQUIV** + multi-tool CORRECT_PICK + nested-args PASS | sidelane-toolcall.py · compare = function name + argument JSON as OBJECTS (never raw text), temp 0 |
| Quality battery (promotion gate) | **8/8 prompts PASS, 2/2 runs each, 0 loops** — arithmetic (7917, 371293), code (SIDELANE-CODE-A-77, FROG-211-LILY), tool×2, reasoning (210, 36). Per-prompt vs production reference at temp 0: 5/8 exact agree (math1, math2, code2, reason2, tool2-structural); the 3 diffs trace to KNOWN reference defects or characterized variant behavior, not soak/load: tool1 prod ref malformed args 1/2 (`locationlocation`, ledgered), reason1 prod ref run2 corrupted (`22<|im_start|>`, the known temp-0 cache split), code1 different-but-valid code realizing the same token (the known short-row code-path diff) | sidelane-quality.py · 8 fixed prompts × 2 runs, expected-literal check + loop detector · samplers explicitly neutralized, temp 0 |
| 97K needle | **PASS @ 98,211 engine-confirmed tokens**, CORRECT, ×3 salted (TTFT 35.3–35.4 s; 76.2/74.5 s under full soak load) | sidelane-needle-probe.py v2.1 (engine-calibrated via usage.prompt_tokens), salted, temp 0 |
| 250K needle | **PASS @ 250,700 tokens**, TTFT 147.3 s (MML 262144 genuinely holds) | sidelane-needle-probe.py, temp 0 |
| Context ceiling | 262144 (their line) — 250,700-ptok needle CORRECT | longgates 2026-09-23 |
| Pre-boot XPU gate | TRITON_XPU_GATE=PASS — triton vector-add compiles + exact result | docker/sidelane/gate.py, mandatory before every model boot |
| Watchdog restart-path | test delivered: tests/watchdog-restart-test-eslane.sh | es-lane watchdog (8022/qwen-256k), py-spy capture first |

## Quick start

```bash
git clone https://github.com/imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s
cd Qwen3.8-Flash-Next-4x-Intel-B70s
scripts/host-setup.sh          # once per host; sudo; REBOOT after (kernel+iommu+GuC)
cp .env.example .env           # the side-lane line; every knob commented
docker build -t es-lane:qwen4exp-a69fba21 docker/sidelane/   # image of record — LOCAL BUILD (GHCR push OPEN, see below)
./start.sh                     # preflight → weights → XPU gate → image check → launch → READY gate
./tests/verify.sh              # gates + the numbers table (structural tool calls first)
./tests/watchdog-restart-test-eslane.sh     # proves the watchdog restart path (restarts the engine by design)
```

`./start.sh` also supports `start|stop|restart|status|logs`; `--launch` is the watchdog's restart path (skips the XPU gate). `./start.sh stop` = watchdog first, then the container.

> **Image of record is LOCAL-BUILD-ONLY (OPEN ITEM).** There is no registry image yet: a fresh host builds it from `docker/sidelane/Dockerfile` (all five fixes baked — PEP 668, pyjwt, setuptools-rust, triton-xpu repair, python3-dev — and the pre-boot XPU gate runs from it before any model boot). The `IMAGE` digest pin in `.env` names the build of record (`15a806fc7367…8417a`); on a fresh host the digest will not exist locally and `start.sh` falls back to the `es-lane:qwen4exp-a69fba21` tag with a loud warning. **GHCR push + authenticated fresh-host pull test: OPEN** — when the push lands, replace the build step with `docker login ghcr.io` + the digest pull.

## Knobs

All in `.env.example`, each commented there. The ones that bite:

- `MAX_MODEL_LEN=262144` — their MML; 250,700-ptok needle CORRECT. start.sh refuses above it.
- `MAX_NUM_SEQS=16` — the gated operating point (×3.45 vs MNS 4); above 16 untested, refused.
- `OVERRIDE_GENERATION_CONFIG` — their sampler pin (temp 0.7 / top_p 0.80 / top_k 20 / presence 1.5). Measurement harnesses neutralize per-request; don't change the pin.
- `IMAGE` — pinned digest; local build of record until GHCR push.
- `XPU_GATE_DISABLE` — mandatory pre-boot triton gate; double opt-out to skip.
- `WEDGE_WATCHDOG_*` — mandatory watchdog (`scripts/wedge-watchdog-eslane.sh`); disable requires the double opt-out (`WEDGE_WATCHDOG_DISABLE=1` **and** `--no-preflight`).
- `PREFLIGHT_*` floors — 4 XPUs, RAM, swap, **kernel**, **GuC hash**, **iommu=off**, disk.

## Known limits

- **Different model variant** — dense-full-context QSA; short-row behavioral fidelity diffs vs the sparse stack are real and characterized (2 code-path diffs). Long-context fidelity agrees.
- **Sampler pin is theirs** — presence_penalty 1.5 ships in the launch line by design.
- **Xe2 Level-Zero wedge every 2–6 h under load** — watchdog captures py-spy + restarts ≤3×; in-flight requests lost on restart. A restart after DEVICE_LOST can stall on worker attach → full host reboot is the recovery (operator call, not automated).
- **/data disk pressure** — 92% used / 18 G free (2026-09-23, report-only standing state).
- **Legacy stack defects don't carry over but are ledgered** — production's malformed tool-call args (`locationlocation`, doubled key, valid JSON) were a defect of the OLD stack's reference capture; the side lane emits canonical calls (see ledger KNOWN-ISSUE-PROD-TOOLCALLS).

## Troubleshooting

- **Preflight: GuC hash mismatch / kernel not recognized** → `scripts/host-setup.sh`, then reboot.
- **XPU GATE FAIL** → triton/JIT gap in the image; fix before any model boot (this gate exists to catch it in seconds). Rebuild via `docker/sidelane/Dockerfile` (fixes 1–5 baked: PEP 668, pyjwt, setuptools-rust, triton-xpu repair, python3-dev).
- **Image not present locally** → build it: `docker build -t es-lane:qwen4exp-a69fba21 docker/sidelane/` (digest pin in `.env` must match).
- **READY GATE FAIL** → `docker logs es-lane`; graphs capture inside torch.compile (~135 s) before `Application startup complete`.
- **Wrong/failed tool calls** → parsers must stay: `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`.
- **16-way number far below expectations** → MNS 4 vs 16? The engine ran MNS 4 for weeks before the 2026-09-23 correction; verify the boot receipt prints `'max_num_seqs': 16`.
- **Rollback** → `legacy/README.md` (one line).

Campaign and incident history: `docs/` (one doc per event, from primary artifacts). License: [LICENSE](LICENSE).

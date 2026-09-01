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

Mia-style clone-and-run: `start.sh · stop.sh · check-weights.sh · wedge-watchdog.sh · .env.sample · files/`. Kit scripts land here as they're built (in flight, committed on arrival). Key decisions:

- **Docker-first** (decided 2026-09-01): image = vLLM XPU runtime base + pinned toolchain + overlay series 0002–0018 + QSA kernels; `/dev/dri` passthrough; host HF cache bind-mounted; weights persist outside the container.
- **Wedge watchdog mandatory**: the Xe2 Level-Zero wedge kills the job every 2–6 h under load — start.sh refuses to launch without it (double opt-out to disable).
- **Preflight gate**: 4×XPU visible, host swap ≥64 GiB, ~200 GiB disk, ≥40 GiB root free — fail with actionable errors.
- **Receipts**: every launch writes `.run/manifest.json`; only measured anchors are ever quoted, spec targets stay marked as targets.

## Evidence (`docs/evidence/`)

Lab lane maps used to build the specs: lab tree map, DSv4 80-tok/s reference lane, B70 cookbook digest (SergiioB), Flash-Next scaffold, Qwen2.7B reference lane, and `qsa_ops.py` — the merged Triton QSA kernel extract (vLLM PR #53896). The Lane-4 evidence pack (`docs/lanes/evidence-lane4-ttft.md`) carries file+line citations for every number above.

**Valid until it's on the rig:** every number here is from the lab campaign's immutable receipts. First thing the rig does is re-verify the anchors — cache-zero, cold prompts, median-of-medians.

# Deploy-kit contract — Qwen3.8-Flash-Next 4×B70 recipe repo

DECISIONS DOCUMENT (EdgeQuant, 2026-09-01). The kit author subagent implements
this verbatim; deviations require a new decision. Layout mirrors Mia's kit
(/tmp/mia-qwen38) so "clone → cp .env.sample .env → edit → ./start.sh" works,
adapted to a single-node 4× Intel Arc Pro B70 host.

## Repo layout (target: github.com/imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s)

```
README.md                  Quick Start (3 steps), Prerequisites, Flags, .env table
start.sh                   THE entrypoint — all mechanics live here
stop.sh                    kill server + watchdog, both graceful
check-weights.sh           checkpoint presence + identity-hash verification
wedge-watchdog.sh          MANDATORY Level-Zero wedge watchdog (see D3)
.env.sample                every knob, annotated with measured values
files/                     overlay/patch artifacts the kit applies
docs/                      lane1..4 specs + evidence (pushed from /tmp/flashnext-lanes)
LICENSE                    (exists)
```

## D1 — Bare metal, not Docker (DECIDED, inverse of Mia)
Mia pins `vllm/vllm-openai:qwen38-flash-next` in docker. We do NOT ship docker
as the default path. Reasons: (a) the lab's receipts (TTFT anchors, A-series
results) are all bare-metal with host RAM floors + 64 GiB swap + PLE pinned-UVA
51 GiB — a containerized variant would be an untested second path on a rig
nobody has run; (b) torch 2.11/2.13 conflict + overlay patch series (0002–0018)
are venv-level. start.sh creates/reuses a venv, applies the overlay, launches
directly. Dockerfile is a stretch goal ONLY after the first live rig run
succeeds; do not write it now.

## D2 — start.sh phase structure (mirrors Mia, XPU-adapted)
1. Load `.env`, validate required vars (same loop pattern as Mia), numeric
   sanity on token/len vars.
2. **Preflight (new vs Mia — non-skippable):**
   - exactly 4 XPU devices visible (`sycl-ls` | count level_zero:gpu)
   - host RAM ≥ 100 GiB free-ish floor check; swap ≥ 64 GiB ON (swapon --show)
   - disk ≥ 200 GiB for checkpoint tree (185.56 GB)
   - root filesystem ≥ 40 GiB free (graphs-era floor)
   - fail with actionable err() message per check. `--no-preflight` exists but
     prints a loud WARN and records `PREFLIGHT_SKIPPED=1` in the run log.
3. Weights: HF download via huggingface-cli with auto-skip if cached (Mia's
   HEAD_HAS pattern, single node — no rsync/ssh_worker). MODEL_ID default
   `Qwen/Qwen3.8-Flash-Next-FP8`.
4. Environment: venv create/reuse, install pinned torch + vLLM overlay per
   files/ (toolchain pins come from lane specs; freeze exact versions in
   .env.sample as comments once receipts exist).
5. Watchdog: spawn wedge-watchdog.sh as a child (same process group), PID
   recorded to `.run/watchdog.pid`.
6. Launch vLLM with the full flag set derived from .env (see D4), tee to
   `.run/server.log`.
7. Readiness: poll OpenAI `/v1/models` up to 15 min (long model load), then
   print the exact verification greps Mia-style (KV pool line, PLE placement
   line, graph status line) with the EXPECTED values from lab receipts.
Flags: `--no-download --no-launch --launch --no-preflight` (Mia semantics +
preflight opt-out).

## D3 — Wedge watchdog (mandatory; Mia has no analog)
The Xe2 Level-Zero wedge kills the job every 2–6 h under load. Non-negotiable:
- start.sh REFUSES to launch the server without the watchdog unless
  `WEDGE_WATCHDOG_DISABLE=1` is set AND `--no-preflight` was passed — both
  explicit acknowledgements; if disabled, print a red banner stating the rig
  WILL wedge unattended within 2–6 h.
- watchdog loop: monitor server liveness + device health (xpu-smi / sycl-ls
  probe cadence 60 s), on detection: capture last 200 log lines + timestamp,
  kill hung process group, restart via start.sh --launch (bounded retries,
  default 3, then give up loudly).
- State under `.run/` only; never write outside repo dir.

## D4 — .env.sample knobs (annotated with MEASURED values, Mia style)
MODEL_ID, SERVED_MODEL_NAME, PORT, TP=4 (comment: TP∈{2,4,8}; 2 KV heads; TP6
impossible), ENABLE_EXPERT_PARALLEL=true (EP4), MTP_NUM_SPECULATIVE_TOKENS=3
(comment: measured 4K 15.502 tok/s @ MTP3; 20.7 @ MTP4 512ctx; see docs/lane2),
MAX_NUM_BATCHED_TOKENS (default 64 = lab baseline; comment: frozen in every lab
launcher; Lane 4 sweep 512/2048/4096 pending — 8192 crashed GLM-5.3 indexer
class), MAX_MODEL_LEN, PLE_CPU_OFFLOAD_GB=12.25 (comment: 12.22 GiB/rank × 4 ≈
51 GiB pinned host), SWAP_GB/PREFLIGHT floors, WEDGE_WATCHDOG_* (interval,
retries), EXTRA_VLLM_ARGS. GRAPH_MODE knob reserved (eager until Lane 1
qualifies; comment: graphs quarantined-negative a1–a7).

## D5 — README shape (mirrors Mia)
Hero header → one-line "what this is" (5.52 tok/s eager baseline → serving
target, hardware, first-public) → Prerequisites (B70 ×4, 128 GiB RAM, 64 GiB
swap, ~200 GiB disk, toolchain note) → Quick Start 3 steps → "Verify it came
up" section with expected log lines + measured anchor table (MTP0 5.515783,
MTP3-4K 15.502 / TTFT 187.9, MTP4-512 20.727) → Flags → .env table → Troubleshooting
(wedge watchdog behavior, OOM-during-compile pointer to docs/lane1, TTFT pointer
to docs/lane4) → link to docs/ for lane specs. Numbers: ONLY measured anchors;
spec targets marked as targets, never implied as achieved.

## D6 — Identity + receipts
- check-weights.sh verifies the HF snapshot and prints the identity hash
  bcd9f01ddc9cff2316eb84281bebcd5b058bddce for operator comparison; kit docs
  freeze it. Mismatch = hard fail (wrong-weights guard).
- Every launch writes `.run/manifest.json` (env hash, git describe of kit,
  .env content minus secrets, start epoch) — receipts discipline from the lab.
- No secrets in any committed file; HF_TOKEN stays in .env (gitignored —
  ship a .gitignore with .env, .run/).

## Authoring split (delegation bar, Ryan 2026-09-01)
Contract = decisions (this doc). Implementation = DSv4-flash subagent AFTER
wave 2 (Lanes 3–4) frees the 2-slot cap: one agent authors start.sh,
stop.sh, wedge-watchdog.sh, check-weights.sh, .env.sample, .gitignore; bash
`bash -n` + shellcheck-level self-check required; THEN a second pass assembles
README.md from lane specs + this contract. Final assembly + push to the repo =
EdgeQuant validation turn.

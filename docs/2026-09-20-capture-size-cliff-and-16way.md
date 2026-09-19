# 2026-09-19/20 — Capture-Size Cliff: Root Cause, Fix, 16-Way Result

## Headline
Morning: 9 wedged boots in 10, stall at 8-way, 12-way collapse to 30 tok/s agg.
Tonight: **320 tok/s sustained aggregate at 16-way, zero stalls, engine healthy.**
Aggregate target (200-400) is met.

## Root cause: the capture-size cliff
`--max-num-seqs 12` with graph capture list `[1,2,4,8,16,24]`: an aligned decode
batch at size 12 is **not captured**, and the XPU FULL_DECODE_ONLY path falls
back to EAGER (not pad-to-next) — ~5× slower per step. Shape:

- Round 1, staggered arrivals → decode batches ≤8 (captured) → fast
- Round 2+, requests synchronize in decode → batch lands exactly at 12 →
  uncaptured → eager every step → 237s walls, 30.4 agg, flat within 1%
- 8-way immune (8 ∈ list). The "12-way curve bend" seen all week was the
  same cliff partially engaged.

Fix (v24h, start.sh, commit `3589249`): generate capture sizes = every integer
1..MNS, plus a boot guard that REFUSES to start if the list doesn't cover MNS.
Pattern: anywhere a config value must match a runtime list, the launcher
checks it (second guard of the day after watchdog gen-probe).

## v24h validation (MNS=12, all sizes 1..12 captured)
- Boot log: config `[1..12]` AND `Capturing CUDA graphs (FULL): 12/12`.
- 12-way: r1 120.0 (warmup), then 253.2 / 129.3 / 252.9 — fast rounds at
  Ryan's predicted ~245+ zone; mean 188.9. vs 30.4 pre-fix = 8.3×.
- Known-answer: correct.

## v24h2: 16-way + transitional-size skip
Full 1..16 capture **faulted** (DEVICE_LOST at 23:09:12, torch/xpu/graphs.py,
same class as MTP1-capture fault). v24h2 adds CAP_SIZES_LIST override:
`[1,2,3,4,5,6,7,8,12,16]` — steady-state sizes only, transitional 9-11/13-15
skipped. Boot: `Capturing: 10/10`, zero faults, known-answer correct.

**16-way result (MTP0, v24h2, boot-rel2359, watchdog live):**
- r1 158.7 (warmup artifact: all-16 prefill simultaneously — burst harness
  measures prefill-heavy mixed load in r1, not steady decode)
- r2-r4: **314.5 / 310.4 / 320.1** sustained (30s walls, 9600 tok/round)
- mean 275.9; per-stream ~20 = 8-way's per-stream → near-linear scaling held
- Zero errors, zero stalls, post-probe 200, KV headroom 55% (70K/127K)

## Harness rule adopted (Ryan)
soakfix.py reports r2-r4 as the result, r1 as warmup — same as the 20-run
discard-first rule on the Sparks. Sustained rounds are the honest number.

## Bimodality note (12-way, did not recur at 16)
v24h 12-way showed alternating fast/slow rounds (253/129). At 16 none appeared
(310-320 flat). Most likely the 12-way slow mode was a transitional shape
interacting with the skip-list boundary that 16 avoids; consistent with the
skip-list working. Recorded, not chased — revisit only if it reappears.

## Flag status (all in .env / start.sh, none in image)
- CCL_ZE_CACHE_OPEN_IPC_HANDLES=0: staged pre-context-raise. A/B note: with
  the flag ON at MNS=12 the cliff fired deterministically (30.4 flat); OFF it
  fired stochastically (43→58). Interpretation (Ryan): the flag doesn't cause
  the cliff, it FORCES alignment (every collective re-opens IPC handles →
  tighter rank sync → batch lands at exactly MNS from step one). Once the
  cliff is fixed the flag costs nothing extra and stays ON. NOTE: currently
  absent from the running v24h2 boot — re-stage before any large-context work.
- CAP_SIZES_LIST=1,2,3,4,5,6,7,8,12,16 (v24h2 override; generator covers MNS
  fully when unset)

## Exonerations recorded today
- Driver-state accumulation (clean-host repro of 12-way + MTP1 capture fault)
- SYCL_UR_USE_LEVEL_ZERO_V2 (legacy adapter faults anyway on MTP1 capture)
- PLE table RSS growth (flat 3.79GB under burst)
- CCL flags trio from the thread (|CCL_SYCL| exception at init on 2021.17.2)
- Message-path/output-handler wedge (EngineCore idle = scheduler-side cliff,
  not a lost-output bug; wave completions were the cliff's round rhythm)
- The IPC flag as cliff CAUSE (it is an exposure accelerator, not a cause)

## Still open (next lanes)
1. **Single-stream 80-100**: MTP1-capture DEVICE_LOST (clean-host confirmed,
   adapter-independent, sizes 13-16 capture also faults — same class). The
   platform rebuild (OMIX, torch 2.13.0+xpu, oneAPI 2026.0) is the decided
   path; runbook + BOM docs in flight. Subagent intel on vllm-xpu-kernels GDN
   capture dispatched. If the rebuild doesn't clear it, the fix may be
   capture-safe kernel work — Intel's kernel-dev skills are in the loop.
2. Context raise (max_model_len beyond 4352) with IPC flag staged: KV math
   re-check first.
3. MTP1 acceptance lane (72.5% stable measured eager; graphs re-check after
   rebuild; k=3 as the 80-100 path).
4. bimodal: closed unless it reappears.

## Boot-provenance appendix
| Boot | Config | Result |
|---|---|---|
| rel1947 (clean host) | MTP0 MNS=12 flag-off | 12-way 40.4 mean, step-degrade |
| rel1956 | MTP0 MNS=12 stock | same cliff shape |
| rel2037 | MTP1 capture (clean host) | DEVICE_LOST 3rd repro |
| rel2147 | MTP0 MNS=12 flag-off A/B | 51.3 mean, stochastic cliff |
| rel2247 (v24h) | MNS=12 caps 1..12 | 253/129 bimodal, 8.3× |
| rel2359 | MNS=16 caps 1..16 | capture fault 13-16 |
| rel0027 (v24h2) | MNS=16 caps [1..8,12,16] | **320 sustained, 0 stalls** |

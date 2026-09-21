# 2026-09-19 — Connector fix v3: runner-thread stream-ordered host staging (stage-v24f) — shipped + smoke evidence

## 1. Mechanism recap — one op, two defects

The connector thread's tensor `.copy_()` at `connector.py:315/344` was a **USM memcpy
through Level Zero** (source = UVA view). That op:

1. **Contended with graph-replay command-buffer appends** inside Level Zero's command-list
   manager. Signature (3 native captures): TP0 sampler block + TP1-3 replay spin — the
   unified wedge/stall site documented in `2026-09-19-stall-mechanism-ple-l0-contention.md`.
2. **Raced device-side writers in the MRV2 era** (InputBuffers written by Triton kernels) —
   connector-thread reads had no ordering vs those writes → torn/stale n-gram reads.

**Two defects, one op.** Any fix that kept the connector thread touching the input buffers
through torch (v24d's `from_numpy` view trick, v24e's `np.copyto` on shared ranges) only
changed the spelling; v24e still raced the MRV2 device-side writers, so the torn-read defect
survived even with the L0 contention gone.

## 2. v3 design — runner-thread stream-ordered D2H + flag-gated consume

- **Producer moved to the runner thread**: stream-ordered D2H into **2-slot pinned mirrors**:
  - `staging_input_ids [2, max_num_batched_tokens]` int32
  - `staging_query_start_loc [2, max_num_reqs+1]`
  - `staging_ngram_context [2, *source.shape]`
- **Flag write LAST**, carrying a per-boot monotonic `seq`, so D2H publish is
  complete-before-flag.
- **Connector consumes via plain numpy load only**, gated on `flag_np[slot]==seq` —
  bounded 2s poll, legacy `copy_` fallback on timeout (error-logged).
- **(slot, seq) token rides inside the queue payload** — atomic handoff; this also fixes a
  shared-mutable race caught in review (updating the mirror set from the connector side).
- **`put_nowait` → bounded `queue.put`**: a full queue previously crashed the engine with
  `queue.Full`; the bounded put makes backpressure explicit and non-fatal.
- **Miswire canary** logged at every boot:
  `PLE v3 staging wired: staged=… tp_rank=…` — one line per rank; a missing/aberrant line
  is a miswire spelled out in the boot log, not a mystery.

## 3. Load-bearing assumption, verified empirically

The flag+gating protocol assumes queue visibility is **in-order**: if a newer payload were
visible before the flag write that guards it, the consumer could read a mirror before the
producer finished. Verified with the in-order visibility test:

- **300/300 trials clean**: two D2H copies back-to-back (the production pattern), plain
  numpy load from another thread, no flag-first, no torn reads, no blocked loads.
- **Idle-engine caveat (registered)**: the microtest runs with the engine idle; the burst
  campaign is the real proof of in-order visibility under live replay load.

## 4. Smoke gates — all green on boot 3915112 (stage-v24f)

| gate | result |
|---|---|
| ranks staged (canary) | **4/4 staged=True** |
| known-answer temp-0 | correct — "Paris" |
| burst 1 | 4 rounds 88.0/182.5/182.6/180.1 agg, **mean 158.3**, 0 errors |
| burst 2 | mean agg **170.4**, 0 errors |
| L0-absence proof | 170 py-spy dumps mid-burst, `appendUSMMemcpy` in **0/170** (was in 27/27 pre-fix dumps) |
| flag timeouts | **0** |
| engine alive post-burst | gen probe **200** (1-token generation health) |

The 0/170 is the direct falsification of the old op: the connector thread is present in 34
of the 170 dumps (idle at queue-get), and not one is executing the L0 memcpy that wedged the
rig in every captured stall.

## 5. Image provenance — stage-v24f = stage-v24e + v3 patches

- Built from stage-v24e by applying the v3 patches, then **committed with explicit
  `--change` ENTRYPOINT/CMD restore**. Footgun: an earlier `docker commit` from an
  `--entrypoint bash` build container captured **bash as the image entrypoint**, which broke
  boot — metadata must be restored on **every** commit, including this one.
- Patch applied **script-file-only** with an **md5 gate**
  (`a4280a54329bdadaca2cd8754a35c9a6`) on the target, after two silent heredoc failures
  earlier today (heredoc produced empty diff but exited 0 — the md5 gate converts that class
  of failure into a hard no-go).

## 6. Campaign — in progress

- **8/8 bursts clean so far** (pre-fix baseline: 5 stalls / 7 bursts on stage-v24d).
- 12–15 planned per `docs/campaigns/2026-09-19-v3-campaign-prereg.md` decision rule:
  **0–1 stalls / 15 = confirmed**; 2–3 = weak evidence (+10 bursts); ≥4 = fix ineffective →
  lock diagnostic / `VLLM_PLE_CPU_OFFLOAD=1` discriminator ladder.
- This doc's smoke is not the campaign proof: serve the count from the campaign log.

## 7. Watchdog v2.5 (also shipped today)

- **Engine-true gen-probe health** replaces the `/v1/models` health check (the models probe
  stays 200 while EngineCore is dead — the "stall #4" hunts all ran blind through it).
- Hardened by subagent with **runtime-verified scenarios**; fixes:
  - **F7a**: phase-detection bug
  - **F7b**: `container_start_epoch` argument-order bug
- Swapped in immediately before the campaign; its kill decisions are stricter than the
  pre-fix watchdog's and **count as stalls** (declared confound in the campaign prereg).

## 8. Evidence anchors / cross-references

- `2026-09-19-stall-mechanism-ple-l0-contention.md` — wedge path, 3 native captures, 27/27
  dumps with `appendUSMMemcpy` pre-fix.
- `2026-09-19-ple-cpoffload-gate-and-campaign.md` — `VLLM_PLE_CPU_OFFLOAD` gate + v24e
  staging fix + pre-registered protocol.
- `2026-09-19-v3-campaign-prereg.md` — campaign hypothesis/protocol/decision rule.
- `2026-09-19-mtp-acceptance-check-prereg.md` — post-campaign MTP1-acceptance recovery check
  (predicted: post-v3 delta > pre-v3 +13%, torn-read removal).

## Status

Fix shipped (stage-v24f), smoke green, campaign running 8/8. This doc recorded for the
post-campaign ledger; nothing in it changes the pre-registered decision rule, except the
failure surface claim is now op-level (connector-thread torch-op on USM storage), not
copy-spelling-level.

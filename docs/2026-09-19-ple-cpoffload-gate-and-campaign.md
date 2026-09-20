# 2026-09-19 — VLLM_PLE_CPU_OFFLOAD gate (stale-comment trap) + connector staging fix v2 (stage-v24e) + campaign pre-registration

## 1. The gate: VLLM_PLE_CPU_OFFLOAD=1 enables the XPU PLE offload lane — start.sh:458 comment is STALE

### The stale comment (start.sh:458)

```
DOCKER_RUN+=(-e "VLLM_PLE_CPU_OFFLOAD=0")   # NVIDIA-only PleOffloadLayer worker path;
                                            # NOT the XPU UVA path (scaffold §2B)
```

inside the block start.sh:456 labels *"Explicit XPU anti-envs — these must NEVER come on for this recipe"*.
Both the comment and the block header are wrong for the deployed build: this vLLM gates the XPU PLE
offload lane on the same env, and the line is also not authoritative (see override order below).

### Proof this env gates the XPU PLE offload subsystem (exact loci, vendor patch 0002 + 0003)

- `vllm/envs.py:300` — `VLLM_PLE_CPU_OFFLOAD: bool = False` (TYPE_CHECKING); `envs.py:2044-2053` —
  runtime parser, commented *"Run n-gram PLE lookup in a dedicated CPU offload worker. The initial
  implementation supports ModelRunner V1 and single-node TP only."* Plain bool read; **no platform
  gate anywhere** in the env definition. Sibling env at envs.py:301 `VLLM_PLE_OFFLOAD_READY_TIMEOUT`
  (default 600.0) ships with it.
- `vllm/config/parallel.py:491-492` — `if envs.VLLM_PLE_CPU_OFFLOAD and not self._ple_offload_ipc_path:
  self._ple_offload_ipc_path = get_open_zmq_ipc_path()`. ParallelConfig is platform-agnostic; XPU
  boots take the same branch.
- `vllm/v1/executor/multiproc_executor.py:673-674` — after `self.worker.init_device()`:
  `if envs.VLLM_PLE_CPU_OFFLOAD: self.worker.spawn_ple_offload()`; and `:683-684` after
  `load_model()`: `if envs.VLLM_PLE_CPU_OFFLOAD: self.worker.wait_ple_offload_ready()` (WorkerProc.run).
  Every worker process spawns the offload subprocess iff the env is truthy — on XPU too.
- `0003-Support-eager-PLE-offload-transport-on-XPU.patch` is the XPU port of exactly this subsystem:
  host-synchronized `CpuGpuSemaphore` (`is_host_synchronized` → `wait_done()`, no CUDA stream ops),
  `self._pin_input_buffers()` guarded by `self.device.type == "cuda"` (XPU keeps shared CPU staging
  unpinned), and an XPU-validator test that runs `_validate_ple_offload_config()` with
  `is_cuda=False, is_xpu=True`. Code is XPU-first, not NVIDIA-only.
- **Runtime proof on this rig**: the native dumps (stall-mechanism doc, boot 3174405) show TP0's
  `ple-offload-dp0` thread in `_request_loop (connector.py:296)` → `_process_request (connector.py:315)`
  → `appendUSMMemcpy`. That thread and the registration log line
  `"PleOffload: registered %d PleOffloadLayer(s) (dp_rank=%d, tp_rank=%d, ipc_addr=%s)"`
  (connector.py `_register_with_offload_worker`) exist only when the env is true — they were live
  during the wedge, so the serving line was running with `VLLM_PLE_CPU_OFFLOAD=1`.

### How the hard-coded `=0` is silently inverted

`start.sh:461-464` appends `EXTRA_DOCKER_ARGS` **after** the anti-env block:
`if [[ -n "$EXTRA_DOCKER_ARGS" ]]; then DOCKER_RUN+=($EXTRA_DOCKER_ARGS); fi`.
docker resolves repeated `-e KEY=val` by key with the **last** occurrence winning, so
`.env` → `EXTRA_DOCKER_ARGS="-e VLLM_PLE_CPU_OFFLOAD=1 …"` overrides `start.sh:458`. The "never come
on" line is decorative; the effective value is set by the rig's `.env`, which is gitignored and never
appears in any repo diff. Net: today's serving stack runs the PLE connector lane (5/7 stall baseline)
with a comment in the launcher claiming that lane cannot run on XPU at all.

### Two-mechanism name collision (why the comment looks plausible)

Two independent PLE host-placement mechanisms share the "PLE offload" name:
- UVA offload of PLE weights (`PLE_CPU_OFFLOAD_GB=12.25` → `cpu_offload_gb`, synchronous pinned UVA,
  12.22 GiB/rank pinned; .env.sample:53) — the "XPU UVA path" the comment refers to.
- the vLLM v1 PLE offload **connector/worker** lane (VLLM_PLE_CPU_OFFLOAD) — separate process +
  per-step staging copies; THIS is the one captured in the wedge stacks.

`start.sh:458`'s comment is true of neither: `VLLM_PLE_CPU_OFFLOAD` is not the UVA path and is
not NVIDIA-only.

### Same trap class as GRAPH_MODE=eager

`start.sh:123-127` still rejects any non-`eager` GRAPH_MODE: *"graphs are QUARANTINED-NEGATIVE
(attempts a1-a7, 2026-08-28; docs/lanes/lane1 §1). Only 'eager' is deployable."* Stale since the
2026-09-18 stage-v24c breakthrough (5-6x, 133 tok/s 8-way — docs/2026-09-18-graph-mode-breakthrough.md)
— and the very line that stalls replays captures: `replay (torch/xpu/graphs.py:108)`.
Same pattern: launcher comment freezes an EOD verdict that later evidence inverted, and every
operator reads the comment as current. Cross-referencing note: graphs went in via env/flags
(`VLLM_XPU_ENABLE_XPU_GRAPH=1` + `--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}`),
not via the GRAPH_MODE knob — so two configuration surfaces encode the same decision and only one
was updated.

### Launcher hygiene to land with the fix (design, not executed)

1. Rewrite the `start.sh:458` comment (and the `:456` block header) to state the env is the XPU
   PLE-offload gate and that the effective value follows `EXTRA_DOCKER_ARGS` (last `-e` wins).
2. Add a post-`docker run` check (or extend the verification greps) that reads the effective
   `VLLM_PLE_*` values via `docker inspect` and prints/logs them — overrides must be visible in the
   boot log, not silent.
3. Fix or wire the GRAPH_MODE guard to the current graph serving line (or delete the verdict from
   the launcher and point at docs/2026-09-18).

## 2. Connector fix v2 — stage-v24e (numpy-level host staging + A/B switch)

### The wedge path (recap, evidence in stall-mechanism doc)

Per step the connector's background thread stages tiny runner inputs into the fixed shared buffers
then publishes a ZMQ request: `_request_loop (connector.py:296)` → `_process_request
(connector.py:315)` → `_copy_cpu_inputs` (MRV1: `_input_ids_buf[...].copy_(...)` etc. — int32
input_ids ≤ max_num_batched_tokens, query_start_loc ≤ max_num_seqs+1, ngram_context) or
`_copy_cuda_inputs` (MRV2 D2H on the background stream). Those `copy_` ops touch buffering that
Level Zero treats as USM transfers (`appendUSMMemcpy` in the stack) and contend with the replay
command-buffer appends on the same L0 command-list manager. Note: in-image line numbers (296/315)
refer to the rig stage (v24d + later overlay patches); the vendored base (0002+0003) has these
functions ~38 lines earlier — the stage patches must be collected from the image before it is
replaced (none of them are in files/overlay today).

### Why v24d's from_numpy approach was wrong

v24d "staged" via `torch.from_numpy(x.numpy())`. `from_numpy` does **not** copy — it returns a
torch tensor **view** over the same storage. When that storage is USM-backed (as it is here — the
stack shows the L0 memcpy path), the resulting tensor is still a torch op on a USM pointer, so
torch may (and in the wedge evidently does) route the subsequent copy through the Level Zero
command path — the identical `appendUSMMemcpy` from the captured stack, plus an extra allocation
and numpy round-trip. The patch changed the copy's *spelling*, not its *mechanism*: the L0
contention was untouched.

### v24e spec

- **Target**: `_copy_cpu_inputs` only (the path whose frame appears in the wedge). The destination
  buffers (`_input_ids_buf`, `_query_start_loc_buf`, `_ngram_context_buf`) and the bound source
  tensors are fixed at connector init and validated by `_validate_input_sources` (dtype/ndim/shape/
  contiguity contract holds), so both sides are stable numpy-viewable host ranges.
- **Copy as numpy, never as torch**: take numpy views on both sides and copy with
  `np.copyto(dst_np, src_np, casting="no")` (dtypes already matched/validated; dtype mismatch must
  raise, not cast). `np.copyto` executes as a host C-level copy — **no torch op is constructed**, so
  nothing can dispatch onto the torch/Level Zero command-list path. This is the entire fix.
- **A/B switch**: new env `VLLM_PLE_HOST_STAGE` (add to envs.py alongside VLLM_PLE_CPU_OFFLOAD):
  default **1** (numpy np.copyto path); `0` = legacy torch `copy_` path, kept in-tree. Read once in
  `PleOffloadConnector.__init__`; branch inside `_copy_cpu_inputs`. Rationale: same-image A/B on the
  rig isolates "the change" from "the new stage", and it is the emergency fallback if the numpy path
  misbehaves — no rebuild. Default must be 1 (never default a known-wedging path).
- **Scope guard**: `_copy_cuda_inputs` (MRV2 D2H + event sync) is NOT touched in v24e. If a future
  wedge ever shows the D2H path, that needs stream-ordering treatment, not numpy staging.
- **Invariants**: ZMQ protocol/socket.send unchanged; worker side untouched; correctness contract
  unchanged (still one pending request, `maxsize=1`, PLE rejects DBO); `signal_dummy_outputs` /
  capture-forwards (dummy runs) unchanged — staging env is irrelevant during capture because dummy
  runs never enqueue.
- **Vendor before commit**: extend `files/overlay/vllm/0003-…` or add `0019-…` and register it in
  `build-image.sh`; delete/never-vendor the v24d from_numpy staging; also make sure the A/B switch
  defaults (1) make the vendored patch and the image line agree (today's vendored 0003 is the
  legacy-repo pre-v24d state — equivalent to switch=0).

## 3. Campaign pre-registration — fix-in-first, 5/7 → 0

**Decision (Ryan)**: stage-v24e goes in ALONE; the lock diagnostic is staged but NOT deployed
(held as fallback instrument); the PLE-disable discriminator (config-only,
2026-09-19-ple-disable-experiment-prereg.md) remains the fallback experiment ladder.

### Baseline and distinguishability (pre-registered arithmetic)

- Baseline to move: **5 stalls / 7 bursts** (0.71/burst) on stage-v24d, 8-way bursts.
- Under p=5/7, P(0 stalls | N): N=7 → 1.6e-4; N=10 → 3.6e-6; N=15 → 6.9e-9.
  → **0/7 already falsifies "still 5/7" at α≈0.00016; 0/10 or 0/15 is decisive.**
- **5/7 vs 2/7 is NOT distinguishable at these N** (N=7: P(≤2 stalls | p=5/7) ≈ 0.023 — an
  improved-but-nonzero rate looks like noise at this sample size). Therefore the only claimable
  clean win is ~zero stalls; a "reduced from 5/7 to 2/7" claim is NOT pre-registered as
  demonstrable. This is why the stability gate is zero-timeouts, not aggregate.

### Protocol (per burst, unchanged from the 5/7-baseline identity)

- Bench: `soakfix.py` 8 streams × 4 rounds × 600 tok (r1-r4 + agg), warmup 200 first.
- **Gate (stability, per Ryan's standing correction — aggregate is measurement, not gate)**:
  PASS requires **zero `sample_tokens` timeouts** across the campaign AND the **post-generation
  probe 200 after every burst** (1-token generation probe — `/v1/models` is blind to engine death,
  confirmed twice; never use it as health) AND no watchdog restart/container death.
- Aggregate (agg, per-round step rate) is recorded and reported per burst; **no PASS/FAIL on it**.
  Warn-level telemetry only: agg < 50% of the healthy reference (r2=168.8, stall#4 doc) sustained
  across 2 rounds — log, continue, do not abort (stall#4 r1=61/r3=107 rounds pre-stall).
- **Burst count**: 10 clean bursts minimum to declare PASS; target 15 (time-boxed). Any stall
  → no declaration (see decision tree).

### Boot provenance on every number

Every burst metric row carries: boot ID, `docker ps --format '{{.Image}}'` = stage-v24e verified
within 15s of launch (v24-lineage lesson, graph-mode doc), `.env` diff vs baseline — **only IMAGE
and VLLM_PLE_HOST_STAGE may differ** (everything else pinned: PLE_CPU_OFFLOAD_GB=12.25,
MAX_NUM_SEQS=8, MAX_NUM_BATCHED_TOKENS=2048, TP4+EP4 allgather_reducescatter, MTP3, `-O 0`,
gpu-mem-util 0.75, XPU-graph flags per the current serving line), KV tokens, boot_clock.jsonl phase
clocks, git describe, `.run/manifest.json`. Per launch, `docker inspect` must show
`VLLM_PLE_CPU_OFFLOAD=1` and `VLLM_PLE_HOST_STAGE=1` — the gate from §1 is part of the check now.

### Abort criteria

1. Any `sample_tokens` timeout / engine death / post-gen probe failure → campaign FAIL; **capture
   before any restart** (py-spy native ×5 procs + stallspy; watchdog-v2 field finding: capture on
   detection, never after rm).
2. Watchdog restart or launch failure → that burst fails; campaign pauses for review.
3. Boot OOM/crash at PLE load → config cannot be represented; abort, record.
4. Warn-level telemetry sustained 3 consecutive bursts → pause + review, resume only on go.

### Decision tree (pre-registered verdicts)

| outcome | verdict |
|---|---|
| 0 stalls across ≥10 bursts (target 15) | **PASS** — staging-copy theory holds; v24e becomes serving line; vendor patch + fix start.sh:458 comment + commit |
| 1 stall in ≥10 | no declared pass. Captures decide: same `appendUSMMemcpy` connector-thread signature → fix incomplete (next suspect: worker-side or MRV2 D2H copy); different site → re-open the unified-mechanism claim |
| ≥2 stalls in ≤10 | no improvement vs 5/7; mechanism as stated incomplete → deploy lock diagnostic (fallback) or PLE-disable discriminator to attribute the wedging actor |

Optional (end of day, Ryan's go only): 2-burst `VLLM_PLE_HOST_STAGE=0` A/B on v24e — deliberate
regression that should re-wedge on burst 1 if the mechanism + switch are right. Run LAST; a wedge
here is a passed test, not a campaign failure.

## 4. Status

- Doc written; **no commit** (Ryan reviews first). v24d still the in-rig line; stage-v24d patch is
  unvendored (collect from the image). Next: build stage-v24e, apply §1 launcher hygiene, run the
  pre-registered campaign. Evidence anchors: stall-mechanism doc (5/7 site), stall#4 doc (partial
  rounds + probe blindspot), graph-mode doc (133 tok/s line), ple-disable prereg doc (fallback ladder).

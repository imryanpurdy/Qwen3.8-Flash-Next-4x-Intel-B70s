# Hermes Agent clients — verified configuration (2026-10-04)

This stack serves OpenAI-compatible chat at `:8022` (served id `qwen-256k`,
262,144-token window). This doc records the client configuration we verified
against the production engine, including an agent-framework (Hermes) delegation
setup and a child-agent verification habit that measurably reduces fabricated
values.

## Why temperature ≤ 0.7

The sampler pin ships `temperature: 0.7` (`top_p 0.80, top_k 20`,
`presence_penalty 1.5`). This is the **ceiling**, not a suggestion: measured
long-context recall (random-position record lookups in 60K/100K-token
documents) holds at ≤0.7, and every recall miss we logged was a confident
*wrong value*, not a refusal. Higher temperatures widen exactly the failure
mode (plausible-but-wrong codes) that long-context work can least afford.
Keep client-side temperature at or below the pinned 0.7; if your client
ignores the server pin, set `temperature: 0.7` (or lower) explicitly.

## Delegated agents (Hermes `delegate_task`)

### The verified config block

These values were verified live against the production engine (children
running 16/16 heavy-file analysis tasks, two rounds, zero loops/timeouts,
zero compactions; see "Results" below). In Hermes `config.yaml`:

```yaml
# 1) The delegation block (child agents)
delegation:
  model: qwen38-flash-next
  provider: custom
  compression_threshold_tokens: 60000   # compact child history at 60K tokens
  max_iterations: 100                   # stop runaway children
  child_timeout_seconds: 1200           # 20-minute inactivity cap
  max_concurrent_children: 16

# 2) Under the custom-provider entry for the gateway, in its `models:` map —
#    this is the pin that actually caps child context:
#    <your-provider-entry>:
#      models:
#        qwen38-flash-next:
#          context_length: 100000        # keeps child contexts ≤ ~100K;
#                                         # caps one tool result at ~60,000 chars

# 3) Top-level providers block (created if absent):
providers:
  custom:
    models:
      qwen38-flash-next:
        stale_timeout_seconds: 300      # give a large prompt up to 5 min to start
```

### Which id each path resolves to (verified)

- Children spawned by `delegate_task` resolve as provider id **`custom`**
  (bare), model `qwen38-flash-next`.
- The `delegation:` block carries `model:`/`provider:`/timeouts; the
  `context_length` pin lives under the **custom_providers entry's** `models:`
  map for `qwen38-flash-next`.
- `stale_timeout_seconds` lives under the **top-level `providers:` →
  `custom:` → `models:` → model entry** (explicit 300 bypasses an implicit
  240s scaling of the default).

### Primary sessions running Qwen as the main model

For a primary (non-delegated) session, the same cap is set on the `model:`
block instead:

```yaml
model:
  provider: custom:<gateway-host:port>
  default: qwen38-flash-next
  base_url: http://<gateway-host:port>/v1
  context_length: 100000
```

There is no separate `delegation.context_length` key in current Hermes
versions — setting one saves but is not read ("not a recognized config key");
the `custom_providers[].models.<model>.context_length` / `model.context_length`
pins are the mechanisms that actually resolve. Same model id, same pin
mechanics: the tool-result cap (~60K chars) follows from the 100K window.

### The delegation-verify skill (verbatim)

Ship this as a Hermes skill (`~/.hermes/skills/delegation-verify/SKILL.md`).
It exists because a child under context pressure will report plausible values
(line numbers, IDs, quotes) it never actually read. Forcing a tool-read
before reporting measurably cuts fabricated values. Text, verbatim:

```markdown
---
name: delegation-verify
description: Use whenever delegating work with delegate_task. Every child goal must require tool-verified exact values.
---
# Delegation verification

Whenever you call delegate_task, end every child goal with this sentence, verbatim:

"Verify exact values (line numbers, IDs, paths, quotes, numbers) with a tool before reporting them; never report a value you didn't read from a tool result in this task."

Do not shorten or paraphrase it. Apply it to every child in a batch.
```

> Note: current Hermes versions cap skill `description:` at ~57 characters in
> the skill index; if creation is refused, shorten the description line (the
> body must stay verbatim). Example accepted description:
> `Use when delegating — every child must tool-verify values.`

**Primary-model variant** (for users running Qwen as their main model, no
delegation): end every prompt that asks for exact values with:

> "Verify exact values (line numbers, IDs, paths, quotes, numbers) with a tool before reporting them; never report a value you didn't read from a tool result in this task."

or save it as a project rule (`.hermes.md` / `AGENTS.md`) so it applies to
every turn.

## Results with this client config (production engine, 2026-10-04)

All runs: thinking off, sampler pin as shipped (temp 0.7). Timestamps UTC.

### Recall — the user regime (≤100K total context)

100 lookups, documents of ~60K and ~100K tokens, 50 records per document at
seeded random positions, concurrency 8:

| Depth | Correct | Wrong values | Non-answers |
|---|---|---|---|
| ~60K | 48/50 | 2 | 0 |
| ~100K | 47/50 | 3 | 0 |
| **Total** | **95/100** | **5** | **0** |

Every miss was a wrong value (`finish=stop`, no refusals, no API errors):

| Doc | Record | Truth | Answered |
|---|---|---|---|
| 60K | 347 | JWTXP | ZYJYB |
| 60K | 232 | AWPHW | AOPHW (1-char slip) |
| 100K | 2663 | DLJPY | EXPDE |
| 100K | 1735 | UZJXR | KFUPA |
| 100K | 962 | XRYUE | GTWMQ |

### The 200K mid-document weakness

In a 200K-token document (records past the 100K regime), recall at the ~118K
position collapsed to **12/34** (22 misreads), while the same run read 60K at
28/34 and 200K-position records at 29/32. The failure is concentrated
mid-document, not monotonic with depth — and it does **not** appear at ≤100K.
This is why the client cap of 100K is the verified operating envelope: inside
it, recall is 95/100; a single 118K-deep lookup in a 200K document is a
coin-flip against the model. Re-test before raising the cap.

### Load behavior

- **60-minute soak** (8-way concurrent, mixed 60K/118K depths): 3,632/3,632
  requests OK, zero engine resets, zero restarts, p50 7.7 s, 49.3 tok/s
  single-stream (−0.4% vs the W4A16 checkpoint's 49.2).
- **Overnight soak** (8h: 1-token canary every 5 min + 4-way mixed 30–100K
  burst every 30 min): 59 canaries (0.06–0.24 s), 10 bursts (4×200 in 29.4 s
  at 30K depth), 0 errors, 0 resets.
- **Heavy-file agent runs** (8 concurrent Hermes children reading/analyzing
  5–10K-line source files, ≤100K context): **16/16 completed** across two
  rounds (20m17s and 18m41s wall), zero loops, zero timeouts, zero stalls,
  zero context compactions. This is the delegated-agent shape the config
  block above was verified with.

## Troubleshooting

### Spotting copy-engine resets in dmesg

The freeze signature is Level-Zero copy-engine (bcs) resets:

```bash
sudo dmesg | grep -aE "Engine reset: engine_class=(ccs|bcs)"
```

- `engine_class=bcs` (copy engine) resets clustering at freeze timestamps =
  the copy-offload failure this stack disables with
  `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1`.
- `engine_class=ccs` (compute) resets alongside usually indicate the shared
  root reset event.
- On a wedge, all 4 TP workers show py-spy stacks stuck in
  `async_tensor_h2d` (H2D copy) and the engine core blocks in
  `shm_broadcast.wait`. A reset burst with no API traffic = the wedge, not
  load.
- dmesg may be restricted (`read kernel buffer failed: Operation not
  permitted`) — use `sudo dmesg`; a monitoring script reading unprivileged
  dmesg silently reports 0 resets.

### The empty-env-var trap

`UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=` (declared but **empty**) is not
"unset": oneCCL parses the empty string and crashes worker init
(`unexpected value: , expected values: 0, 1`). Same class for
`CCL_ZE_CACHE_OPEN_IPC_HANDLES`. In `start.sh`, pass such flags only when
set:

```bash
${UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD:+-e UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=$UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD}
```

Unset in `.env` → omitted from the docker line entirely. Never
`-e "VAR=${VAR:-}"`.

### Rebooting after DEVICE_LOST bursts

A burst of Level-Zero resets can leave devices in `DEVICE_LOST` that a
container restart alone does not clear (re-launches fail with
`UR_RESULT_ERROR_OUT_OF_RESOURCES` on all workers). Recovery order:

1. `./start.sh stop` (watchdog + container).
2. Retry one boot. If workers fail OUT_OF_RESOURCES again on an otherwise
   idle host, the device state is dirty — reboot the host.
3. After reboot, run the pre-boot XPU gate (automatic in `start.sh`) and the
   standard READY gate before declaring recovery.

## Ruled out (with evidence)

- **Prompt-length overflow as the freeze cause** — freezes reproduced at
  60K–118K depths far below MML 262,144; not a length bug.
- **Kernel 7.x as escape hatch** — 7.0.0-31 carries the job-timeout fix but
  lacks the flat-CCS fix (landed 6.18.51); and 7.x/GuC-newer wedges
  permanently on B70 (vllm#41663). Stay on 6.17.0-1010-intel.
- **Prefix caching as the wedge trigger** — freezes fired on prefix-cold
  random-block loads; disabling prefix caching would not have prevented
  them (the copy-engine resets are load-shape-triggered, not cache-hit
  -triggered).
- **Watchdog float-compare bug** — tested live: awk parses `4.06e+06`-style
  labeled floats and bash `-gt` compares them numerically. The watchdog
  restarts were real load-gate misreads (now fixed by the load-aware gate),
  not arithmetic bugs.

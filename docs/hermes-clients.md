# Hermes Agent clients — verified configuration

This stack serves OpenAI-compatible chat at `:8022` (served id **`qwen-256k`**, 262,144-token window) on the Lumnus `b70-flash-next` engine. This doc records the client configuration verified against the production engine, including the Hermes delegation setup and a child-agent verification habit that measurably reduces fabricated values.

## Endpoint + sampler block

```json
POST http://<host>:8022/v1/chat/completions
{
  "model": "qwen-256k",
  "temperature": 0.7,
  "top_p": 0.80,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 1.5
}
```

### Why temperature ≤ 0.7

The engine ships this sampler pin (`--override-generation-config`), and it is the **ceiling, not a suggestion**: measured long-context recall (random-position record lookups in 60K/100K-token documents) holds at ≤0.7, and every logged miss was a confident *wrong value*, not a refusal. Higher temperatures widen exactly the failure mode (plausible-but-wrong codes) that long-context work can least afford. If your client ignores the server pin, send the values above explicitly.

Reasoning: the engine parses `qwen3` reasoning — thinking is off with `enable_thinking: false` or `reasoning_effort: "none"`; when on, per-effort budgets are capped server-side (512–12288 tokens). Tool calls use the `qwen3_xml` parser (verified: 20/20 structural, multi-tool, nested args).

## Hermes `config.yaml` block

```yaml
# Primary session (or just the delegation block below for delegated children)
model:
  provider: custom:<host:8022>
  default: qwen-256k
  base_url: http://<host>:8022/v1
  api_key: <any-nonempty-string>      # the engine has no auth; the client requires a non-empty key
  context_length: 100000              # the verified envelope — see "The 200K mid-document weakness"

# Delegated children (delegate_task)
delegation:
  model: qwen-256k
  provider: custom
  compression_threshold_tokens: 60000   # compact child history at 60K tokens
  max_iterations: 100                   # stop runaway children
  child_timeout_seconds: 1200           # 20-minute inactivity cap
  max_concurrent_children: 16

# Where the pins actually resolve (verified):
# - child context cap: the custom-providers entry's models: map —
#     custom_providers[].models.qwen-256k.context_length: 100000
#   (there is no delegation.context_length key; setting one saves but is not read)
# - stale timeout: top-level providers: block —
providers:
  custom:
    models:
      qwen-256k:
        stale_timeout_seconds: 300      # a large prompt gets up to 5 min to start
                                        # (explicit 300 bypasses the implicit 240s default)
```

The 100,000-token context pin also caps a single tool result at ~60,000 chars — that is the verified operating shape, not a limitation of Hermes.

### The ~6-minute retry caveat during restarts

The watchdog restarts the engine on a wedge (and any operator restart takes ~4–5 min cold: graph capture runs inside torch.compile before `Application startup complete`). During that window `/health` may keep answering 200 while completions hang, and a client with a short stale-timeout gives up too early or double-fires. Practical rules:

- A stalled request during a restart window resolves once the engine is READY again — allow up to **~6 minutes** of retry/backoff before declaring failure (`stale_timeout_seconds: 300` + one retry covers it).
- Liveness probe = a 1-token chat completion, never `/health` or `/v1/models` (both stay 200 while the engine is wedged).
- Allow ≥180 s for a completion under load: a busy engine queues; it is not down.

### The delegation-verify habit

A child under context pressure will report plausible values (line numbers, IDs, quotes) it never actually read. End **every** delegated child goal with this sentence, verbatim, and it measurably cuts fabricated values:

> "Verify exact values (line numbers, IDs, paths, quotes, numbers) with a tool before reporting them; never report a value you didn't read from a tool result in this task."

Do not shorten or paraphrase it; apply it to every child in a batch. Same sentence as a project rule (`.hermes.md` / `AGENTS.md`) covers primary sessions running Qwen as the main model.

## Results with this client config (production Lumnus line)

All runs: thinking off, sampler pin as shipped (temp 0.7), concurrency 8, ≤100K total context.

### Recall — the user regime (≤100K)

100 lookups, documents of ~60K and ~100K tokens, 50 records/doc at seeded random positions:

| Depth | Correct |
|---|---|
| ~60K | 50/50 |
| ~100K | 49/50 |
| **Total** | **99/100** (production baseline, devan fork engine + wtdcode AWQ checkpoint: 95/100 historical) |

The single miss: 100K doc, record 2663 — truth `DLJPY`, answered `EXPDE`. Every miss on this stack class is a wrong value with `finish=stop`; no refusals, no API errors.

### The 200K mid-document weakness

In a 200K-token document, recall at the ~118K position collapsed to **12/34** while the same run read 60K at 28/34 and 200K-position records at 29/32 — concentrated mid-document, not monotonic with depth, and absent at ≤100K. **This is why the 100K client cap is the verified envelope.** Re-test before raising it.

### Load behavior

- **60-minute soak** (8-way concurrent, mixed depths): 389 waves, 3,112/3,112 OK, 0 errors, p50 8.9 s, max 10.0 s, 0 restarts, 0 engine resets.
- **Heavy-file agent batch** (8 concurrent Hermes children reading/analyzing 5–10K-line source files, ≤100K context): **8/8 completed, 39 tool calls clean, 0 loops, 695 s wall** (previous production line: 7/8).
- **Prefix reuse** (same document, new question, 118K): warm hit 118,144 tokens, cold-cold/cold-warm drift 0.0 (limit 0.06), 48 responses, 0 errors, 0 degenerate — the engine's offload fix working; long shared-prefix agent sessions get their TTFT back.

### Agent-loop A/B (the repetition-stop regression, resolved)

5 replays of the original PLE-guard trace (the trace that, on the previous engine, began repetitive streaming at ~43 min and died in an engine repetition stop at ~51 min without completing) on the current production engine:

| | Original engine | Current engine (5 replays) |
|---|---|---|
| Completed naturally, full findings | 0/5 | **2/5** |
| Hit harness iteration budget while still working productively | — | 3/5 (~70 min) |
| Repetitive streaming observed | began ~43 min | **0/5** |
| Ended in a repetition stop | ~51 min | **0/5** |

The failure mode is gone; the remaining 3/5 are a harness budget, not an engine fault. If your harness iteration cap is < ~100, raise it for long agent tasks on this model.

## Troubleshooting (client-side)

- **~6-min stall after a watchdog restart** → see the retry caveat above; probe with a 1-token completion.
- **Confident wrong values on exact lookups** → temperature is above 0.7 somewhere in your client, or context exceeds ~100K, or (delegated children) the goal lacked the verify sentence.
- **HTTP 400 on very long requests** → prompt exceeded 262,144 tokens; the 100K client cap exists well below this for recall, not capacity, reasons.
- Server-side freezes (dmesg `engine_class=bcs` resets, empty-env trap, DEVICE_LOST recovery) are in the main README's Troubleshooting.

# QSA replacement verdict — devan-carlin fork (subagent deleg_080947ad, 2026-09-23)

**Verdict: DENSE-FULL-CONTEXT — unconditional, no flag, no runtime knob.**
Their stack does NOT compute the same model. All 12 full-attention layers run
stock dense `Qwen3NextAttention`; the `.indexer.*` checkpoint weights are
unconditionally skipped at load; the real sparse path exists only in upstream
vLLM PR #53896 (merged `e126687a…`), which their fork base `c39076fef` predates.

## Evidence (from subagent report, verbatim tables)

| Claim | Fork file:line | Patch lines | Decisive code |
|---|---|---|---|
| Full-attn layers instantiate DENSE `Qwen3NextAttention` | `qwen4_exp.py:571-579` @ `a69fba21` | 1076-1084 | `elif self.layer_type == "full_attention":` `self.self_attn = Qwen3NextAttention(...)` |
| forward() has no indexer/selection call | `qwen4_exp.py:637-640` | 1142-1145 | `if self.layer_type == "linear_attention": … else: cur = self.self_attn(...)` |
| Weight-loading skip of `.indexer.` | `qwen4_exp.py:1023-1031` (also :1201-1205, :1377-1382 VL) | 1528-1536 / 1703-1712 / 1885-1895 | `AutoWeightsLoader(self, skip_substrs=[".ngram_embedding.", ".indexer."])` |
| Author's own statement | `qwen4_exp.py:16-17` | 521-522 | "**QSA indexer** on the full-attention layers (v1 falls back to dense attention)." |
| Config indexer params unused | `vllm/transformers_utils/configs/qwen4_exp.py:96-101` | 2458-2463 | `indexer_budget=2048, indexer_compress_ratio=4, indexer_head_dim=128, indexer_kv_heads=1, indexer_n_heads=4` — never consumed |
| No sparse/env toggle | grep `os.environ` fork `qwen4_exp.py` | — | only `QWEN4EXP_DEBUG_*`, `QWEN4EXP_DISABLE_PLE`, `QWEN4EXP_FORCE_STD_ROPE` |
| Underlying op is stock dense FlashAttn | fork `qwen3_next.py:268-…` | 448-497 | `qkv_proj → _project_qkv_gate → self.attn(q,k,v)` (patch adds debug dumps only) |

Their own docs corroborate (`electric-sheep/docs/qwen4exp-vllm-port.md` L34-36,
L76-81, L115): "v1: use DENSE attention instead", "true sparse indexer is a
later phase", "Not bit-exact vs llama.cpp (dense vs sparse indexer)".

## Unused checkpoint tensors (per full-attention layer, ×12)

Whole `layers.N.self_attn.indexer` submodule skipped:
- `index_qk_proj` — (4+1)×128 = 640 × 2560
- `q_layernorm` [128], `k_layernorm` [128] (GemmaRMSNorm)
- fp8 `weight_scale` variants under `.indexer.`
- functionally unused: compressed QSA key cache (compress_ratio=4), top-k
  selection from indexer logits (`indexer_budget=2048`)

## Fidelity implication (subagent)

- **≤ ~2048 tokens: near-identical semantics** (top-k would select the whole
  sequence anyway; residual differences numeric only).
- **> ~2048 tokens: structurally divergent.** The model was trained with the
  sparse gate deciding what each full-attn layer sees (~2048 indexer-scored
  tokens); their dense path attends to ALL tokens. Divergence concentrates in
  needle retrieval / long-doc QA (attention dilution — systematic softmax
  difference, not a perturbation) and accumulates per decode token via shifted
  prefill logits.
- No sparse path exists in their tree (`vllm/models/qwen4_exp/` absent; the
  in-tree `sparse_attn_indexer.py` is the DeepSeek-V4 indexer, unwired). No
  runtime knob. Upstream #53896 (`vllm/models/qwen4_exp/nvidia/indexer_qsa.py`,
  `qsa.py`, `common/qsa_cache.py`) is the real implementation; fork base is an
  ancestor of that merge (behind 0, ahead 897) — they never pulled it.

## Rig rider — predictions for the fidelity gate (added by Hermes)

The sidelane-fidelity gate (8 fixed short prompts ×2 temp-0 + ~32K/~80K + the
128K/250K needles) should therefore show:
1. Eight fixed prompts: high token agreement (near-identical semantics) —
   agreement here CANNOT prove same-model.
2. Long 32K/80K: first-divergence well before the marker, final-answer match
   on the marker questions likely degraded vs production.
3. 128K/250K needles (gate 4): where divergence is maximal by mechanism.
Per Ryan's directive: if the long-context rows diverge, report the stack as a
DIFFERENT MODEL VARIANT — switch decision is Ryan's, nothing auto-switches.

## Sources

Local patch `es_vllm_patches_qwen4exp-xpu-port.patch` (118 KB); fork raw files
@ `a69fba21212952601bbacd81a2f2f93f72744eae` (qwen4_exp.py 1435 lines,
qwen3_next.py, configs/qwen4_exp.py); fork tree API; electric-sheep
qwen4exp-vllm-port.md; upstream PR #53896 @ merge `e126687a9a82…`;
compare API c39076fef…e126687a; HF Qwen/Qwen3.8-Flash-Next config.json.
Full transcript: hermes\cache\delegation\live\deleg_080947ad\task-0.log

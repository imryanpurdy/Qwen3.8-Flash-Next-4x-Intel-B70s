# Dense-QSA model-variant verdict — devan-carlin fork

**Verdict: DENSE-FULL-CONTEXT — unconditional, no flag, no runtime knob.**
The fork does NOT compute the same model as the sparse-QSA checkpoints. All
12 full-attention layers run stock dense `Qwen3NextAttention`; the
`.indexer.*` checkpoint weights are unconditionally skipped at load; the real
sparse path exists only in upstream vLLM PR #53896 (merged `e126687a…`),
which the fork base `c39076fef` predates.

## Evidence

| Claim | Fork file:line @ `a69fba21` | Decisive code |
|---|---|---|
| Full-attn layers instantiate DENSE `Qwen3NextAttention` | `qwen4_exp.py:571-579` | `elif self.layer_type == "full_attention":` `self.self_attn = Qwen3NextAttention(...)` |
| forward() has no indexer/selection call | `qwen4_exp.py:637-640` | `if self.layer_type == "linear_attention": … else: cur = self.self_attn(...)` |
| Weight-loading skip of `.indexer.` | `qwen4_exp.py:1023-1031` (also :1201-1205, :1377-1382 VL) | `AutoWeightsLoader(self, skip_substrs=[".ngram_embedding.", ".indexer."])` |
| Author's own statement | `qwen4_exp.py:16-17` | "**QSA indexer** on the full-attention layers (v1 falls back to dense attention)." |
| Config indexer params unused | `vllm/transformers_utils/configs/qwen4_exp.py:96-101` | `indexer_budget=2048, indexer_compress_ratio=4, indexer_head_dim=128, indexer_kv_heads=1, indexer_n_heads=4` — never consumed |
| No sparse/env toggle | grep `os.environ` in fork `qwen4_exp.py` | only `QWEN4EXP_DEBUG_*`, `QWEN4EXP_DISABLE_PLE`, `QWEN4EXP_FORCE_STD_ROPE` |
| Underlying op is stock dense FlashAttn | `qwen3_next.py:268-…` | `qkv_proj → _project_qkv_gate → self.attn(q,k,v)` |

The fork's own docs corroborate (`electric-sheep/docs/qwen4exp-vllm-port.md`
L34-36, L76-81, L115): "v1: use DENSE attention instead", "true sparse
indexer is a later phase", "Not bit-exact vs llama.cpp (dense vs sparse
indexer)".

## Unused checkpoint tensors (per full-attention layer, x12)

Whole `layers.N.self_attn.indexer` submodule skipped:
- `index_qk_proj` — (4+1)x128 = 640 x 2560
- `q_layernorm` [128], `k_layernorm` [128] (GemmaRMSNorm)
- fp8 `weight_scale` variants under `.indexer.`
- functionally unused: compressed QSA key cache (compress_ratio=4), top-k
  selection from indexer logits (`indexer_budget=2048`)

## Fidelity implication

- **≤ ~2048 tokens: near-identical semantics** (top-k would select the whole
  sequence anyway; residual differences numeric only).
- **> ~2048 tokens: structurally divergent.** The model was trained with the
  sparse gate deciding what each full-attn layer sees (~2048 indexer-scored
  tokens); the dense path attends to ALL tokens. Divergence concentrates in
  needle retrieval / long-doc QA (attention dilution — systematic softmax
  difference, not a perturbation) and accumulates per decode token via
  shifted prefill logits.
- No sparse path exists in the fork tree (`vllm/models/qwen4_exp/` absent;
  the in-tree `sparse_attn_indexer.py` is the DeepSeek-V4 indexer, unwired).
  No runtime knob. Upstream #53896
  (`vllm/models/qwen4_exp/nvidia/indexer_qsa.py`, `qsa.py`,
  `common/qsa_cache.py`) is the real implementation; the fork base is an
  ancestor of that merge (behind 0, ahead 897) — it was never pulled.

## Prediction confirmed by the fidelity gate

An 18-row harness comparison (8 fixed short prompts x2 temp-0 + ~32K/~80K
long rows + the 97K/250K needles) shows: long-context rows (32K, 80K) agree
at 1.0000, and all divergence is short-row behavioral (2 real code-path
diffs) or comparator artifact — verdict `DIFFERENT_MODEL_VARIANT_SHORTS_ONLY`.
If the long-context rows had diverged, the correct report would still be
DIFFERENT MODEL VARIANT; nothing auto-switches between checkpoints.

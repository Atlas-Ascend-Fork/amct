# Qwen3-30B-A3B

## Summary

- This is the `qwen3_moe` adapter validation case under the current amct_llm flow with `seq_len=4096`.
- Model path: `<path-to>/Qwen3-30B-A3B`
- Adapter: `amct_pytorch/common/models/llm/qwen/qwen3/qwen3_moe.py`
- Fixed Wikitext parquet: `<path-to>/wikitext-2-raw-v1/test-00000-of-00001.parquet`
- Completed: BF16 baseline, quant-disabled equivalence, MoE minimum PTQ smoke, first-round direct quant eval, deploy artifact review, and dynamic-quant-qmm theoretical performance review.

## Structure

- Qwen3 MoE decoder; adapter name is `qwen3_moe`.
- Key config: `hidden_size=2048`, `num_hidden_layers=48`, `num_experts=128`, `num_experts_per_tok=8`, `moe_intermediate_size=768`, `num_attention_heads=32`, `num_key_value_heads=4`.
- Per-expert checkpoint weights are packed by `pack_gated_expert_weights`; quantization exposes expert PTQ units through `QuantGatedExperts`.

## Adapter Validation

- BF16 baseline:
  - Wikitext PPL, `seq_len=4096`
  - `ppl = 8.044384956359863`
  - Output: `<eval: BF16 baseline (seq4096)>`
- Quant-disabled equivalence:
  - Target: `attn-linear + moe`
  - Config: `a_bits=16, w_bits=16`
  - `ppl = 8.03922176361084`
  - Output: `<eval: quant-disabled attn+moe (seq4096)>`
  - Result: close to BF16 baseline; difference is about `-0.00516`, acceptable for this validation step.
- Minimum PTQ smoke:
  - Target: `block0/moe`
  - Enumeration: 128 MoE expert PTQ units on block0.
  - Input source: real Wikitext hidden states routed through the block0 gate; selected actually-hit `expert_120`.
  - Input cache: `<adapter cache: ptq-smoke moe block0 input>`
  - Exported params: `<adapter exported params: ptq-smoke moe block0 expert>`
  - Result: a MoE expert unit can be enumerated, prepared with input/GT, and exported through the PTQ chain.

## First-Round Quantization

- Recommended scheme: `attn-linear + moe`
- `quant_dtype`: `int`
- Bits: `a_bits=8, w_bits=8`
- `granularity`: `block`
- Algorithm: direct quant; no PTQ parameter directory is used.
- Eval: Wikitext PPL, `seq_len=4096`.
- Output: `<eval: W8A8 attn+moe (seq4096)>`
- Result:
  - `ppl_bf16 = 8.044384956359863`
  - `ppl_quant = 8.028042793273926`
  - `delta = -0.0163421630859375`
- Decision:
  - Passes the default `delta <= 0.2` threshold.
  - Quant PPL is slightly lower than BF16; this is treated only as acceptable, not as evidence that quantization improves quality.
  - Sanity check passed: quant-disabled equivalence is aligned, dataset and `seq_len=4096` match the current flow.

## Dynamic Quant + QMM Performance Review

- Skill: `skills/quant-tools/dynamic-quant-qmm-analysis`
- Scope: theoretical Linear `dynamic quant + qmm` benefit versus BF16 matmul only; this is not an end-to-end runtime measurement and does not prove infer `MoEGMM` performance.
- Platform assumption: `A2/A3` bucket, `int_w8a8`, offline-prepared weights, dynamic activation quant.
- Shape source: `M=4096` from Wikitext `seq_len=4096`; routed expert average `M_eff = 4096 * 8 / 128 = 256`; `K/N` from model config and Linear weight shapes.

| Compare unit | Shape scope | Direction | Estimated speedup | Main bottleneck / risk |
| --- | --- | --- | --- | --- |
| Attention projection `q/k/v` group | q: `M=4096,K=2048,N=4096`; k/v: `M=4096,K=2048,N=512`, shared activation quant | Positive | about `1.25x - 1.35x` | narrow `k/v` N remains a group-level risk |
| Attention projection `o_proj` | `M=4096,K=4096,N=2048` | Positive | about `1.25x - 1.35x` | qmm cube / baseline MTE2 |
| Routed MoE expert `up/gate` | average `M_eff=256,K=2048,N=768` x2, shared activation quant | Uncertain to negative | no positive speedup claimed | dynamic quant / fixed overhead / small M |
| Routed MoE expert `down` | average `M_eff=256,K=768,N=2048` | Uncertain to negative | no positive speedup claimed | dynamic quant / fixed overhead / small M |

Performance conclusion: the `attn-linear + moe` W8A8 Int8 scheme is precision-acceptable, but performance is `Conditional Go`. The attention projection path is theoretically positive; the MoE path is risky under the AMCT per-expert dynamic-quant-qmm model because average routed expert `M_eff` is small. If downstream uses packed `MoEGMM` and can merge expert work efficiently, MoE performance may improve, but that must be proven by infer runtime validation or MoEGMM profiling.

## Deploy Artifact

- Directory: `<deploy artifact: W8A8 attn+moe>`
- Mode: direct quant
- Key files:
  - `config.json`
  - `model.safetensors.index.json`
  - 49 `*.safetensors` files
  - `deploy_quantization.md`
- `.exit_code = 0`
- Note: `deploy_quantization.md` records the downstream consumption contract; this casebook records the scheme decision, precision result, and qmm performance conclusion.

## Known Issue

- Issue: a broad `hook_name="mlp"` capture can also match `mlp.gate`, and Qwen3 MoE gate returns a tuple.
- Symptom: `AttributeError: 'tuple' object has no attribute 'detach'` during activation capture.
- Workaround used here: build real Wikitext hidden states, route them through the gate, select an actually-hit expert, save `block_0_moe_in.pkl`, and calibrate that expert unit only for the minimum PTQ smoke.

## Reuse Notes

- For another Qwen3 MoE model, start from `qwen3_moe.py`, `QuantGatedExperts`, and `pack_gated_expert_weights`.
- Keep the current default `seq_len=4096` and the fixed Wikitext parquet path.
- A first pass can validate precision with `attn-linear + moe` W8A8 Int8 direct quant. If performance is a hard gate, add routed token count analysis or infer-side MoEGMM runtime validation before accepting the MoE path.


## QMM Runtime Validation

- Runner: `skills/quant-tools/dynamic-quant-qmm-analysis/validate_qmm.py`
- Profile output: `<perf data: QMM 8B/30B validation>`
- Consumed target: W8A8 `DynamicQuant + QuantBatchMatmul`; weight preparation is offline.
- MoE shape uses average routed expert `M_eff = 4096 * 8 / 128 = 256`.

| Compare unit | BF16 baseline | DynamicQuant + QMM | Measured speedup | Decision |
| --- | ---: | ---: | ---: | --- |
| Attention `q/k/v` group | 464.139 us | 359.639 us | 1.291x | Positive |
| Attention `o` | 332.787 us | 273.703 us | 1.216x | Positive |
| Routed MoE `up/gate`, average expert | 31.747 us | 75.765 us | 0.419x | Negative in AMCT per-expert qmm path |
| Routed MoE `down`, average expert | 16.369 us | 40.178 us | 0.407x | Negative in AMCT per-expert qmm path |

Runtime-validation conclusion: measured AMCT qmm validation supports `attn-linear`, but not the average routed MoE per-expert path. The final MoE performance decision remains conditional on infer-side packed `MoEGMM` runtime validation.

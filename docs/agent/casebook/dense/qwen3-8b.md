# Qwen3-8B

## Summary

- This is the `qwen3` dense adapter validation case under the current amct_llm flow with `seq_len=4096`.
- Model path: `/home/xyj/dev/models/Qwen/Qwen3-8B`
- Adapter: `amct_pytorch/common/models/llm/qwen/qwen3/qwen3.py`
- Fixed Wikitext parquet: `/home/xyj/dev/datasets/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet`
- Completed: BF16 baseline, quant-disabled equivalence, minimum PTQ smoke, first-round direct quant eval, deploy artifact review, and dynamic-quant-qmm theoretical performance review.

## Structure

- Dense Qwen3 decoder, no MoE.
- Key config: `hidden_size=4096`, `intermediate_size=12288`, `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=128`.
- Reuses `QuantQwen3Attn` and `QuantGatedMLP`; no workflow, solver, or quantization module changes are required for this case.

## Adapter Validation

- BF16 baseline:
  - Wikitext PPL, `seq_len=4096`
  - `ppl = 8.946352005004883`
  - Output: `<eval: BF16 baseline (Wikitext local, seq4096)>`
- Quant-disabled equivalence:
  - Target: `attn-linear + mlp`
  - Config: `a_bits=16, w_bits=16`
  - `ppl = 8.946352005004883`
  - Output: `<eval: quant-disabled attn+mlp (seq4096)>`
  - Result: matches BF16 baseline.
- Minimum PTQ smoke:
  - Target: `block0/mlp`
  - Input cache: `<adapter cache: ptq-smoke mlp block0 input>`
  - Exported params: `<adapter exported params: ptq-smoke mlp block0>`
  - Result: at least one `mlp` unit can be enumerated, prepared with input/GT, and exported through the PTQ chain.

## First-Round Quantization

- Recommended scheme: `attn-linear + mlp`
- `quant_dtype`: `int`
- Bits: `a_bits=8, w_bits=8`
- `granularity`: `block`
- Algorithm: direct quant; no PTQ parameter directory is used.
- Eval: Wikitext PPL, `seq_len=4096`.
- Output: `<eval: W8A8 attn+mlp (seq4096)>`
- Result:
  - `ppl_bf16 = 8.946352005004883`
  - `ppl_quant = 8.86723518371582`
  - `delta = -0.079116821289063`
- Decision:
  - Passes the default `delta <= 0.2` threshold.
  - Quant PPL is slightly lower than BF16; this is treated only as acceptable, not as evidence that quantization improves quality.
  - Sanity check passed: quant-disabled equivalence is aligned, dataset and `seq_len=4096` match the current flow.

## Dynamic Quant + QMM Performance Review

- Skill: `skills/quant-tools/dynamic-quant-qmm-analysis`
- Scope: theoretical Linear `dynamic quant + qmm` benefit versus BF16 matmul only; this is not an end-to-end runtime measurement.
- Platform assumption: `A2/A3` bucket, `int_w8a8`, offline-prepared weights, dynamic activation quant.
- Shape source: `M=4096` from Wikitext `seq_len=4096`; `K/N` from model config and Linear weight shapes.

| Compare unit | Shape scope | Direction | Estimated speedup | Main bottleneck / risk |
| --- | --- | --- | --- | --- |
| MLP `up_proj/gate_proj` group | `M=4096,K=4096,N=12288` x2, shared activation quant | Positive | about `1.5x - 1.6x` | qmm cube / baseline MTE2 |
| MLP `down_proj` | `M=4096,K=12288,N=4096` | Positive | about `1.5x - 1.6x` | qmm cube / baseline MTE2 |
| Attention projection `q/k/v` group | q: `M=4096,K=4096,N=4096`; k/v: `M=4096,K=4096,N=1024`, shared activation quant | Positive | about `1.3x - 1.4x` | narrow `k/v` N remains a group-level risk |
| Attention projection `o_proj` | `M=4096,K=4096,N=4096` | Positive | about `1.3x - 1.4x` | qmm cube / baseline MTE2 |

Performance conclusion: the `attn-linear + mlp` W8A8 Int8 scheme is theoretically positive under the AMCT qmm estimator, with medium-high confidence. End-to-end runtime still requires downstream validation.

## Deploy Artifact

- Directory: `<deploy artifact: W8A8 attn+mlp>`
- Mode: direct quant
- Key files:
  - `config.json`
  - `model.safetensors.index.json`
  - 37 `*.safetensors` files
  - `deploy_quantization.md`
- `.exit_code = 0`
- Note: `deploy_quantization.md` records the downstream consumption contract; this casebook records the scheme decision, precision result, and qmm performance conclusion.

## Reuse Notes

- For another dense Qwen3 model, start from `qwen3.py`, `quant_module.py`, and `QuantGatedMLP`.
- Keep the current default `seq_len=4096` and the fixed Wikitext parquet path.
- A first pass can start from `attn-linear + mlp` W8A8 Int8 direct quant. If `delta <= 0.2` and qmm compare units are not negative as a group, proceed to deploy.


## QMM Runtime Validation

- Runner: `skills/quant-tools/dynamic-quant-qmm-analysis/validate_qmm.py`
- Profile output: `<perf data: QMM 8B/30B validation>`
- Consumed target: W8A8 `DynamicQuant + QuantBatchMatmul`; weight preparation is offline.

| Compare unit | BF16 baseline | DynamicQuant + QMM | Measured speedup | Decision |
| --- | ---: | ---: | ---: | --- |
| MLP `up/gate` group | 3520.589 us | 2211.306 us | 1.592x | Positive |
| MLP `down` | 1916.417 us | 1253.673 us | 1.529x | Positive |
| Attention `q/k/v` group | 990.955 us | 674.569 us | 1.469x | Positive |
| Attention `o` | 598.220 us | 453.544 us | 1.319x | Positive |

Runtime-validation conclusion: the measured AMCT qmm validation supports the W8A8 `attn-linear + mlp` scheme for the covered Linear subpaths. This is still not an end-to-end infer benchmark.

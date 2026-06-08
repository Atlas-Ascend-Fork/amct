# Qwen3-30B-A3B

## Summary

- This is the `qwen3_moe` adapter validation case under the current amct_llm flow with `seq_len=4096`.
- Model path: `/home/xyj/dev/models/Qwen/Qwen3-30B-A3B`
- Adapter: `amct_pytorch/common/models/llm/qwen/qwen3/qwen3_moe.py`
- Fixed Wikitext parquet: `/home/xyj/dev/datasets/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet`
- Completed: BF16 baseline, quant-disabled equivalence, MoE minimum PTQ smoke, first-round direct quant eval, deploy artifact review, and dynamic-quant-qmm theoretical performance review; and a full W4A16 weight-only INT4 path (GPTQ + group_size=64) with deploy artifact.

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


## W4A16 Quantization (weight-only INT4)

This case is the W4A16 weight-only INT4 path requested by the infer side for the
decode-latency deployment scenario (W8A8 was measured decode-neutral; see the infer
repo `topics/amct_w4a16_request_qwen3_30b.md`). Activations stay BF16; only Linear /
MoE-expert weights are quantized to INT4.

### Direct Quant Baseline

- Scheme: `attn-linear + moe`, `quant_dtype=int`, `a_bits=16, w_bits=4`, per-channel.
- `ppl_bf16 = 8.044384956359863`
- `ppl_quant = 8.859646797180176`, `delta = +0.815`
- Output: `<eval: W4A16 attn+moe direct (seq4096)>`
- Result: far above the `delta <= 0.2` threshold. INT4 per-channel direct quant is not viable.

### Error Localization

Single-target W4A16 per-channel direct quant:

| Target | `ppl_quant` | `delta` |
| --- | ---: | ---: |
| `attn-linear` only | 8.513859 | +0.469 |
| `moe` only | 8.291750 | +0.247 |

- `attn-linear` is the dominant contributor (~58% of the combined drop); `moe` second.
- Both subtargets individually exceed `0.2`; neither can be skipped. The drop is the
  systematic error of per-channel INT4 rounding, not a few sensitive layers.

### Algorithm Validation (attn-linear, Stage 1)

| Scheme | `ppl_quant` | `delta` | Note |
| --- | ---: | ---: | --- |
| direct per-channel | 8.513859 | +0.469 | baseline |
| AWQ per-channel | 8.601780 | +0.557 | worse than direct; AWQ dropped |
| GPTQ per-channel | 8.370209 | +0.326 | real gain `+0.144`, still over threshold |
| direct group-wise `g64` | 8.262518 | +0.218 | group-wise alone, still over threshold |
| **GPTQ + group-wise `g64`** | **8.042663** | **-0.0017** | essentially lossless |

- AWQ (the minimal in-repo implementation) degrades W4A16 and is not used.
- GPTQ alone and group-wise alone each fail the threshold; **both are required**.
- `GPTQ + group_size=64` makes attn-linear W4A16 essentially lossless.

### First-Round Quantization (final scheme)

- Recommended scheme: `attn-linear + moe`
- `quant_dtype`: `int`
- Bits: `a_bits=16, w_bits=4` (weight-only)
- `granularity`: `block`; weight `group_size=64`
- Algorithm: `gptq` (calibration on the fixed Wikitext parquet, `nsamples=128, seq_len=512`).
- Eval: Wikitext PPL, `seq_len=4096`.
- Output: `<eval: W4A16-g64 attn+moe GPTQ (seq4096)>`
- Result:
  - `ppl_bf16 = 8.044384956359863`
  - `ppl_quant = 8.132679939270020`
  - `delta = +0.088`
- Decision:
  - Passes the default `delta <= 0.2` threshold.
  - Per-subtarget: attn-linear GPTQ-g64 about `-0.002`, moe GPTQ-g64 about `+0.090`.
  - GPTQ params are computed once at calibration and persisted (`qweight` in the PTQ
    param file), so eval and deploy reuse them without recomputing the Cholesky path.

### Deploy Artifact

- Directory: `<deploy artifact: W4A16-g64 attn+moe>`
- Mode: GPTQ-calibrated W4A16
- Key files:
  - `config.json` with `quantization_config`: `compressed-tensors`, `format=int-quantized`,
    weights `{num_bits:4, group_size:64, strategy:"group", symmetric:true, type:"int"}`,
    `input_activations: null`.
  - `model.safetensors.index.json`
  - 49 `*.safetensors` files
  - `deploy_quantization.md`
- Weight tensor format: INT4 packed 8-per-`int32` (nibble `i` at bits `[4i,4i+4)`,
  stored unsigned `signed+8`), per-group `weight_scale` (fp32), zero `weight_offset`.
  This packing was round-trip verified bit-exact against the infer-side
  `CompressedTensorW4A16Int4MoEGMMMethod.unpack_from_int32`.
- Size: about `20 GB` vs BF16 `57 GB` (about `-65%` overall; quantized Linear/MoE
  weights about `-75%`).

### Known Issue

- The in-repo GPTQ originally fed single columns to the dtype quantizer, which
  degenerated the per-row scale to per-element (an identity transform) — GPTQ was not
  actually quantizing. Fixed: GPTQ now computes one fixed per-group scale from the full
  weight and applies it column-by-column with second-order compensation.
- AWQ `observe_input` accumulated CPU stats into an NPU buffer (device mismatch crash);
  fixed by aligning the reduction to the buffer device.
- `BaseSolver.finalize()` only collected trainable parameters, discarding the export of
  calibration-based algos (gptq/awq); fixed to also collect `export_ptq_params()`.

### Reuse Notes

- For another Qwen3 MoE model needing W4A16: use `int`, `w_bits=4, a_bits=16`,
  `group_size=64`, `algos=gptq`. Per-channel INT4 is not enough even with GPTQ; group-wise
  is mandatory.
- MoE GPTQ PTQ is heavy (128 experts x 48 layers); shard by block range across NPUs.
- The MoE performance decision is unchanged: W4A16 targets decode HBM savings; end-to-end
  decode latency still needs infer-side `MoEGMM` runtime validation.

## W4A16 MoE-only Re-export (2026-05-19)

The infer side updated the W4A16 request to **MoE-only** on 2026-05-19 (see the infer
repo `topics/amct_w4a16_request_qwen3_30b.md` §1.1): the framework `compressed_tensors.py`
Linear path has no wNa16 scheme (only MoE GMM does), and attention Linear is ~3% of
30B-A3B weight with ~0 decode benefit, so quantizing attention to INT4 is a net negative.
The earlier `attn-moe` W4A16 artifact was therefore re-exported as MoE-only.

### Deploy Artifact (MoE-only)

- 状态: 产物已于 2026-05-21 清理；以下为命令、配置与实测指标的归档记录（路径已脱敏）。
- Mode: PTQ (GPTQ-calibrated W4A16); reuses the `moe-w4a16-g64-gptq` PTQ params.
- Command: `python -m amct_pytorch.deploy --model_name qwen3_moe --quant_target moe --quant_dtype int
  --bit_config amct_pytorch/configs/w4a16-moe.yaml --algos gptq
  --moe_mlp_param_dir <moe-w4a16-g64-gptq ptq_params dir>`
- ⚠️ 待补特性（迁移注记）：本 W4A16 weight-only + group-wise INT4 路径来自 AMCT-Q `qwen3-adapter` 分支的**未合入提交**（`1324bea` group-wise INT4 + `54486c0` W4A16 weight-only deploy，~78 行 / 6 文件），AMCT-Q `main` 与 amct_llm 任何分支均**无此代码**。amct_llm `bit_policy` 可表达 `w4/a16`（顶层留 16、`moe` 块 `w_bits:4,a_bits:16`，即 `amct_pytorch/configs/w4a16-moe.yaml`），但 int4 分组打包与 W4A16 deploy 导出尚未移植，且 LLM CLI `--bit_config` 无 `group_size` 字段——**本案例在 amct_llm 当前不可直接跑通**。要启用需先把 `1324bea`+`54486c0` 移植/上游进 amct_llm（再随 amct_llm 回合 amct）。
- `--quant_target moe` only: `build_quant_block` wraps just `mlp.experts`; attention
  Linears stay float `nn.Linear` and are auto-collected into `quantization_config.ignore`
  (192 entries = 48 layers x q/k/v/o_proj), exported BF16 in `rest_00000.safetensors`.
- `--algos gptq` is **required** for deploy: the GPTQ PTQ param files carry an
  `algorithms.gptq` submodule; without `--algos gptq` the deploy block lacks that
  submodule and `load_module` raises `KeyError` on `up_proj.weight_quantizer.algorithms.gptq`.
- Key files: `config.json` (`compressed-tensors`, `format=int-quantized`, weights
  `{num_bits:4, group_size:64, strategy:"group", symmetric:true}`, `input_activations:null`,
  `ignore`=192), `model.safetensors.index.json` (55731 entries, 49 shards, 0 missing),
  48 `layer_*.safetensors` + 1 `rest_00000.safetensors`, `deploy_quantization.md`.
- Each MoE expert projection exports `weight` (int32, INT4 packed 8-per-int32),
  `weight_scale` (fp32, one per group of 64 input channels), `weight_offset` (fp32, zero).
- Size: ~20 GB (BF16 ~57 GB).
- Verification: round-trip dequant of layer0/expert0 gate_proj — unpacked INT4 in
  `[-7,7]`, offset all-zero, scale all-positive, cosine `0.9938` vs original BF16.
- Precision (measured, MoE-only quant_target=moe eval, Wikitext PPL seq_len=4096):
  `ppl_bf16 = 8.044385`, `ppl_quant = 8.178064`, `delta = +0.134`. Within the
  `delta <= 0.2` threshold; the MoE-only artifact passes.
  Output: `<eval: W4A16-g64 MoE-only GPTQ (seq4096)>`.
  Note: the measured `+0.134` is higher than a naive per-subtarget estimate
  (`combined +0.088` minus `attn-only -0.002` would suggest `~+0.090`). Block-wise
  quant error is non-linear across blocks, so per-subtarget deltas are not additive
  and must not be used to infer a single-target delta -- it needs a direct eval.

### `deploy_quantization.md` note

The in-code `_write_deploy_quantization_doc` (in `amct_pytorch/workflows/llm_deploy.py`) emits a
W8A8-templated document (`导出类型: direct-quant`, "W8A8 int direct quant", "int8",
"dynamic token activation"), which is inaccurate for a weight-only W4A16 / GPTQ /
MoE-only artifact. Per the `$deploy-export` skill, the document was regenerated by the
skill from `reference/deploy_quantization_template.md` to reflect the real export:
PTQ mode, INT4 int32-packed weights, no activation quantization, `moe -> MoEGMM` only,
attention in `ignore`. The deploy code's auto-doc remains the W8A8-flavored stub.

# DeepSeek V4.1 Deployment Weight Export

This example shows how to use AMCT to convert the official DeepSeek-V4.1-Flash checkpoint into deployable weights for the following targets:

| Target platform | Inference framework | Export format | Weight format |
| --- | --- | --- | --- |
| Atlas A3 | vLLM-Ascend | `ascend` | `int` / W8A8 dynamic quantization |
| Ascend 950 (950PR/950DT) | vLLM-Ascend | `ascend` | `hifp` / HiFloat8 W8A8 |

The adapter performs tensorwise checkpoint conversion only. It does not load the model or tokenizer and does not run PTQ.

## Prerequisites

- Obtain the official DeepSeek-V4.1-Flash weights and store them in a local Hugging Face safetensors model directory.
- Install AMCT and its dependencies as described in [Installation and Verification](../../../README_en.md#installation--verification).
- The Python, PyTorch, and safetensors environment must support `torch.float8_e8m0fnu` and its serialization.
- The output directory must be new or empty. It must not be the same as, a parent of, or a child of the source model directory.
- Run the following commands from the AMCT repository root.

## Atlas A3 (`int`) Deployment

The A3 `int` path targets vLLM-Ascend. It decodes MXFP/FP8/FP4 tensors in the official checkpoint to BF16, exports the modules selected by the configuration as INT8 W8A8 dynamic-quantized weights, and converts unselected source-quantized weights to BF16.

```shell
python3 amct_pytorch/cli/llm/deploy.py \
  --trust_remote_code \
  --model <MODEL_DIR> \
  --model_name deepseek_v4_1 \
  --device cpu \
  --granularity tensor \
  --quant_layers_config amct_pytorch/configs/deploy/deepseek/deepseek4_1/deepseek4_1_w8a8_int.json \
  --output_dir <A3_OUTPUT_DIR> \
  --deploy_platform A3
```

Key arguments:

- `<MODEL_DIR>`: directory containing the official DeepSeek-V4.1-Flash weights.
- `<A3_OUTPUT_DIR>`: output directory for the A3 deployment checkpoint.
- `--trust_remote_code`: allows custom code from the model repository to be loaded.
- `--model_name deepseek_v4_1`: selects the DeepSeek V4.1 model adapter.
- `--device cpu`: processes tensor deployment weights on the CPU.
- `--granularity tensor`: processes the checkpoint tensor by tensor without building the full model.
- `--quant_layers_config`: selects the layers to export as `int` W8A8 from the structured configuration.
- `--deploy_platform A3`: enables the DeepSeek V4.1 A3 conversion path.

The main output artifacts are:

- Converted safetensors weight shards.
- `model.safetensors.index.json`: the refreshed weight index.
- `config.json`: contains `quant_method=ascend` and `model_quant_type=W8A8_DYNAMIC`.
- `quant_model_description.json`: labels every output tensor as `W8A8_DYNAMIC` or `FLOAT`.
- `deployment_validation.json`: records integrity checks for weights, index mappings, dtypes, shapes, and quantization descriptions.

After export, verify that `status` in `deployment_validation.json` is `passed` before starting the model in a vLLM-Ascend A3 environment.

## Ascend 950 HiFloat8 Deployment

The HiFloat8 path targets vLLM-Ascend on Ascend 950. It encodes selected Linear/MoE weights as HiFloat8 W8A8, accepting floating-point source weights and FP8 weights with scales; unselected weights follow the FLOAT path.

```shell
python -m amct_pytorch.cli.llm.deploy \
  --model <MODEL_DIR> \
  --model_name deepseek_v4_1 \
  --granularity tensor \
  --deploy_platform ascend950 \
  --quant_layers_config amct_pytorch/configs/deploy/deepseek/deepseek4_1/deepseek4_1_w8a8_hifp.json \
  --output_dir <HIF8_OUTPUT_DIR>
```

Key arguments:

- `<MODEL_DIR>`: directory containing the official DeepSeek-V4.1-Flash weights.
- `<HIF8_OUTPUT_DIR>`: HiFloat8 checkpoint directory for vLLM-Ascend.
- `--model_name deepseek_v4_1`: selects the DeepSeek V4.1 model adapter.
- `--granularity tensor`: processes the checkpoint tensor by tensor without building the full model.
- `--deploy_platform ascend950`: enables the Ascend 950 HiFloat8 conversion path.
- `--quant_layers_config`: uses `quant_dtype=hifp`, `w_bits=8`, and `a_bits=8`; `pattern` selects Linear/MoE weights and `ignore_layers` excludes `mtp.*`.

After conversion, a selected layer has a `U8 [out, in]` weight and an `F32 [out, 1]` scale; `config.json` is copied from the source unchanged.

## Limitations

- DeepSeek V4.1 currently supports deployment conversion with `granularity=tensor` only. This adapter does not support PTQ.
- A3 supports `int` W8A8; Ascend 950 supports both `int` W8A8 and `hifp` (HiFloat8) W8A8.
- The quantization type and bit width are specified by the rules in `--quant_layers_config`.
- HiFloat8 selected weights must be non-empty two-dimensional tensors. BF16/F16/F32/F64 and scaled FP8 inputs are supported; encoding requires an NPU and `torch_npu` with native HiFloat8 cast support.
- None of these paths accepts `--algos`, `--attn_linear_param_dir`, `--attn_cache_param_dir`, or `--moe_mlp_param_dir`.

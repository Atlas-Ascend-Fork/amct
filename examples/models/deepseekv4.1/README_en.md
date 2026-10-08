# DeepSeek V4.1 Deployment Weight Export

This example shows how to use AMCT to convert the official DeepSeek-V4.1-Flash checkpoint into deployable weights for the following targets:

| Target platform | Inference framework | Export format | Weight format |
| --- | --- | --- | --- |
| Atlas A3 | vLLM-Ascend | `ascend` | W8A8 dynamic quantization |
| Atlas A5 (Ascend 950PR/950DT) | cann-recipes-infer | `legacy` | Preserve the source MXFP/FP8/FP4 bit widths |

The adapter performs tensorwise checkpoint conversion only. It does not load the model or tokenizer and does not run PTQ.

## Prerequisites

- Obtain the official DeepSeek-V4.1-Flash weights and store them in a local Hugging Face safetensors model directory.
- Install AMCT and its dependencies as described in [Installation and Verification](../../../README_en.md#installation--verification).
- The Python, PyTorch, and safetensors environment must support `torch.float8_e8m0fnu` and its serialization.
- The output directory must be new or empty. It must not be the same as, a parent of, or a child of the source model directory.
- Run the following commands from the AMCT repository root.

## Atlas A3 Deployment

The A3 path targets vLLM-Ascend. It decodes MXFP/FP8/FP4 tensors in the official checkpoint to BF16, exports the modules selected by the configuration as INT8 W8A8 dynamic-quantized weights, and converts unselected source-quantized weights to BF16.

```shell
python3 amct_pytorch/cli/llm/deploy.py \
  --trust_remote_code \
  --model <MODEL_DIR> \
  --model_name deepseek_v4_1 \
  --device cpu \
  --granularity tensor \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/deploy/tensor_w8a8.yaml \
  --deploy_format ascend \
  --quant_layers_config amct_pytorch/configs/deploy/deepseek_v4_1_ascend_w8a8.json \
  --output_dir <A3_OUTPUT_DIR> \
  --deploy_platform A3
```

Key arguments:

- `<MODEL_DIR>`: directory containing the official DeepSeek-V4.1-Flash weights.
- `<A3_OUTPUT_DIR>`: output directory for the A3 deployment checkpoint.
- `--granularity tensor`: processes the checkpoint tensor by tensor without building the full model.
- `--deploy_format ascend`: generates vLLM-Ascend ModelSlim-compatible artifacts.
- `--deploy_platform A3`: enables the DeepSeek V4.1 A3 conversion path.

The main output artifacts are:

- Converted safetensors weight shards.
- `model.safetensors.index.json`: the refreshed weight index.
- `config.json`: contains `quant_method=ascend` and `model_quant_type=W8A8_DYNAMIC`.
- `quant_model_description.json`: labels every output tensor as `W8A8_DYNAMIC` or `FLOAT`.
- `deployment_validation.json`: records integrity checks for weights, index mappings, dtypes, shapes, and quantization descriptions.

After export, verify that `status` in `deployment_validation.json` is `passed` before starting the model in a vLLM-Ascend A3 environment.

## Atlas A5 (Ascend 950) Deployment

The A5 path targets the cann-recipes-infer format for Ascend 950PR/950DT. It does not dequantize or requantize the weights. Instead, it preserves the source weight bit widths and bytes, and converts eligible 32x32 FP8 scales to the layout expected by the infer repository.

```shell
python -m amct_pytorch.cli.llm.deploy \
  --model <MODEL_DIR> \
  --model_name deepseek_v4_1 \
  --granularity tensor \
  --deploy_format legacy \
  --deploy_platform ascend950 \
  --quant_dtype mxfp \
  --quant_layers_config amct_pytorch/configs/deploy/deepseek_v4_1_tensor_mxfp.json \
  --output_dir <A5_OUTPUT_DIR>
```

Key arguments:

- `<MODEL_DIR>`: directory containing the official DeepSeek-V4.1-Flash weights.
- `<A5_OUTPUT_DIR>`: A5 deployment checkpoint directory for cann-recipes-infer.
- `--deploy_format legacy`: generates the legacy configuration expected by the infer repository.
- `--deploy_platform ascend950`: enables the A5/Ascend 950 conversion path.
- `--quant_dtype mxfp`: preserves the source MXFP bit-width semantics.

During conversion, eligible scales are expanded from `[N/32, K/32]` to `[N, K/32]`. FP4 experts, already row-expanded Engram scales, BF16/FP32 tensors, and tensors that do not meet the conversion conditions remain unchanged. The workflow also refreshes `model.safetensors.index.json` and `config.json`. The resulting directory can be consumed by cann-recipes-infer on Ascend 950PR/950DT.

## Limitations

- DeepSeek V4.1 currently supports deployment conversion with `granularity=tensor` only. This adapter does not support PTQ.
- The platform and format combinations are fixed:
  - A3: `--deploy_platform A3 --deploy_format ascend --quant_dtype int`.
  - A5: `--deploy_platform ascend950 --deploy_format legacy --quant_dtype mxfp`.
- The A5 path preserves the source bit widths and does not accept `--bit_config`.
- Neither path accepts `--algos`, `--attn_linear_param_dir`, `--attn_cache_param_dir`, or `--moe_mlp_param_dir`.
- The A5 configuration must cover every eligible 32x32 FP8 layer in the checkpoint. This prevents partial scale conversion from being combined with a global MXFP configuration.

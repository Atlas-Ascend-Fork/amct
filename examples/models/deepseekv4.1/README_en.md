# DeepSeek V4.1 Deployment Weight Export

This example shows how to use AMCT to convert the official DeepSeek-V4.1-Flash checkpoint into deployable weights for the following targets:

| Target platform | Inference framework | Export format | Weight format |
| --- | --- | --- | --- |
| Atlas A3 | vLLM-Ascend | `ascend` | W8A8 dynamic quantization |
| Atlas A5 (Ascend 950PR/950DT) | cann-recipes-infer | `legacy` | Preserve the source MXFP/FP8/FP4 bit widths |
| Atlas A5 (Ascend 950PR/950DT) | vLLM-Ascend | `legacy` | Attention Linears re-encoded to HiFloat8 |

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

## Atlas A5 (Ascend 950) HiFloat8 Deployment

The HiFloat8 path targets vLLM-Ascend on Atlas A5. It dequantizes the selected attention Linears from 32x32 block FP8 and re-encodes them to HiFloat8, stored as uint8 with one float32 scale per output channel. Tensors that are not selected are copied byte for byte, so the MXFP4 experts, the MXFP8 shared experts and the Engram table are untouched.

```shell
python -m amct_pytorch.cli.llm.deploy \
  --model <MODEL_DIR> \
  --model_name deepseek_v4_1 \
  --granularity tensor \
  --deploy_format legacy \
  --deploy_platform ascend950 \
  --quant_dtype hifp \
  --quant_layers_config amct_pytorch/configs/deploy/deepseek_v4_1_hif8.json \
  --output_dir <HIF8_OUTPUT_DIR>
```

Key arguments:

- `<HIF8_OUTPUT_DIR>`: HiFloat8 checkpoint directory for vLLM-Ascend.
- `--quant_dtype hifp`: enables the HiFloat8 re-encoding, alongside the bit-width preserving `mxfp` path.
- `--quant_layers_config`: selects `wq_a`/`wkv`/`wq_b`/`wo_a`/`wo_b` under attn, `wq_b` under the indexer, `wkv` under engram, and the same Linears in the MTP blocks.

For a selected layer the weight changes from `F8_E4M3 [out, in]` to `U8 [out, in]` with the shape unchanged, and the scale from `F8_E8M0 [out/32, in/32]` to `F32 [out, 1]`. The row scale is `max|row| / 16`, the same weight scope the HiFloat8 Linear uses. `config.json` keeps the source `quantization_config` and adds `hif8_weight_range`, `hif8_o_groups` and `hif8_o_lora_rank`.

`attn.wo_a` is block diagonal. Its rows are regrouped before quantization so that one rank row shares a scale across every group, which is what `npu_transpose_quant_batchmatmul` requires; the codes are still written in the flat 2-D layout and regrouped at load time.

## Limitations

- DeepSeek V4.1 currently supports deployment conversion with `granularity=tensor` only. This adapter does not support PTQ.
- The platform and format combinations are fixed:
  - A3: `--deploy_platform A3 --deploy_format ascend --quant_dtype int`.
  - A5: `--deploy_platform ascend950 --deploy_format legacy --quant_dtype mxfp`.
  - A5 HiFloat8: `--deploy_platform ascend950 --deploy_format legacy --quant_dtype hifp`.
- The A5 `mxfp` path preserves the source bit widths and does not accept `--bit_config`.
- Every layer selected by the HiFloat8 path must be a 32x32 block FP8 weight; no other layer is re-encoded.
- Neither path accepts `--algos`, `--attn_linear_param_dir`, `--attn_cache_param_dir`, or `--moe_mlp_param_dir`.
- The A5 configuration must cover every eligible 32x32 FP8 layer in the checkpoint. This prevents partial scale conversion from being combined with a global MXFP configuration.

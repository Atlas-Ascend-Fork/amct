# AMCT PyTorch LLM Quantization Tool Usage Guide

<!-- md-trans-meta sourceCommit=d6d6703a20d913bc6cd256ccf1d000139f5c7c3f translatedAt=2026-10-16T07:51:47.000Z pushedAt=2026-11-05T01:12:03.398Z nl=2 -->

This document describes the quantization tool for large language models (LLMs) in `amct_pytorch`. Centered on a **command-line workflow**, the tool connects key capabilities such as PPL (Perplexity) measurement, PTQ (Post-Training Quantization) data extraction, PTQ parameter training, and quantized weight deployment and export. Meanwhile, with the help of **model adapters**, the **quantization algorithm registry**, and the **quantization data type registry**, it enables flexible combinations of different model architectures and quantization strategies.

## Overview

### Capability Overview

The quantization tool mainly consists of the following modules:

| Module | Path | Description |
| ------ | ------ | ------ |
| CLI entry | `amct_pytorch/cli/llm/` | Provides four command-line entries: `eval`, `extract_ptq_data`, `ptq`, and `deploy`. |
| Workflow | `amct_pytorch/workflows/` | Supports orchestrating PPL evaluation, calibration data extraction, PTQ optimization training, and deployment export. |
| Model adapter | `amct_pytorch/common/models/llm/` | Adapts model structures such as DeepSeek, Qwen, LongCat, Pangu, and GLM. |
| Quantization algorithm | `amct_pytorch/algorithms/quant/` | Provides algorithm components such as LWC, LAC, LET, and AutoRound. (LWC+LET constitutes the complete omniquant algorithm.) |
| Quantization data type | `amct_pytorch/quantization/dtypes/` | Provides quantization/dequantization implementations such as `mxfp` and `int`. |
| Quantization configuration | `amct_pytorch/configs/` | Provides example configurations for BF16, W8A8, W4A4, and W4A8. |

Supported typical capabilities:

- **Benchmark evaluation**: Supports computing the perplexity (PPL) of BF16 or quantized models on the WikiText dataset.
- **Data extraction**: Uses Pileval data to efficiently extract the input activations and intermediate layer results required for PTQ calibration.
- **Block-level optimization**: Performs PTQ parameter training and optimization on specified quantization targets based on block granularity.
- **Model export**: Outputs `safetensors` weight files adapted to the deployment environment and the corresponding quantization configuration information.
- **Flexible configuration**: Flexibly sets global or grouped weight/activation (W/A) quantization bit widths through `bit_config`.
- **Algorithm selection**: Freely combines and selects trainable quantization algorithm strategies through the `algos` parameter.

### Environment Setup

Ensure that the model directory is in HuggingFace/safetensors format and contains `config.json`, tokenizer-related files, and `model.safetensors.index.json`. The supported model adapter names and their corresponding large models are as follows:

| Model Name (`--model_name`) | Description |
| ------ | ------ |
| [`deepseek_v3_2`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main) | DeepSeek V3.2 |
| [`deepseek_v4`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main) | DeepSeek V4 |
| [`qwen3`](https://huggingface.co/Qwen/Qwen3-4B/tree/main) | Qwen3 Dense |
| [`qwen3_moe`](https://huggingface.co/Qwen/Qwen3-235B-A22B/tree/main) | Qwen3 MoE |
| [`qwen3_5`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/main) | Qwen3.5 Dense |
| [`qwen3_5_moe`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/main) | Qwen3.5 MoE |
| [`qwen3_6_moe`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/main) | Qwen3.6 MoE |
| [`qwen3_next`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/tree/main) | Qwen3 Next |
| [`longcat_lite`](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite/tree/main) | LongCat Flash Lite |
| [`longcat_next`](https://huggingface.co/meituan-longcat/LongCat-Next/tree/main) | LongCat Next |
| [`glm5`](https://huggingface.co/zai-org/GLM-5.1/tree/main) | GLM-5.1 |

## CLI Entries and Corresponding Instantiation Operations

The implementations of the four entries described in this section are located in the `amct_pytorch/cli/llm/` directory. The package top level also provides wrapper modules with the same names (such as amct_pytorch/eval.py), and you can start them through `python -m amct_pytorch.eval`. Both ultimately run the same workflow. The bash examples in this document uniformly use the top-level module form.

### `eval`

**Entry**: `amct_pytorch/cli/llm/eval.py`

**Functional features**: This entry is used to evaluate perplexity (PPL) and supports two evaluation modes:

- BF16 baseline performance evaluation: Loads the original model block and computes perplexity (PPL) at BF16 precision (`eval_mode=bf16`).
- Quantization performance evaluation: Builds the quantized block, enables the quantizer based on `bit_config` and `quant_target`, and computes the quantized perplexity (PPL) (`eval_mode=quant`).

**Output**: An evaluation log containing the PPL is generated in the `output_dir/logs/` directory.

**Constraints**:

- Considering the limitations of the user environment, select `block` for `granularity` whenever possible.
- Configure `bit_config` and `model` with local paths. For the yaml file corresponding to `bit_config`, refer to the files in `amct_pytorch/configs` in the repository.

#### BF16 Baseline PPL Measurement

First, calculate the perplexity (PPL) of the BF16 full-precision model, which serves as the evaluation baseline for the accuracy after quantization:

```bash
python -m amct_pytorch.eval \
  --trust_remote_code \
  --model /path/to/model \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml
```

For parameter details, see [Parameters](#parameters).

#### Quantized Model PPL Measurement

Quantization evaluation rebuilds the quantization block and determines whether to enable the quantizer based on `bit_config`.

```bash
python -m amct_pytorch.eval \
  --trust_remote_code \
  --model /path/to/model \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --eval_mode quant \
  --quant_target mlp \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/example_w4a8.yaml
```

For parameter details, see [Parameters](#parameters).

#### PPL Test of Quantized Model with PTQ Training Results

Quantization evaluation rebuilds the quantized block and determines whether to enable the quantizer based on `bit_config`.

```bash
python -m amct_pytorch.eval \
  --trust_remote_code \
  --model /path/to/model \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --eval_mode quant \
  --quant_target mlp \
  --algos lwc lac \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/example_w4a8.yaml \
  --moe_mlp_param_dir ./outputs/qwen3_ptq/ptq_params/qwen3/mlp
```

For parameter details, see [Parameters](#parameters).

For `eval`-related startup commands, see [eval.sh](../../examples/eval.sh).

### `extract_ptq_data`

**Entry**: `amct_pytorch/cli/llm/extract_ptq_data.py`

**Functional features**: This entry is used to extract PTQ calibration data. The process is as follows:

1. Load calibration data: Load the Pileval calibration samples.
2. Perform inference: Perform Embedding encoding and block-by-block forward propagation.
3. Collect intermediate data: Capture the intermediate data required by PTQ, such as inputs and keyword arguments (kwargs), at the specified hook positions for use in the subsequent PTQ process.

**Output**: PTQ data such as block/unit inputs and kwargs under `--data_dir`.

**Constraints**:

- The `quant_target` parameter must specify exactly one target.
- `granularity` must be consistent with that in ptq.

#### PTQ Data Extraction

In the Block-based PTQ data extraction process, the system automatically locates the target layers based on the `quant_target` configuration and registers hooks to capture activation values or weight data. The data collection positions for different quantization targets are as follows:

- Attention module (`attn-linear`, `attn-cache`): Data is collected after the **Attention Norm**.
- FFN/MoE module (`mlp`, `moe`): Data is collected after the **FFN Norm**.

```bash
python -m amct_pytorch.extract_ptq_data \
  --trust_remote_code \
  --model /path/to/model \
  --model_name qwen3 \
  --device npu:0 \
  --quant_target mlp \
  --seq_len 4096 \
  --data_dir ./outputs/qwen3_ptq_data
```

For parameter details, see [Parameters](#parameters).

For startup commands related to `extract_ptq_data`, see [extract_ptq_data.sh](../../examples/extract_ptq_data.sh).

### `ptq`

**Entry**: `amct_pytorch/cli/llm/ptq.py`

**Functional features**: This entry is used to perform PTQ optimization training. The process is as follows:

1. Data loading: Loads the calibration dataset generated by `extract_ptq_data`.
2. Layer-by-layer optimization: Performs layer-by-layer quantization optimization on the specified `quant_target` unit.
3. Loss minimization: Uses the output of the original unquantized model as the ground truth (GT), and optimizes the quantization parameters by minimizing the mean squared error (MSE) between the quantized module output and the ground truth.
4. Parameter export: After optimization is complete, automatically exports the PTQ parameters of each unit.

**Output**: The `layer_*.pt` files in the PTQ parameter directory.

**Constraints**:

- The `quant_target` parameter must specify exactly one quantization target unit.
- Currently, only `granularity=block` is supported. The PTQ logic at the `model` granularity is currently a reserved API.
- `granularity` must remain consistent with that in **extract_ptq_data**.

#### Training PTQ Parameters

`PTQ` currently requires processing only one `quant_target` quantization target at a time, and supports block-granularity quantization training. After training is complete, the PTQ quantization parameters are saved to the corresponding parameter directory.

```bash
python -m amct_pytorch.ptq \
  --trust_remote_code \
  --model /path/to/model \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --quant_target mlp \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/example_w4a8.yaml \
  --algos lwc lac \
  --data_dir ./outputs/qwen3_ptq_data \
  --start_block_idx 0 \
  --end_block_idx 32 \
  --epochs 15 \
  --base_lr 1e-5 \
  --optimizer adamw \
  --output_dir ./outputs/qwen3_ptq
```

For parameter details, see [Parameters](#parameters).

### `deploy`

**Entry**: `amct_pytorch/cli/llm/deploy.py`

**Functional features**: This entry is used to export deployment weights. It supports blockwise quantization export and tensorwise weight conversion. The process is as follows:

- Load the base safetensors file: The system loads the original safetensors weight index from the source file as the basis for subsequent operations.
- Build the quantization module: Based on the specified PTQ parameters, the system builds the block module used for quantization processing.
- Export deployment tensors: The system generates and exports the quantized weight tensors, as well as related scale parameters and other tensor data used for deployment.
- Manage unreplaced weights: The original weights that are not processed during quantization are re-sharded and stored in the `rest_*.safetensors` file for recovery or further use.
- Update the configuration files: After the operation is complete, the system updates the core configuration files `model.safetensors.index.json` and `config.json` to ensure file integrity and consistency.

**Output**: Deployment model directory. The blockwise export includes `layer_*.safetensors`, `rest_*.safetensors`, the updated `model.safetensors.index.json`, and `config.json`; the tensorwise export preserves the source weight sharding layout and refreshes the weight index. When `deploy_format=ascend`, `quant_model_description.json` and `deployment_validation.json` are also generated.

**Constraints**:

- Module granularity: blockwise quantization export supports only `granularity=block`; `granularity=tensor` converts weights at tensor granularity (for example, FP8/FP4 → BF16 weight conversion, as shown in the DeepSeek-V4-Flash example). The two paths are mutually exclusive.
- Custom hook constraint: If a custom `quantize()` hook (used for the weight quantization algorithm) is used, the current version does not support deployment through the `export_deploy()` function.

For the single-device startup command of `ptq`, see [ptq_single_npu.sh](../../examples/ptq_single_npu.sh).
For the multi-device startup command of `ptq`, see [ptq_multi_npu.sh](../../examples/ptq_multi_npu.sh).

#### Export Deployment Weights

`deploy` currently supports block-granularity export. This process copies the model's non-weight auxiliary files, exports the quantized weights layer by layer, rewrites `model.safetensors.index.json`, and writes `quantization_config` into `config.json`.

```bash
python -m amct_pytorch.deploy \
  --trust_remote_code \
  --model /path/to/model \
  --model_name qwen3 \
  --granularity block \
  --quant_target mlp \
  --quant_dtype mxfp \
  --bit_config amct_pytorch/configs/example_w4a8.yaml \
  --moe_mlp_param_dir ./outputs/qwen3_ptq/ptq_params/qwen3/mlp \
  --output_dir ./outputs/qwen3_deploy
```

For parameter details, see [Parameters](#parameters).

#### Tensorwise Deployment Export

`granularity=tensor` supports the following two workflows:

- Pure dequantization: Converts source weights such as FP8/FP4 to BF16. It does not require `quant_target`, `bit_config`, or `quant_layers_config`; the specific capability is provided by the model adapter.
- Direct quantization: Explicitly selects the checkpoint modules to be quantized and ignored through `quant_layers_config`, and specifies the bit-width strategy through `bit_config`. The current general path supports INT W8A8.

`quant_layers_config` is a JSON file. The module name uses the complete path in the checkpoint without the final parameter suffix, and supports case-sensitive glob:

```json
{
  "quant_layers": ["model.layers.*.self_attn.o_proj"],
  "ignore_layers": ["model.layers.2.*"]
}
```

`deploy_format=legacy` outputs the existing AMCT deployment format; `deploy_format=ascend` outputs vLLM-Ascend ModelSlim-compatible W8A8 dynamic quantization weights and description files. For the A3/Ascend 950 input requirements, complete commands, and artifact descriptions of DeepSeek V4.1, see [DeepSeek V4.1 Deployment Instructions](../../examples/models/deepseekv4.1/README.md).

## Parameters

### Common Parameters

| Parameter | Default Value | Description |
| ------ | ------ | ------ |
| `--model` | `deepseek-ai/DeepSeek-V4-Flash` | Model weight path or model identifier. During deployment export, this should be a local model directory. |
| `--model_name` | **Required** | Name of the AMCT internal model adapter, which must match a registered model, such as `qwen3` or `deepseek_v4`. |
| `--trust_remote_code` | Disabled by default | Whether to allow the HuggingFace model repository to execute custom code. Enabled when this switch is passed. Models that depend on local model implementations, such as DeepSeek V3.2 and DeepSeek V4, must enable it; a Warning is given for other models when it is not enabled. |
| `--device` | `npu:0` | Specifies the running device. NPU/GPU is recommended to accelerate computation. |
| `--granularity` | `block` | Working granularity, that is, the unit of quantization or processing. Valid values are checked per subcommand: <br> `block`: Process by module block (finer granularity, potentially better results but more time-consuming). This is the default value for all subcommands. <br> `model`: Process the entire model uniformly (forward after loading the whole model, coarse granularity, fast). Supported only by `eval`; it is a reserved API in `ptq`. <br> `tensor`: Convert weights at tensor granularity (used for dequantization or direct quantization). Supported only by `deploy`. |
| `--seed` | `42` | Random seed used to control the consistency of random behavior in experiments (such as data sampling and initialization). Setting a fixed value ensures reproducible results. |
| `--quant_target` | `[]` (**Required** for `ptq` / `extract_ptq_data`; **Required** for blockwise export of `deploy`) | Quantization target. Values: mlp (multilayer perceptron) / moe (mixture of experts) / attn-linear (attention linear layer) / attn-cache (attention cache). Usage differs by command: `eval` and `deploy` can specify multiple targets at once (for example, `--quant_target mlp attn-linear`); `ptq` / `extract_ptq_data` allow only one target at a time, so execute multiple targets sequentially. In `deploy` blockwise, this parameter determines which quantization modules are built and exported; when missing, unquantized weights are silently exported. When `granularity=tensor`, pure dequantization (such as FP8/FP4 → BF16 weight conversion) does not use this parameter, while tensorwise direct quantization (for some model families, such as hy_v3 / glm5_2) still requires it. |
| `--seq_len` | `4096` | Input sequence length used during calibration and evaluation. It determines the model's performance when processing text of a certain length (especially important in the quantization calibration phase). Optional values: `1024`, `2048`, `4096`. |
| `--data_dir` | Empty string (**Required** for `ptq` / `extract_ptq_data`) | Directory for saving/reading PTQ intermediate data: the output directory of `extract_ptq_data`, from which `ptq` reads. |
| `--output_dir` | `./outputs` | Output directory for storing all generated content, such as log files, PTQ parameter configuration, and the final deployed model. |

### Deployment Parameters

The following parameters are used only for the `deploy` subcommand:

| Parameter | Default Value | Description |
| ------ | ------ | ------ |
| `--deploy_platform` | `None` | Target deployment hardware. Required for DeepSeek V4.1, supporting `A3` and `ascend950`; for other models, whether it is used is determined by the corresponding adapter. |
| `--deploy_format` | `legacy` | Deployment artifact format. `legacy` uses the existing AMCT format; `ascend` generates W8A8 dynamic quantization artifacts compatible with vLLM-Ascend ModelSlim. |
| `--quant_layers_config` | `None` | Path to the JSON configuration for Tensorwise direct quantization, which selects checkpoint modules through the `quant_layers` and `ignore_layers` glob patterns; an explicitly selected path overrides `quant_target`. |

### PPL Evaluation Parameters

| Parameter | Default Value | Description |
| ------ | ------ | ------ |
| `--eval_mode` | `bf16` | Evaluation mode, which specifies the precision format used by the model when computing perplexity (PPL). The following two formats are supported:<br> `bf16`: Half-precision floating-point number (BF16), commonly used to accelerate inference while maintaining high precision.<br>`quant`: Quantization mode, which may refer to quantization formats such as INT8/INT4, used to reduce memory and computation overhead, but may sacrifice a small amount of precision. |
| `--wikitext_final_out` | empty string | **[Reserved parameter, not effective temporarily]** WikiText final output path parameter, used to specify the file path of the final output after the WikiText dataset is processed. In the current version, the main pipeline (`eval` / `extract_ptq_data` / `ptq` / `deploy`) does not read this parameter yet, so the passed value does not take effect and only a warning is printed. The parameter itself is reserved and will be connected on demand in a later version. |

Note:

- When `eval_mode=bf16`, `bit_config` should not contain any linear/cache configuration lower than 16 bit.
- When `eval_mode=quant`, if `bit_config` has no configuration lower than 16 bit, a quantization module is built but the quantizer is disabled.

### Data Extraction Parameters

| Parameter | Default Value | Description |
|------|------|------|
| `--nsamples` | `128` | Number of calibration samples. `extract_ptq_data` loads this number of samples from Pileval. |

### Quantization Configuration Parameters

| Parameter | Default Value | Description |
| ------ | ------ | ------ |
| `--quant_dtype` | Empty string (**required** for blockwise export in `deploy`; must be explicitly specified when `eval_mode=quant`) | Data type after quantization, with values `int` / `mxfp` / `hifp`. For `deploy` blockwise export, it must be explicitly specified to ensure that the quantization declaration in `config.json` is consistent with the PTQ artifacts. When `eval_mode=quant`, the quantization module is built by selecting the quantization data type implementation based on this parameter; if it is missing, an error is reported at runtime. When `granularity=tensor`, pure dequantization does not use this parameter, but tensorwise direct quantization (for some model families) still requires it (currently supporting `int` / `mxfp`). |
| `--bit_config` | `None` | Path to a YAML-format configuration file that defines the specific bit-width strategy. If this parameter is not provided, the default `W16A16` configuration is used. |
| `--algos` | `[]` | List of enabled quantization algorithms. Each algorithm is automatically applied to the model's weights, activations, or structure based on its registered `target`. |
| `--is_per_tensor` | `False` | For the LAC (Learned Activation Clipping) algorithm, whether to use per-tensor statistics to determine the clipping range. |
| `--k_size` | `128` | **[Reserved parameter, not effective temporarily]** Size of the learnable Hadamard-type structural transformation matrix. In the current version, this size is determined by each model's `quant_module` (fixed at 128). This parameter is not yet connected, so the passed value does not take effect and only a warning is printed. The parameter itself is reserved and will be connected on demand in a later version. |

Common `bit_config`:

| File | Description |
| ------ | ------ |
| [bf16.yaml](../../amct_pytorch/configs/bf16.yaml) | All layers remain at 16 bit. |
| [w8a8.yaml](../../amct_pytorch/configs/w8a8.yaml) | Global W8A8 (both weights and activations are 8 bit). |
| [w4a8.yaml](../../amct_pytorch/configs/w4a8.yaml) | Global W4A8 (weights 4 bit, activations 8 bit). |
| [w4a4.yaml](../../amct_pytorch/configs/w4a4.yaml) | Global W4A4 (both weights and activations are 4 bit). |
| [example_w4a8.yaml](../../amct_pytorch/configs/example_w4a8.yaml) | Example of a custom quantization scheme. |

- `bit_config` supports top-level `w_bits` and `a_bits`, and also supports group overrides by `attn-linear`, `mlp`, `moe`, and `attn-cache`. Within the linear group, if `w_bits` or `a_bits` is written, both must appear together.
- When `quant_dtype` is passed as `int`, weights use `per-channel` quantization and activations use `dynamic-per-token` quantization, both adopting **symmetric quantization**.
- `bit_config` requires a YAML file. It is recommended that users configure the YAML locally. Refer to the [YAML examples under the configs path](../../amct_pytorch/configs/) or click the symlinks above to navigate directly.

### PTQ Parameters

| Parameter | Default Value | Description |
| ------ | ------ | ------ |
| `--cali_bsz` | `4` | Batch size for PTQ calibration training.<br>During PTQ, a small portion of data (called the calibration dataset) is used to adjust quantization parameters (such as scaling factors and zero points). This parameter defines how many samples are processed at a time for calibration. |
| `--base_lr` | `1e-5` | Base learning rate. This parameter determines the step size by which model parameters are updated in each iteration. |
| `--optimizer` | `adamw` | Optimizer, which specifies the algorithm used to update model parameters. The following optimizers are supported.<br> `adamw`: An improved version of Adam, which generally handles weight decay better.<br>`adam`: The classic Adam optimizer.<br/>`sgd`: Stochastic gradient descent.<br/>`signsgd`: A variant of stochastic gradient descent based on the sign of the gradient. |
| `--weight_decay` | `0.0` | Weight decay, applicable to some optimizers. It is a regularization technique that prevents model overfitting by adding a penalty term proportional to the weight magnitude to the loss function. A value of `0.0` means that weight decay is not used in this optimization process. |
| `--momentum` | `0.9` | SGD momentum.<br>When the SGD optimizer is used, momentum helps accelerate convergence and helps the model escape local optima. It updates parameters by accumulating the exponentially weighted moving average of previous gradients. |
| `--lr_scheduler` | `cosine` | Learning rate scheduler, which defines how the learning rate is dynamically adjusted during training. The following schedulers are supported:<br> `none`: The learning rate remains unchanged.<br>`cosine`: Cosine annealing, in which the learning rate decreases smoothly along a cosine curve.<br/>`step`: Step decay, in which the learning rate is multiplied by a coefficient at fixed step intervals.<br/>`cosine` is used by default because it provides a smoother learning rate decrease, which helps the model converge to a better state. |
| `--min_lr` | `0.0` | **[Reserved parameter, not effective temporarily]** Minimum learning rate parameter, used to set the lower limit of the learning rate. In the current version, the general scheduler has not yet read this parameter (the `eta_min` of the `cosine` scheduler is always `base_lr * 1e-3`), so the passed value does not take effect and only a warning is printed. The parameter itself is reserved and will be connected on demand in a later version. |
| `--lr_step_size` | `1` | Step size of StepLR.<br>This parameter takes effect only when `--lr_scheduler` is set to `step`. It defines after how many epochs the learning rate is decayed once. |
| `--lr_gamma` | `0.1` | Decay coefficient of StepLR.<br>This parameter is effective only when `--lr_scheduler` is `step`. It defines the coefficient by which the learning rate is multiplied at each decay. |
| `--epochs` | `15` | Number of PTQ optimization epochs, which specifies the total number of rounds for calibration and optimization. One epoch means that the entire calibration dataset is traversed once. |
| `--start_block_idx` | `0` | Starting block index, inclusive of that layer.<br>This parameter specifies from which part (block) of the model the optimization starts. |
| `--end_block_idx` | `61` | Ending block index (specifies the last block to be optimized), exclusive of that layer. |
| `--attn_linear_param_dir` | empty string | Specifies the PTQ parameter directory of the linear layers in the attention mechanism. |
| `--attn_cache_param_dir` | empty string | Specifies the PTQ parameter directory of the cache part in the attention mechanism. |
| `--moe_mlp_param_dir` | empty string | Specifies the PTQ parameter directory of the mixture-of-experts (MoE) or multi-layer perceptron (MLP) part. |

If the corresponding parameter directory is not explicitly passed during `ptq`, the system automatically creates it in the following form:

```text
{output_dir}/ptq_params/{model_name}/{quant_target}/
```

The parameter file of each PTQ unit is usually named as follows:

```text
layer_{layer_idx}_{unit_name}.pt
```

## Common Precautions

- The `quant_target` of `extract_ptq_data` and `ptq` must be consistent.
- `ptq` and `extract_ptq_data` currently process only one `quant_target` at a time. When multi-target quantization is required, it is recommended to execute them target by target.
- When `deploy` reads the PTQ parameter directory, ensure that the parameters of the corresponding `quant_target` have been trained.
- The default value of `--granularity` is `block`. `ptq` / `extract_ptq_data` support only `block` (the `model` granularity is a reserved implementation), `deploy` additionally supports `tensor`, and `eval` additionally supports `model`.
- `--model_name` is the internal adapter name and is not necessarily the same as the HuggingFace model path.
- Configure local paths for `bit_config` and `model`. For the YAML file corresponding to `bit_config`, refer to the files in the `amct_pytorch/configs` directory in the repository.

## One-Stop Example

This section provides a one-stop example to help developers better understand the workflow of this feature. Taking the `qwen3.6moe` model as an example, see [Qwen-3.6-MoE One-Stop Example](../../examples/models/qwen3.6/Qwen3.6-Moe.md).

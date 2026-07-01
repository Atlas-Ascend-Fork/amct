# MiniMax-M2.5 SmoothQuant MXFP4 Sample

This sample provides a two-stage SmoothQuant + MXFP4 quantization workflow for MiniMax-M2.5:

1. `scripts/run_vllm_stage1.sh`: Launches MiniMax-M2.5 on the vLLM Ascend execution path and records the activation statistics required by SmoothQuant.
2. `scripts/run_stage2.sh`: Reads the stage1 activations, performs SmoothQuant fusion, and exports the model as MXFP4 HuggingFace safetensors.

## Directory Layout

- `scripts/`: Entry-point run scripts
- `src/`: Core implementations of stage1 / stage2
- `mxfp4_quantizer/`: MXFP4 quantization and packing logic
- `patches/`: Patches required to run MiniMax on Ascend vLLM

## Environment

Recommended container: `quay.io/ascend/vllm-ascend:v0.14.0rc1-a3`

The container already includes the following vLLM-related components (installed in editable mode under `/vllm-workspace/`):

| Component | Repository | Commit |
|-----------|------------|--------|
| vLLM | https://github.com/vllm-project/vllm.git | `d7de043d55d1dd629554467e23874097e1c48993` |
| vLLM-Ascend | https://github.com/vllm-project/vllm-ascend | `52d4acfa51fb868823d1070b81cbd2d97e9e4696` |

It is recommended to run inside the container with the following directories mounted in advance:

- Source code directory, e.g. mounted at `/workspace/amct`
- BF16 or FP8 MiniMax-M2.5 model directory, e.g. mounted at `/model/MiniMax/MiniMax-M2.5-bf16`
- Calibration data directory, e.g. mounted at `/data/minimax/data.jsonl`

The examples below assume you have already entered the repository directory:

```bash
cd /workspace/amct/amct_pytorch/experimental/quantization/MiniMax-M2.5
```

## How to Provide External Inputs

When running this sample, you need to provide the following external inputs yourself.

### `MODEL_DIR`

`MODEL_DIR` is the BF16 model directory of MiniMax-M2.5, shared by both stage1 and stage2.

- The directory must at least contain model weights and configuration files that can be loaded normally by `transformers` / `vllm`.
- It is recommended to mount the BF16 model directory directly.
- If you use an FP8 model, please make sure the current vLLM patch and runtime environment support the corresponding loading path.

Example:

```bash
export MODEL_DIR=/model/MiniMax/MiniMax-M2.5-bf16
```

### `CALIB_DATA`

`CALIB_DATA` is the calibration set used by stage1, in `jsonl` format. The script reads it line by line, extracts the `messages` field from each line, and then constructs inputs using the tokenizer's chat template.

Example of a single line:

```json
{"messages":[{"role":"user","content":"Introduce the basic principles of quantization."}]}
```

Example:

```bash
export CALIB_DATA=/data/minimax/data.jsonl
```

### `VLLM_REPO_DIR`

`VLLM_REPO_DIR` is the Ascend vLLM source directory; it is only required by stage1. `scripts/run_vllm_stage1.sh` will check and try to apply the MiniMax patch from this directory's `patches/` folder.

Example:

```bash
export VLLM_REPO_DIR=/vllm-workspace/vllm
```

### `VLLM_PATCH_PATH`

`VLLM_PATCH_PATH` points to the patch file under the current directory by default. You normally don't need to change it manually; only override it when you are using a different patch file.

Example:

```bash
export VLLM_PATCH_PATH=$(pwd)/patches/0001-MiniMax-M2-adapt-Ascend-fp8-loading-and-qk-norm-path.patch
```

## Stage1: Recording Activations

Stage1 launches MiniMax-M2.5 on the real vLLM Ascend execution path, attaches hooks to attention-related modules, collects the maximum activation values required by SmoothQuant, and persists the results to `RECORD_DIR`.

### Example Launch

```bash
MODEL_DIR=/model/MiniMax/MiniMax-M2.5-bf16 \
CALIB_DATA=/data/minimax/data.jsonl \
RECORD_DIR=$(pwd)/record_data_vllm \
NUM_CALIB_DATA=2048 \
SEQ_LEN=32768 \
TP_SIZE=16 \
GPU_MEMORY_UTILIZATION=0.9 \
VLLM_REPO_DIR=/vllm-workspace/vllm \
VLLM_PATCH_PATH=$(pwd)/patches/0001-MiniMax-M2-adapt-Ascend-fp8-loading-and-qk-norm-path.patch \
ENABLE_EXPERT_PARALLEL=1 \
VLLM_MAX_NUM_SEQS=32 \
VLLM_MAX_NUM_BATCHED_TOKENS=32768 \
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 \
VLLM_ENFORCE_EAGER=1 \
VLLM_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
bash scripts/run_vllm_stage1.sh
```

### Outputs

After stage1 finishes, per-layer activation statistic files will be generated under `RECORD_DIR`, for example:

- `layers_0_self_attn_q_proj.pt`
- `layers_0_self_attn_k_proj.pt`
- `layers_0_self_attn_v_proj.pt`
- `layers_0_self_attn_o_proj.pt`
- `metadata.json`

### Parameter Description

- `MODEL_DIR`: Original MiniMax-M2.5 model directory.
- `CALIB_DATA`: Path to the calibration `jsonl` file; each line must contain a `messages` field.
- `RECORD_DIR`: Output directory for stage1 activation statistics; stage2 reads the `.pt` files from here directly.
- `NUM_CALIB_DATA`: Number of calibration samples used for statistics.
- `SEQ_LEN`: Truncation length for calibration samples; also determines vLLM's maximum context length setting.
- `TP_SIZE`: Number of parallel devices launched by `torchrun`, corresponding to the tensor parallel size.
- `GPU_MEMORY_UTILIZATION`: Fraction of device memory vLLM is allowed to use.
- `VLLM_REPO_DIR`: vLLM source directory.
- `VLLM_PATCH_PATH`: Path to the MiniMax Ascend adaptation patch.
- `ENABLE_EXPERT_PARALLEL`: Whether to enable expert parallel; `1` to enable, `0` to disable.
- `VLLM_MAX_NUM_SEQS`: Maximum number of requests allowed per batch in vLLM.
- `VLLM_MAX_NUM_BATCHED_TOKENS`: Maximum number of tokens allowed per batch in vLLM.
- `VLLM_ASCEND_ENABLE_FLASHCOMM1`: Whether to enable the Ascend FlashComm1 capability.
- `VLLM_ENFORCE_EAGER`: Whether to enforce eager execution; `1` to enable, `0` to disable.
- `VLLM_COMPILATION_CONFIG`: Compilation configuration passed to vLLM, expressed as a JSON string.

## Stage2: SmoothQuant + MXFP4 Export

Stage2 no longer performs forward calibration. Instead, it reads the activation statistics from stage1, computes the SmoothQuant scale, fuses the scale into the relevant parameters, and then exports the quantizable weights in MXFP4 format.

### Example Launch

```bash
MODEL_DIR=/model/MiniMax/MiniMax-M2.5-bf16 \
RECORD_DIR=$(pwd)/record_data_vllm \
OUTPUT_DIR=$(pwd)/exported_model_vllm \
LOAD_DEVICE_MAP=auto \
ALPHA=0.8 \
bash scripts/run_stage2.sh
```

If you want to explicitly specify a device map file, you can launch it as follows:

```bash
MODEL_DIR=/model/MiniMax/MiniMax-M2.5-bf16 \
RECORD_DIR=$(pwd)/record_data_vllm \
OUTPUT_DIR=$(pwd)/exported_model_vllm \
LOAD_DEVICE_MAP=auto \
DEVICE_MAP_FILE=/path/to/device_map.json \
ALPHA=0.8 \
bash scripts/run_stage2.sh
```

### Outputs

After stage2 finishes, the HuggingFace safetensors export results will be generated under `OUTPUT_DIR`, including:

- `model-00001-of-xxxxx.safetensors`
- `model.safetensors.index.json`
- `config.json`
- `generation_config.json`
- Tokenizer-related files

### Parameter Description

- `MODEL_DIR`: MiniMax-M2.5 BF16 model directory.
- `RECORD_DIR`: Activation statistics directory output by stage1.
- `OUTPUT_DIR`: Export directory for the MXFP4-quantized model. It must be empty or not exist beforehand.
- `LOAD_DEVICE_MAP`: Model loading strategy; can be `auto` or `cpu`. `auto` distributes the model across devices automatically, while `cpu` loads to CPU first.
- `DEVICE_MAP_FILE`: Optional device map JSON file used to explicitly specify the mapping between submodules and devices.
- `ALPHA`: The balancing coefficient of SmoothQuant, with a value range of `[0, 1]`.

## Running Tips

- Run stage1 first and then stage2; stage2 depends on the activation statistics files produced by stage1.
- `SEQ_LEN`, `NUM_CALIB_DATA`, and `TP_SIZE` significantly affect the device memory consumption and runtime of stage1, and should be adjusted according to the actual number of devices and available memory.
- If stage1 reports a patch mismatch at startup, please verify that the vLLM version pointed to by `VLLM_REPO_DIR` is compatible with the patch under `patches/`.

## SmoothQuant Principle

The core idea of SmoothQuant is: through mathematically equivalent transformations, "smooth" the hard-to-quantize outliers in activations into the weights, so that both activations and weights become easier to quantize at low bit widths.

### Basic Formula

For a linear layer $Y = XW$, introduce a per-channel scaling factor $s$, and apply the equivalent transformation:

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$

That is, the activation is divided by $s$ and the weight is multiplied by $s$. Thanks to the associativity of linear operations, the final output remains unchanged.

The scaling factor $s$ is computed as:

$$s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}$$

where $\alpha \in [0, 1]$ controls how the "quantization difficulty" is distributed between activations and weights:
- A larger $\alpha$ shifts more quantization difficulty from activations to weights.
- A smaller $\alpha$ keeps the weights closer to their original values, leaving the activations to absorb more quantization error.

### How the Scale is Fused in Stage2

In inference deployment scenarios, we do not apply $\text{diag}(s)^{-1}$ scaling to activations at runtime (which would introduce extra operator overhead). Therefore, stage2 **fuses $s$ into the weights of the preceding module**, leveraging the natural connectivity of the network structure to eliminate runtime overhead:

#### Group 1: LayerNorm → QKV Projection

```
input_layernorm.weight  ÷=  s
q_proj.weight           *=  s     (along the input_channel dimension)
k_proj.weight           *=  s
v_proj.weight           *=  s
```

Since the output of LayerNorm is directly used as the input to QKV, `layernorm.weight /= s` is equivalent to applying `/= s` to the activation, while `*= s` on the QKV weights compensates for this scaling. Combined, they are mathematically equivalent to the original computation, but the dynamic range of the activations is smoothed.

In the computation of $s$, `act_scale` is taken from the per-channel maximum absolute value of the `q_proj` input activations recorded in stage1, while `weight_scale` is taken as the joint maximum, across the q/k/v weight matrices, of the per-input-channel maximum absolute values.

#### Group 2: V Projection → O Projection

```
v_proj.weight  ÷=  s     (along the output_channel dimension)
v_proj.bias    ÷=  s     (if present)
o_proj.weight  *=  s     (along the input_channel dimension)
```

The output of `v_proj` is the input of `o_proj` (after the attention computation), so dividing the output of `v_proj` by $s$ is equivalent to dividing the input of `o_proj` by $s$. In implementation, $s$ is fused into the output dimension of `v_proj.weight` and the input dimension of `o_proj.weight`.

For GQA (Grouped Query Attention), since the number of KV heads is smaller than the number of Q heads, $s$ must be reduced via max within each head group before being applied to `v_proj` and `o_proj`, ensuring that multiple Q heads corresponding to the same KV head share the same scaling factor.

### Summary

Through the two fusion groups above, stage2 achieves "zero-runtime-overhead" SmoothQuant: all scaling operations are folded into existing weights offline, with no extra operators required at inference time. The fused weights then go through MXFP4 quantization and packing, and are finally exported as safetensors that can be deployed directly.

## Quantization Accuracy Results

To evaluate the impact of the SmoothQuant + MXFP4 quantization scheme on model accuracy, we compared the BF16 baseline (Baseline) with the PTQ-quantized model (PTQ) on several mainstream benchmarks, and report the accuracy retention ratio of PTQ relative to Baseline (ratio = PTQ / Baseline).

| Task     | drop   | gpqa_diamond | gsm8k  | humaneval+ | math500 | lcb    | longbenchV2 |
|----------|--------|--------------|--------|------------|---------|--------|-------------|
| Baseline | 90.478 | 85.15        | 96.01  | 91.33      | 93.06   | 62.8   | 56.76       |
| PTQ      | 90.04  | 84.64        | 95.89  | 91.21      | 92.92   | 63.83  | 55.09       |
| Ratio    | 0.99   | 0.99         | 0.99   | 0.99       | 0.99    | 1.0    | 0.97        |

### Notes

- The accuracy evaluation above is performed with the inference framework [cann-recipes-infer / minimax_m2.5_mxfp4](https://gitcode.com/cann/cann-recipes-infer/tree/master/contrib/minimax_m2.5_mxfp4). This framework provides a **simulated implementation** of MXFP4-quantized MiniMax-M2.5 inference on Ascend NPUs (i.e., it follows MXFP4 quantization/dequantization rules through equivalent computation on high-precision operators, rather than truly invoking MXFP4 low-bit hardware operators), and is used to evaluate the accuracy of the quantization scheme in an end-to-end inference pipeline.
- As can be seen from the results, the PTQ-quantized model retains an accuracy ratio of **0.99** or higher on tasks such as drop, gpqa_diamond, gsm8k, humaneval+, and math500. On lcb, PTQ is slightly better than Baseline (ratio 1.0), and on longbenchV2 the ratio is **0.97**, indicating an overall controllable accuracy loss.
- Due to factors such as sampling temperature, randomness, and differences in inference backend implementations during evaluation, **some results may fluctuate**. When reproducing, the numbers under the same configuration may differ slightly from the table above. It is recommended to take the average of multiple runs or reproduce with the same framework version.

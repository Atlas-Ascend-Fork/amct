# AMCT Large Model FlatQuant Quantization

> **Official pipeline (Qwen3)**: `run_qwen_samples.py` has switched to the official
> quantization pipeline `quantize -> sequential_calibrate -> convert` (no
> `--experimental` build option required). `run_llama2_samples.py` remains an
> experimental-pipeline sample and requires `bash build.sh --torch --experimental`.

## 1 Quantization Prerequisites

### 1.1 Install Dependencies

The dependency packages for this sample can be found in [requirements.txt](requirements.txt)

Note that the TorchNPU package version needs to match the Python and torch package versions, and the CANN package needs to be installed

### 1.2 Model and Dataset Preparation

This sample uses Llama2-7b/Qwen3-0.6B and wikitext2 dataset as examples. Please download them yourself and pass the actual directory in the script.

### 1.3 Simple Quantization Configuration
The quantization configuration used in this sample is built into the tool and can be obtained and used in the following ways:

For the official pipeline (Qwen3), the built-in configurations can be obtained as follows:

`from amct_pytorch.common.config.config import INT8_INT4_FLAT_QUANT_CFG` (W4A8, fake-quant accuracy evaluation only)
`from amct_pytorch.common.config.config import INT4_INT4_FLAT_QUANT_CFG` (W4A4, full pipeline including deployment)

For the experimental pipeline (Llama2), use:

`from amct_pytorch.experimental.flatquant.config import INT4_FLAT_QUANT_CFG`

We have added a 'use_down_quant' configuration in the quantization configuration to control whether down_proj is quantized. For models sensitive to down_proj quantization, you can skip down_proj quantization.
If you need to modify the detailed configuration, please refer to the documentation to construct the required quantization configuration dict.

The FlatQuant algorithm supports the following partial quantization:
- True quantization: Weights and inputs of q_proj, k_proj, v_proj in self_attn and up_proj, gate_proj, down_proj in mlp are quantized together (using Kronecker product), where inputs are per token, weights are per channel, both using symmetric quantization
- Fake quantization: kv_cache and o_proj (currently recommend disabling, refer to `INT4_FLAT_QUANT_CFG`)

Supported quantization types and quantization configurations:

| Field | Type | Description | Value Range | Notes |
|:--| :-: | :-- | :-: | :-- |
|skip_layers|list[str] / set[str]|Layers to skip quantization |/|Fuzzy matching is applied to each string in the list or set. A layer is skipped when its name contains any configured string and no quantization configuration is generated. For example: {"lm_head"}.|
|algorithm|dict|Quantization algorithm configuration used|{'flatquant'}|Refer to the `INT8_INT4_FLAT_QUANT_CFG`/`INT4_INT4_FLAT_QUANT_CFG` examples.|

## 2 Quantization Example

### 2.1 Llama2 Quantization

**step 1.** Please execute the following command in the current directory to run the sample program and modify the model path in the sample program according to actual conditions:
```python
python3 src/run_llama2_samples.py --model_path <llama2 model path>
```

If the following information appears, it indicates that quantization is successful:
```none
INFO All done!
```

The following information in the log is the evaluation task result (percentage accuracy):
```
INFO ACC: {'arc_challenge': 42.83, 'arc_easy': 70.88, 'hellaswag': 73.63, 'lambada_openai': 72.0, 'piqa': 77.48, 'winogrande': 67.88, 'acc_avg': 67.45}
```
The following information is perplexity (wikitext, max length 512):
```
INFO PPL score: 5.870388984680176
```
The following information is the inference speed (ms) of the original model and the true quantization model:
```
INFO Time diff orig: 929.0580000000001 ms
INFO Time diff after real quant: 139.707 ms
```

After the script finishes running, calibration parameters `./outputs/llama2_7b/flat_matrices.pth` and quantization log file `./amct_log/amct_pytorch.log` will be generated and saved in the current directory. If you want to directly load calibration parameters, use the following settings:
```python
python3 src/run_llama2_samples.py --model_path <llama2 model path> --load_matrix --flat_matrix_path <matrix path, e.g. ./outputs/llama2_7b/flat_matrices.pth>
```

### 2.2 Qwen3 Quantization (Official Pipeline)

Official pipeline walkthrough: `quantize` builds the quantization-calibration model, then
`sequential_calibrate` performs layer-by-layer gradient calibration (the framework captures
inputs per block, and the FlatQuant gradient block calibrator trains each layer with AdamW).
The calibration artifact `flat_matrices.pth` can be saved/reused, and finally `convert`
solidifies the transforms into NPU deployment modules (W4A4 real quantization).

**step 1.** Please execute the following command in the current directory to run the sample program and modify the model path in the sample program according to actual conditions:
```python
python3 src/run_qwen_samples.py --model_path <qwen3-0.6b model path>
```
To reuse calibrated matrices and skip calibration:
```python
python3 src/run_qwen_samples.py --model_path <qwen3-0.6b model path> --load_matrix --flat_matrix_path ./outputs/qwen/flat_matrices.pth
```

If the following information appears, it indicates that quantization is successful:
```none
INFO All done!
```

The example shows different generation results based on prompt before and after model quantization:
The prompt is:
```
prompt = "Give me a short introduction to the Ascend Model Compression Toolkit(AMCT). /no_think"
```

The generation result before quantization is:
```
content:  ============================================================================
<>
The Ascend Model Compression Toolkit (AMCT) is a powerful tool designed to ...
```

The generation result after quantization is:
```
content:  ============================================================================
<>
The Ascend Model Compression Toolkit (AMCT) is a powerful tool designed to ...
```

> After quantization, to export the quantized model as a torchair offline model (.air) for ATC-to-om conversion and on-board inference, refer to [AMCT large-model offline torchair model export](../export/README_en.md).

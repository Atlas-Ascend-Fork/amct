# AMCT Large Model FlatQuant Quantization for LLAMA2/Qwen3

> **Note (Experimental feature)**: This sample depends on `amct_pytorch/experimental/flatquant/`.
> Build and install the package with `bash build.sh --torch --experimental` before running.

## 1 Quantization Prerequisites

### 1.1 Install Dependencies

The dependency packages for this sample can be found in [requirements.txt](requirements.txt)

Note that the torch_npu package version needs to match the Python and torch package versions, and the CANN package needs to be installed

### 1.2 Model and Dataset Preparation

This sample uses Llama2-7b/Qwen3-8b and wikitext2 dataset as examples. Please download them yourself and pass the actual directory in the script.

### 1.3 Simple Quantization Configuration
The quantization configuration used in this sample is built into the tool and can be obtained and used in the following ways:

`from amct_pytorch.experimental.flatquant.config import INT4_FLAT_QUANT_CFG`

We have added a 'use_down_quant' configuration in the quantization configuration to control whether down_proj is quantized. For models sensitive to down_proj quantization, you can skip down_proj quantization.
If you need to modify the detailed configuration, please refer to the documentation to construct the required quantization configuration dict.

The FlatQuant algorithm supports the following partial quantization:
- True quantization: Weights and inputs of q_proj, k_proj, v_proj in self_attn and up_proj, gate_proj, down_proj in mlp are quantized together (using Kronecker product), where inputs are per token, weights are per channel, both using symmetric quantization
- Fake quantization: kv_cache and o_proj (currently recommend disabling, refer to `INT4_FLAT_QUANT_CFG`)

Supported quantization types and quantization configurations:

| Field | Type | Description | Value Range | Notes |
|:--| :-: | :-- | :-: | :-: | :-- |
|skip_layers|str|Layers to skip quantization |/|Skip quantization layers support fuzzy matching. When the configured string is a layer name substring or matches the layer name, skip quantization for that layer and do not generate quantization configuration. The string must contain numbers or letters|
|algorithm|dict|Quantization algorithm configuration used|{'flatquant'}|Refer to `INT4_FLAT_QUANT_CFG` example

## 2 Quantization Example

### 2.1 Llama2 Quantization

**step 1.** Please execute the following command in the current directory to run the sample program and modify the model path in the sample program according to actual conditions:
```python
python3 src/run_llama2_samples.py --model_path <llama2 model path>
```

If the following information appears, it indicates that quantization is successful:
```none
All done!
```

The following information in the log is the evaluation task result (percentage accuracy):
```
ACC: {'arc_challenge': 42.83, 'arc_easy': 70.88, 'hellaswag': 73.63, 'lambada_openai': 72.0, 'piqa': 77.48, 'winogrande': 67.88, 'acc_avg': 67.45}
```
The following information is perplexity (wikitext, max length 512):
```
PPL score: 5.870388984680176
```
The following information is the inference speed (ms) of the original model and the true quantization model:
```
Time diff orig: 929.0580000000001
Time diff after real quant: 139.707
```

After the script finishes running, calibration parameters `./outputs/llama2_7b/flat_matrices.pth` and quantization log file `./amct_log/amct_pytorch.log` will be generated and saved in the current directory. If you want to directly load calibration parameters, use the following settings:
```python
python3 src/run_llama2_samples.py --model_path <llama2 model path> --load_matrix --flat_matrix_path <matrix path, e.g. ./outputs/llama2_7b/flat_matrices.pth>
```

### 2.2 Qwen3 Quantization

**step 1.** Please execute the following command in the current directory to run the sample program and modify the model path in the sample program according to actual conditions:
```python
python3 src/run_qwen_samples.py --model_path <qwen3-8b model path>
```

If the following information appears, it indicates that quantization is successful:
```none
All done!
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
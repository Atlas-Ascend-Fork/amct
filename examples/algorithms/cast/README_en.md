# AMCT Large Model Quantization

## 1 Quantization Prerequisites

### 1.1 Install Dependencies

The dependency packages for this sample can be found in [requirements.txt](requirements.txt)

Note that the torch_npu package version needs to match the Python and torch package versions, and the CANN package needs to be installed

### 1.2 Model and Dataset Preparation

This sample uses Llama2-7b, qwen2-7b, and qwen3-8b models with wikitext2 dataset as examples.
Please download the models yourself and pass the model path to the script. The dataset is loaded online.

### 1.3 Simple Quantization Configuration
The quantization configuration used in this sample is built into the tool and can be obtained and used in the following ways:

`from amct_pytorch import HIFP8_CAST_CFG`

If you need to modify the detailed configuration, please refer to the documentation to construct the required quantization configuration dict.

The cast algorithm supports weight-only quantization and full quantization. The supported quantization types and quantization configurations are:

| Field | Type | Description | Value Range | Notes |
|:--| :-: | :-- | :-: | :-: | :-- |
|batch_num|uint32|Number of batches used for quantization |1|/|
|skip_layers|str|Layers to skip quantization |/|Skip quantization layers support fuzzy matching. When the configured string is a layer name substring or matches the layer name, skip quantization for that layer and do not generate quantization configuration. The string must contain numbers or letters|
|weights.type|str|Quantized weight type|'hifloat8'|/|
|weights.symmetric|bool|Symmetric quantization|TRUE|/|
|weights.strategy|str|Quantization granularity|'tensor'/'channel'|/|
|inputs.type|str|Quantized activation type|'hifloat8'|/|
|inputs.symmetric|bool|Symmetric quantization|TRUE|/|
|inputs.strategy|str|Quantization granularity|'tensor'/'token'|/|
|algorithm|dict|Quantization algorithm configuration used|{'cast'}|/|

### 1.4 HiFloat4 (hifloat4) Direct-Cast Configuration

The cast algorithm also supports HiFloat4 direct-cast. The built-in config is:

`from amct_pytorch import HIFP4_CAST_CFG`

Difference from HiFloat8: the current torch_npu has no native hifloat4 dtype or low-bit matmul, so
HiFloat4 direct-cast is a **pure-FP simulation (fake-quant)** used for offline accuracy/PPL evaluation.
Set `weights.type` / `inputs.type` to `'hifloat4'`; the other fields (symmetric=True,
strategy=tensor/channel) are the same as HiFloat8.

> ⚠️ **Do NOT call `amct.convert()` for HiFloat4.** convert() would look for a HiFloat4 NPU deploy op
> that does not exist yet and raise. Run the (simulated) model directly after `amct.quantize()` to
> measure PPL. The native NPU operator is future work (Path 2, see `HiFloat4_migration_guide.md`).

## 2 Quantization Example

### 2.1 Use Interface Method to Call

**step 1.** Please execute the following command in the current directory to run the sample program. Users need to modify the model path in the sample program according to actual conditions:

```python
python3 src/run_llama2_samples.py --model_path=/data/Llama2_7b_hf/
```

HiFloat4 simulation example (uses `HIFP4_CAST_CFG`, no convert):

```python
python3 src/run_llama2_hifloat4_samples.py --model_path=/data/Llama2_7b_hf/
```

```python
python3 src/run_qwen_samples.py --model_path=/data/Qwen2-7b/
```

```python
python3 src/run_qwen_samples.py --model_path=/data/Qwen3-8B/
```


If the following information appears, it indicates that quantization is successful:

```none
Test time taken:  1.0 min  59.24865388870239 s
Score:  5.477707
```

Where Score is the quantized model PPL. For specific values, refer to the following table:

| Model | Calibration Set | Dataset | Pre-quantization PPL | Post-quantization PPL | 
| :-: | :-: | :-: | :-: | :-: |
|LLAMA2-7B|pileval|wikitext2|5.472|5.524|
|QWEN2-7B|pileval|wikitext2|7.137|7.188|
|QWEN3-8B|pileval|wikitext2|9.715|9.745|

After inference succeeds, a quantization log file ./amct_log/amct_pytorch.log is generated in the current directory
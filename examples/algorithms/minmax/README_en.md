# AMCT Large Model MINMAX Quantization

## 1 Quantization Prerequisites

### 1.1 Install Dependencies

The dependency packages for this sample can be found in [requirements.txt](requirements.txt)

Note that the torch_npu package version needs to match the Python and torch package versions, and the CANN package needs to be installed

### 1.2 Model and Dataset Preparation

This sample uses Llama2-7b, qwen2-7b, and qwen3-8b models, pileval data, and wikitext2 dataset as examples. Data is loaded online, and models need to be downloaded by users themselves and the model path needs to be specified when executing the script.

Note: The quantization data type combination float8_e4m3fn * float4_e2m1 only supports quantizing original data type torch.bfloat16. Please modify the data type when getting the model in the src/utils.py file.


> **NPU Operator Dimension Limitation:** The NPU quantization operator `aclnnWeightQuantBatchMatmulV2` has an upper limit of 65535 for both k (input feature dimension) and n (output feature dimension). Large-vocabulary models such as Qwen2-7B / Qwen3-8B have a vocabulary size of approximately 152K, which far exceeds this limit, causing the `lm_head` layer to fail when invoking this operator during PPL evaluation. The built-in quantization config `INT8_MINMAX_WEIGHT_QUANT_CFG` now includes `lm_head` in `skip_layers` by default, and the tool also automatically skips layers exceeding the dimension limit in `check_quant_op_constraint`. If you use a custom quantization config, make sure to add `lm_head` to `skip_layers` for large-vocabulary models:
> 

### 1.3 Simple Quantization Configuration
The quantization configuration used in this sample is built into the tool and can be obtained and used in the following ways:

INT8 weight-only quantization configuration:
`from amct_pytorch import INT8_MINMAX_WEIGHT_QUANT_CFG`
Quantization data type combination float8_e4m3fn * float4_e2m1 configuration:
```python
cfg = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'float4_e2m1',
            'symmetric': True,
            'strategy': 'group',
            'group_size': 32
        },
        'inputs': {
            'type': 'float8_e4m3fn',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'minmax'},
    'skip_layers': {'lm_head'}
}
```

If you need to modify the detailed configuration, please refer to the documentation to construct the required quantization configuration dict.

The MINMAX algorithm supports weight-only quantization and full quantization. The supported quantization types and quantization configurations are:

| Field | Type | Description | Value Range | Notes |
|:--| :-: | :-- | :-: | :-: | :-- |
|batch_num|uint32|Number of batches used for quantization |1|/|
|skip_layers|str|Layers to skip quantization |/|Skip quantization layers support fuzzy matching. When the configured string is a layer name substring or matches the layer name, skip quantization for that layer and do not generate quantization configuration. The string must contain numbers or letters|
|weights.type|str|Quantized weight type|'int4'/'int8'/'float4_e2m1'|/|
|weights.symmetric|bool|Symmetric quantization|TRUE/FALSE|When quantization data type is float4_e2m1, only symmetric quantization is supported|
|weights.strategy|str|Quantization granularity|'tensor'/'channel'/'group'|When quantization strategy is group, only quantization data type float4_e2m1 is supported, and float4_e2m1 only supports group|
|inputs.type|str|Quantized weight type|'int8'/'float8_e4m3fn'|Full quantization scenario does not support configuring weight quantization type int4|
|inputs.symmetric|bool|Symmetric quantization|TRUE/FALSE|When quantization data type is float8_e4m3fn, only symmetric quantization is supported|
|inputs.strategy|str|Quantization granularity|'tensor'/'token'|When quantization data type is float8_e4m3fn, only quantization strategy tensor is supported|
|algorithm|dict|Quantization algorithm configuration used|{'minmax'}|/|

## 2 Quantization Example

### 2.1 Use Interface Method to Call

**step 1.** Please execute the following command in the current directory to run the sample program. Users need to modify the model and dataset paths in the sample program according to actual conditions:

```python
python3 src/run_llama2_samples.py --model_path=/data/Llama2_7b_hf/
```

```python
python3 src/run_qwen_samples.py --model_path=/data/Qwen2-7b/
```

```python
python3 src/run_qwen_samples.py --model_path=/data/Qwen3-8b/
```

If the following information appears, it indicates that quantization is successful:

```none
Test time taken:  1.0 min  59.24865388870239 s
Score:  5.477707
```

Where Score is the quantized model PPL. For specific values, refer to the following table:

| Model | Calibration Set | Dataset | Pre-quantization PPL | Post-int8 quantization PPL  | Post-float8_e4m3fn*float4_e2m1 quantization PPL | 
| :-: | :-: | :-: | :-: | :-: | :-: |
|LLAMA2-7B|pileval|wikitext2|5.472|5.477|5.702|
|QWEN2-7B|pileval|wikitext2|7.137|7.139|7.602|
|QWEN3-8B|pileval|wikitext2|9.715|9.692|10.668|


After inference succeeds, a quantization log file ./amct_log/amct_pytorch.log is generated in the current directory

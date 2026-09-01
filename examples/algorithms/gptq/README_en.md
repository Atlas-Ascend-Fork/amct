# AMCT Large Model GPTQ Quantization

## 1 Quantization Prerequisites

### 1.1 Install Dependencies

The dependency packages for this sample can be found in [requirements.txt](requirements.txt)

Note that the torch_npu package version needs to match the Python and torch package versions, and the CANN package needs to be installed

### 1.2 Model and Dataset Preparation

This sample uses Llama2-7b and qwen2-7b models, pileval data, and wikitext2 dataset as examples. Data is loaded online, and models need to be downloaded by users themselves and the model path needs to be specified when executing the script.

### 1.3 Simple Quantization Configuration
The quantization configuration used in this sample is built into the tool and can be obtained and used in the following ways:

INT4 weight-only quantization configuration:
`from amct_pytorch import INT4_GPTQ_WEIGHT_QUANT_CFG`
MXFP4_E2M1 weight-only quantization configuration:
```python
cfg = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'mxfp4_e2m1',
            'symmetric': True,
            'strategy': 'group',
            'group_size': 32
        },
    },
    'algorithm': {'gptq'},
    'skip_layers': {'lm_head'}
}
```

If you need to modify the detailed configuration, please refer to the documentation to construct the required quantization configuration dict.

The GPTQ algorithm only supports weight quantization. The supported quantization types and quantization configurations are:

| Field | Type | Description | Value Range | Notes |
|:--| :-: | :-- | :-: | :-: | :-- |
|batch_num|uint32|Number of batches used for quantization |1|/|
|skip_layers|str|Layers to skip quantization |/|Skip quantization layers support fuzzy matching. When the configured string is a layer name substring or matches the layer name, skip quantization for that layer and do not generate quantization configuration. The string must contain numbers or letters|
|weights.type|str|Quantized weight type|'int4'/'int8'/'float4_e2m1'/'mxfp4_e2m1'|/|
|weights.symmetric|bool|Symmetric quantization|TRUE/FALSE|float4_e2m1 and mxfp4_e2m1 only support symmetric quantization configuration|
|weights.strategy|str|Quantization granularity|'tensor'/'channel'/'group'|float4_e2m1 and mxfp4_e2m1 only support group strategy configuration|
|algorithm|dict|Quantization algorithm configuration used|{'gptq'}|/|

## 2 Quantization Example

### 2.1 Use Interface Method to Call

**step 1.** Please execute the following command in the current directory to run the sample program. Users need to modify the model and dataset paths in the sample program according to actual conditions:

```python
python3 src/run_llama2_samples.py --model_path=/data/Llama2_7b_hf/
```

```python
python3 src/run_qwen_samples.py --model_path=/data/Qwen2-7b/
```

If the following information appears, it indicates that quantization is successful:

```none
Test time taken:  1.0 min  59.24865388870239 s
Score:  5.477707
```

**step 2.** Recommended to use the following configuration

Where Score is the quantized model PPL. For specific values, refer to the following table:

| Model | Calibration Set | Dataset | Pre-quantization PPL | Post-INT4 quantization PPL | Post-MXFP4 quantization PPL |
| :-: | :-: | :-: | :-: | :-: | :-: |
|LLAMA2-7B|pileval|wikitext2|5.472|5.601|5.799
|QWEN2-7B|pileval|wikitext2|7.137|7.253|7.305


After inference succeeds, a quantization log file ./amct_log/amct_pytorch.log is generated in the current directory
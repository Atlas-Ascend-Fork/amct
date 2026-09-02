# AMCT Large Model Quantile Quantization

## 1 Quantization Prerequisites

### 1.1 Install Dependencies

The dependency packages for this sample can be found in [requirements.txt](requirements.txt)

Note that the torch_npu package version needs to match the Python and torch package versions, and the CANN package needs to be installed

### 1.2 Model and Dataset Preparation

This sample uses Llama2-7b, qwen2-7b, and qwen3-8b models, pileval data, and wikitext2 dataset as examples. Data is loaded online, and models need to be downloaded by users themselves and the model path needs to be specified when executing the script.

### 1.3 Simple Quantization Configuration
The quantization configuration used in this sample is built into the tool and can be obtained and used in the following ways:

HIF8 full quantization configuration:
`from amct_pytorch import HIFP8_QUANTILE_CFG`

```python
cfg = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',
        },
        'inputs': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'quantile'},
    'skip_layers': {'lm_head'}
}
```

If you need to modify the detailed configuration, please refer to the documentation to construct the required quantization configuration dict.

The Quantile algorithm supports weight-only quantization and full quantization. The supported quantization types and quantization configurations are:

| Field | Type | Description | Value Range | Notes |
| :------------------| :------:| :--------------------| :------------------:| :-----------------------------------------------------------------------------------------------------------------------|
| batch_num | uint32 | Number of batches used for quantization | 1 | / |
| skip_layers | str | Layers to skip quantization | / | Skip quantization layers support fuzzy matching. When the configured string is a layer name substring or matches the layer name, skip quantization for that layer and do not generate quantization configuration. The string must contain numbers or letters |
| weights.type | str | Quantized weight type | 'hifloat8' | The Quantile algorithm is mainly optimized for HIF8 data type |
| weights.symmetric | bool | Symmetric quantization | True/False | HIF8 data type supports both symmetric and asymmetric quantization |
| weights.strategy | str | Quantization granularity | 'tensor'/'channel' | Supports per-tensor and per-channel |
| inputs.type | str | Quantized activation type | 'hifloat8' | Required configuration in full quantization scenario |
| inputs.symmetric | bool | Symmetric quantization | True/False | HIF8 data type supports both symmetric and asymmetric quantization |
| inputs.strategy | str | Quantization granularity | 'tensor'/'token' | Supports per-tensor and per-token (static/dynamic) |
| inputs.dynamic | bool | Quantization mode | True/False | When inputs.strategy is configured as per-token, both static and dynamic are supported |
| algorithm | dict | Quantization algorithm configuration used | {'quantile'} | / |

## 2 Quantization Example

### 2.1 Use Interface Method to Call

**step 1.** Please execute the following command in the current directory to run the sample program. Users need to modify the model and dataset paths in the sample program according to actual conditions:

Use built-in configuration for HIF8 full quantization:

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

| Model | Calibration Set | Dataset | Pre-quantization PPL | Post-quantization PPL |
| :---------:| :-------:| :---------:| :---------:| :---------:|
| LLAMA2-7B | pileval | wikitext2 | 5.472 | 5.507 |
| QWEN2-7B | pileval | wikitext2 | 7.137 | 7.169 |
| QWEN3-8B | pileval | wikitext2 | 9.715 | 9.760 |


After inference succeeds, a quantization log file ./amct_log/amct_pytorch.log is generated in the current directory
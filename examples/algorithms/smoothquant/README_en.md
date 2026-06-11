# AMCT Large Model SMOOTHQUANT Quantization

## 1 Quantization Prerequisites

### 1.1 Install Dependencies

The dependency packages for this sample can be found in [requirements.txt](requirements.txt)

Note that the torch_npu package version needs to match the Python and torch package versions, and the CANN package needs to be installed

### 1.2 Model and Dataset Preparation

This sample uses Llama2-7b, qwen2-7b, and qwen3-8b models, pileval data, and wikitext2 dataset as examples. Data is loaded online, and models need to be downloaded by users themselves and the model path needs to be specified when executing the script.

Note: The quantization data type combination float8_e4m3fn * float4_e2m1 only supports quantizing original data type torch.bfloat16. Please modify the data type when getting the model in the src/utils.py file.

### 1.3 Simple Quantization Configuration
The quantization configuration used in this sample is built into the tool and can be obtained and used in the following ways:

Quantization data type combination int8 * int8 configuration:
`from amct_pytorch import INT8_SMOOTHQUANT_CFG`
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
    'algorithm': {'smoothquant': {'smooth_strength': 0.77}},
    'skip_layers': {'lm_head', 'down_proj'}
}
```
Quantization data type combination hifloat8 * hifloat8 configuration:
`from amct_pytorch import HIFP8_SMOOTHQUANT_CFG`

The base configuration is as follows, and `smooth_strength=0.5` and `0.8` can be tested respectively:
```python
{
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'channel',
        },
        'inputs': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'smoothquant': {'smooth_strength': 0.5}},
    'skip_layers': {'lm_head'}
}
```

If you need to modify the detailed configuration, please refer to the documentation to construct the required quantization configuration dict.

The SmoothQuant algorithm only supports full quantization. The supported quantization types and quantization configurations are:

| Field | Type | Description | Value Range | Notes |
|:--| :-: | :-- | :-: | :-: | :-- |
|batch_num|uint32|Number of batches used for quantization |1|/|
|skip_layers|str|Layers to skip quantization |/|Skip quantization layers support fuzzy matching. When the configured string is a layer name substring or matches the layer name, skip quantization for that layer and do not generate quantization configuration. The string must contain numbers or letters|
|weights.type|str|Quantized weight type|'int8'/'float4_e2m1'/'hifloat8'|/|
|weights.symmetric|bool|Symmetric quantization|TRUE/FALSE|When quantization data type is float4_e2m1, only symmetric quantization is supported|
|weights.strategy|str|Quantization granularity|'tensor'/'channel'/'group'|When quantization strategy is group, only quantization data type float4_e2m1 is supported, and float4_e2m1 only supports group|
|inputs.type|str|Quantized activation type|'int8'/'float8_e4m3fn'/'hifloat8'|/|
|inputs.symmetric|bool|Symmetric quantization|TRUE/FALSE|When quantization strategy is token, asymmetric quantization is not supported, and per-group quantization is not supported; when quantization data type is float8_e4m3fn, only symmetric quantization is supported|
|inputs.strategy|str|Quantization granularity|'tensor'/'token'|When quantization data type is float8_e4m3fn, only quantization strategy tensor is supported|
|algorithm|dict|Quantization algorithm configuration used|{'smoothquant'}|/|
|algorithm.smoothquant.smooth_strength|float|SmoothQuant algorithm parameter: migration strength|0-1|Does not include 0/1|


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

| Model | Calibration Set | Dataset | Pre-quantization PPL | Post-int8*int8 quantization PPL | Post-float8_e4m3fn*float4_e2m1 quantization PPL | 
| :-: | :-: | :-: | :-: | :-: | :-: |
|LLAMA2-7B|pileval|wikitext2|5.472|5.673|5.589|
|QWEN2-7B|pileval|wikitext2|7.137|7.155|7.252|
|QWEN3-8B|pileval|wikitext2|9.715|9.861|9.931|


### 2.2 hifloat8 * hifloat8 accuracy reference (Qwen3-8B + wikitext2)

Using `HIFP8_SMOOTHQUANT_CFG` (weights: hifloat8/channel, inputs: hifloat8/tensor, skip_layers=`{'lm_head'}`),
full PPL evaluation on Qwen3-8B and wikitext2-raw-v1/test. Accuracy before and after quantization:

Full 2048-token evaluation (146 segments, pre-quantization PPL 9.7252):

| smooth_strength | Post-quantization PPL | PPL delta | Relative degradation |
| ---: | ---: | ---: | ---: |
| 0.5 | 9.7719 | +0.0467 | +0.4804% |
| 0.8 | 9.7357 | +0.0105 | +0.1080% |

Full 4096-token evaluation (73 segments, pre-quantization PPL 8.9975):

| smooth_strength | Post-quantization PPL | PPL delta | Relative degradation |
| ---: | ---: | ---: | ---: |
| 0.5 | 9.0288 | +0.0313 | +0.3481% |
| 0.8 | 9.0359 | +0.0385 | +0.4276% |

Under 2048-token evaluation, `smooth_strength=0.8` is more accurate than the default `0.5`; under 4096-token evaluation, `smooth_strength=0.5` is slightly better than `0.8`. In practice, `smooth_strength` can be tuned according to sequence length.


After inference succeeds, a quantization log file ./amct_log/amct_pytorch.log is generated in the current directory
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

## 3 Exporting an offline torchair model

### 3.1 Prerequisites

- The model has been quantized by AMCT and is the quantized deploy model produced by `amct.convert` (see [2.1 Invoking via the API](#21-invoking-via-the-api) for the quantization workflow);
- CANN, torch_npu and torchair are installed with matching versions;
- Source the CANN environment before running the script (adjust to your actual installation path):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

- Python dependencies (`torch`/`torch_npu`/`torchair`/`transformers`/`amct_pytorch`) are installed with matching versions; if the tokenizer needs to be downloaded online but the network is blocked, set the mirror via `export HF_ENDPOINT=https://hf-mirror.com`.

**How to save the quantized deploy model**: the model after `amct.convert` contains AMCT custom quantization modules (e.g. `NpuQuantizationLinear`) and cannot be restored via `save_pretrained`/`AutoModel.from_pretrained`. Save the whole model with `torch.save` right after `amct.convert`, and keep the tokenizer files of the original model:

```python
amct.convert(quant_model)
torch.save(quant_model.cpu(), '/path/to/quantized_llm/model.pth')
enc.save_pretrained('/path/to/quantized_llm/')  # tokenizer files, used by the export script to build the input
```

`--model_path` should point to that directory (containing `model.pth`), or directly to the `model.pth` file. The required directory structure is:

```
/path/to/quantized_llm/
├── model.pth              # quantized deploy model saved by torch.save
├── tokenizer.json         # tokenizer files saved by enc.save_pretrained
├── tokenizer_config.json
└── ...                    # other tokenizer files (vocab.json/merges.txt etc., depending on the model)
```

### 3.2 Usage

For LLM networks (e.g. Qwen/Llama-like models), the export input is a `[1, seqlen]` int64 tensor tokenized from real text; by default a built-in sample text is used, or specify a local text file via `--input_file`:

```python
python3 src/export_torchair_sample.py --model_path=/path/to/quantized_llm/ --seqlen=2048
```

For VLM networks (e.g. CLIP-ViT-like models), the export input is a `[batch_size, 3, H, W]` tensor preprocessed from a real image:

```python
python3 src/export_torchair_sample.py --model_type=vlm --model_path=/path/to/quantized_clip/ --image_path=/path/to/image.jpg --batch_size=1 --dtype=float16
```

Common parameters:

| Parameter | Required | Default | Description |
|:--|:--:|:--:|:--|
|--model_path|Yes|/|Quantized deploy model directory (containing `model.pth`) or the pth file path|
|--model_type|No|llm|`llm`/`vlm`; vlm uses the CLIP-ViT path|
|--output_dir|No|./torchair_output|Directory for exported artifacts|
|--export_name|No|Model directory name|AIR file name (without the .air suffix)|
|--seqlen|No|2048|Sequence length of the LLM export input|
|--input_file|No|Built-in sample text|Text file tokenized as the LLM export input|
|--image_path|Required for vlm|/|Path of the real image used to trace the VLM forward|
|--batch_size|No|1|Batch size of the VLM export input|
|--dtype|No|float16|Dtype of the VLM image input (float16/bfloat16)|

### 3.3 Exported artifacts

After a successful export, the layout under `--output_dir` is:

```
output_dir/
├── <export_name>.air      # AIR offline graph (weights embedded)
├── input.bin              # export input binary
└── weights/               # generated automatically for large models (total weights over ~1.8G)
    ├── dynamo.pbtxt       # TorchDynamo intermediate representation (text)
    └── xxx_weight ...     # externalized weight files (referenced by .air via FileConstant)
```

| File | Description |
|:--|:--|
|`<export_name>.air`|AIR offline graph exported by torchair, for offline inference; weights are embedded when small, or referenced from `weights/` via FileConstant when large|
|`weights/dynamo.pbtxt`|TorchDynamo intermediate representation (text format)|
|`weights/xxx_weight`|Externalized weight binary files; must be delivered together with the `.air` and must not be moved or renamed independently|
|`input.bin`|Binary file of the export input, for downstream offline inference framing|

**Note**: when the `weights/` directory is generated, keep the relative layout between `.air` and `weights/` when copying for offline inference deployment.

### 3.4 Notes

- The shape and dtype of the export input must be consistent with downstream offline inference; control them via `--seqlen`/`--batch_size`/`--dtype`;
- LLM export forces the eager attention implementation (the script sets `config._attn_implementation='eager'` automatically): sdpa attention is redispatched by torch_npu to the `npu_fusion_attention_v3` op, for which the current torchair release has no AscendIR converter yet;
- After loading, the script automatically moves tensors not migrated by `.npu()` in quantization modules (e.g. `bias`/`offset_bias`) to NPU;
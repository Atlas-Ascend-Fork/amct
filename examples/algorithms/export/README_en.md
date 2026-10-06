# AMCT large-model offline torchair model export

> This document describes how to use `src/export_torchair_sample.py` to export an AMCT-quantized deploy model to a torchair AIR offline graph for downstream offline inference.

## 1 Prerequisites

- The model has been quantized by AMCT and is the quantized deploy model produced by `amct.convert` (see the quantization documentation of the corresponding algorithm, e.g. [smoothquant](../smoothquant/README_en.md));
- CANN, torch_npu and torchair are installed with matching versions;
- Source the CANN environment before running the script (adjust to your actual installation path):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

- Python dependencies (`torch`/`torch_npu`/`torchair`/`transformers`/`amct_pytorch`/`Pillow`) are installed with matching versions (see [requirements.txt](requirements.txt)); if the tokenizer needs to be downloaded online but the network is blocked, set the mirror via `export HF_ENDPOINT=https://hf-mirror.com`.
- Security note: the script disables `--trust_remote_code` by default. Only load trusted local model directories; enabling `--trust_remote_code` or `torch.load` deserialization may execute custom code from the model directory. Do not load models from untrusted sources.

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

## 2 Usage

For LLM networks (e.g. Qwen/Llama-like models), the export input is a `[1, seqlen]` int64 tensor tokenized from real text; by default a built-in sample text is used, or specify a local text file via `--input_file`:

```python
python3 src/export_torchair_sample.py --model_path=/path/to/quantized_llm/ --seqlen=2048
```

For VLM networks (e.g. CLIP-ViT-like, image-only input), `--image_path` may be a single image file or a directory containing multiple images; the script preprocesses them into a `[batch_size, 3, H, W]` tensor:

```python
python3 src/export_torchair_sample.py --model_type=vlm --model_path=/path/to/quantized_clip/ --image_path=/path/to/image.jpg --batch_size=1 --dtype=float16

python3 src/export_torchair_sample.py --model_type=vlm --model_path=/path/to/quantized_clip/ --image_path=/path/to/image_dir/ --batch_size=1 --dtype=float16
```

> Note: this script only supports image-only VLM models (e.g. CLIP); multimodal text+image VLMs (Qwen2-VL/LLaVA...) are not supported.

Common parameters:

| Parameter | Required | Default | Description |
|:--|:--:|:--:|:--|
|--model_path|Yes|/|Path of the quantized deploy model or its containing directory|
|--model_type|No|llm|`llm`/`vlm`; vlm only supports image-only CLIP-like models|
|--output_dir|No|./torchair_output|Directory for exported artifacts|
|--export_name|No|Model directory name for a directory, or file name without suffix for a .pth file|AIR file name (without the .air suffix)|
|--seqlen|No|2048|Sequence length of the LLM export input (only applies to the llm scenario)|
|--input_file|No|Built-in sample text|Text file used to build the LLM export input (its content is tokenized into the export input)|
|--image_path|No|/|Image file or directory of images (jpg/jpeg/png/bmp/tiff/webp) used to trace the VLM forward (optional; a placeholder input with the processor default size is used if not provided)|
|--batch_size|No|1|Batch size of the VLM export input (must be <= number of images)|
|--dtype|No|float16|Dtype of the VLM image input (float16/bfloat16/float32, must match the model floating-point weight dtype); only applies to the vlm scenario, as the LLM export input is int64 token ids with no float precision to tune|
|--trust_remote_code|No|False|Execute custom Python code in the model directory; disabled by default, enable only for trusted local models|

## 3 Exported artifacts

After a successful export, the layout under `--output_dir` is:

```
output_dir/
├── <export_name>.air      # AIR offline graph (weights embedded)
├── input.bin              # export input binary
├── dynamo.pbtxt           # TorchDynamo intermediate representation (generated at the top level for small models with embedded weights)
└── weights/               # generated automatically for large models (total weights over ~1.8G)
    ├── dynamo.pbtxt       # TorchDynamo intermediate representation (generated under weights/ for large models)
    └── xxx_weight ...     # externalized weight files (referenced by .air via FileConstant)
```

| File | Description |
|:--|:--|
|`<export_name>.air`|AIR offline graph exported by torchair, for offline inference; weights are embedded when small, or referenced from `weights/` via FileConstant when large|
|`dynamo.pbtxt`|TorchDynamo intermediate representation (text format); generated at the top level for small models with embedded weights, or under `weights/` for large models|
|`weights/xxx_weight`|Externalized weight binary files; must be delivered together with the `.air` and must not be moved or renamed independently|
|`input.bin`|Binary file of the export input, for downstream offline inference framing|

**Note**: when the `weights/` directory is generated, keep the relative layout between `.air` and `weights/` when copying for offline inference deployment.

## 3.1 Convert to om with ATC

After exporting the `.air`, use ATC to convert it into an offline inference model `.om`:

```bash
cd ./<output_dir>
atc --model=./<export_name>.air --framework=1 --output=./<export_name> --soc_version=<target_soc_version>
```

`--framework=1` means the input is in AIR format; `--soc_version` is the version of the target chip (e.g. `Ascend910B`, `Ascend310P`), which can be checked via `npu-smi info`.

## 4 Notes

- The shape and dtype of the export input must be consistent with downstream offline inference; control them via `--seqlen`/`--batch_size`/`--dtype`; the VLM export `--dtype` must match the model floating-point weight dtype;
- LLM export forces the eager attention implementation (the script sets `config._attn_implementation='eager'` automatically): sdpa attention is redispatched by torch_npu to the `npu_fusion_attention_v3` op, for which the current torchair release has no AscendIR converter yet;
- After loading, the script automatically moves tensors not migrated by `.npu()` in quantization modules (e.g. `bias`/`offset_bias`) to NPU;

# DeepSeek-V4 Model Quantization Conversion Tool

The DeepSeekV4 model attention module uses FP8 data type, and the MoE module uses MXFP4 data type. Atlas A2 and Atlas A3 series products do not support FP8/MXFP4.

This feature converts FP8/MXFP4 to int8 data type, enabling developers to experience the inference effect of the DeepSeekV4 model on Atlas A2 and Atlas A3 series products.

## Usage Method

1. Download the DeepSeekV4 model to local.
2. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
3. Execute the model conversion script.
    ```bash
    python convert_model.py \
        --input_fp8_hf_path <FP8 model path> \
        --output_hf_path <output path> \
        --quant_type <quantization type>
    ```
4. Obtain target type (e.g., int8) weights and quantization parameters.
5. Perform inference verification on Atlas A2 and Atlas A3 series products.

### Parameter Description

| Parameter | Required | Default Value | Description |
|------|------|--------|------|
| `--input_fp8_hf_path` | Yes | - | FP8 model directory path |
| `--output_hf_path` | Yes | - | Output directory path |
| `--quant_type` | No | `w8a8-int` | Quantization type |

### Supported Quantization Types

| Type | Description |
|------|------|
| `bfloat16` | Convert to BF16 (no quantization) |
| `w8a8-int` | W8A8 integer quantization |
| `w8a8-mx` | W8A8 MX floating-point quantization |
| `w4a8-mx` | W4A8 MX floating-point quantization |

## Output Results

```
<output_hf_path>/
├── model-00001-of-000xx.safetensors   # Quantized weights
├── model.safetensors.index.json       # Weight index
├── config.json                        # Model configuration (containing quantization parameters)
└── *.py / *.json / *.jinja            # Original directory file copies
```

### Weight Changes

| Change Item | Description |
|--------|------|
| Quantization layer weights | INT8 / MXFP4 / MXFP8 format |
| `.scale` tensors | New scaling factors for each quantization layer |
| Non-quantization layers | BF16 format |

### config.json New Fields

| Field | Description |
|------|------|
| `quant_method` | `compressed-tensors` or `mxfp8` |
| `format` | `int-quantized` or `float-quantized` |
| `ignore` | Layers to skip quantization |
| `kv_cache_scheme` | KV cache quantization configuration |
| `config_groups` | Quantization parameters for each layer |
| `weight_block_size` | Weight block size (MX format) |


## Conversion Description

### Quantization Layers

The following layers will be re-quantized when quant_type is `w8a8-*` or `w4a8-*`:

| Layer Type | Layer Name Pattern | Original Format |
|-------|---------|---------|
| MoE Experts | `layers.{i}.ffn.experts.{j}.w1/w2/w3` | MXFP4 |
| Shared Experts | `layers.{i}.ffn.shared_experts.w1/w2/w3` | MXFP4 |
| Attention | `layers.{i}.attn.wq_a/wkv/wo_a/wq_b/wo_b` | FP8 |
| Indexer | `layers.{i}.attn.indexer.wq_b` | FP8 |
| MTP | `mtp.0.*` related layers | FP8/MXFP4 |

**Quantization Output Format:**

| quant_type | Output Format |
|-----------|---------|
| w8a8-int | INT8 + scale |
| w8a8-mx | MXFP8 + scale |
| w4a8-mx | MXFP4 + scale |

### Non-Quantization Layers

The following layers only perform FP8→BF16 dequantization without re-quantization:

| Layer Type | Layer Name Pattern |
|-------|---------|
| Word Embedding | `embed.weight` |
| Output Head | `head`, `mtp.0.head` |
| Indexer Weight Projection | `layers.{i}.attn.indexer.weights_proj` |
| Compressor | `layers.{i}.attn.indexer.compressor.wgate/wkv`, `layers.{i}.attn.compressor.wgate/wkv` |

### Weight Conversion

Original model weight format:
- Attention: FP8, with every 128×128 weight block sharing one scale.
- MoE: MXFP4, two 4bit values packed and stored in one uint8, with every 32 elements (by column) sharing one scale.

Traverse weight tensors in safetensor files:

1. **Scaling factors** (`.scale` suffix): Skip during traversal, read as needed when processing corresponding weights.

2. **FP8/MXFP4 weights** (`.weight` suffix):
    - FP8: dtype is floating-point type, read directly.
    - MXFP4: dtype=int8, needs unpacking (extract low/high 4 bits, map to E2M1 floating-point values), element count doubles.
    - Read corresponding scale, expand to same dimension as weight, multiply element-wise to get BF16.
    - If quantization is needed and layer name is in quantization list, re-quantize.

3. **Other weights**: Convert to BF16.
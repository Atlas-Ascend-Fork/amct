# DeepSeek-V4 模型量化转换工具

deepseekv4模型attention模块采用FP8数据类型，moe模块采用MXFP4数据类型，Atlas A2、Atlas A3系列产品不支持FP8/MXFP4。

本特性将FP8/MXFP4转换为int8数据类型，支撑开发者在Atlas A2、Atlas A3系列产品上体验deepseekv4模型的推理效果。

## 使用方法

1. 下载deepseekv4模型到本地。
2. 依赖安装。
    ```bash
    pip install -r requirements.txt
    ```
3. 执行模型转换脚本。
    ```bash
    python convert_model.py \
        --input_fp8_hf_path <FP8模型路径> \
        --output_hf_path <输出路径> \
        --quant_type <量化类型>
    ```
4. 得到目标类型(e.g. int8)权重和量化参数。
5. 基于Atlas A2、Atlas A3系列产品进行推理验证。

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_fp8_hf_path` | 是 | - | FP8 模型目录路径 |
| `--output_hf_path` | 是 | - | 输出目录路径 |
| `--quant_type` | 否 | `w8a8-int` | 量化类型 |

### 支持的量化类型

| 类型 | 说明 |
|------|------|
| `bfloat16` | 转换为 BF16（无量化） |
| `w8a8-int` | W8A8 整数量化 |
| `w8a8-mx` | W8A8 MX 浮点量化 |
| `w4a8-mx` | W4A8 MX 浮点量化 |

## 输出结果

```
<output_hf_path>/
├── model-00001-of-000xx.safetensors   # 量化后权重
├── model.safetensors.index.json       # 权重索引
├── config.json                        # 模型配置（含量化参数）
└── *.py / *.json / *.jinja            # 原目录文件复制
```

### 权重变更

| 变更项 | 说明 |
|--------|------|
| 量化层权重 | INT8 / MXFP4 / MXFP8 格式 |
| `.scale` 张量 | 每个量化层对应的新增缩放因子 |
| 非量化层 | BF16 格式 |

### config.json 新增字段

| 字段 | 说明 |
|------|------|
| `quant_method` | `compressed-tensors` 或 `mxfp8` |
| `format` | `int-quantized` 或 `float-quantized` |
| `ignore` | 跳过量化的层 |
| `kv_cache_scheme` | KV cache 量化配置 |
| `config_groups` | 各层量化参数 |
| `weight_block_size` | 权重块大小（MX 格式） |


## 转换说明

### 量化层

以下层在 quant_type 为 `w8a8-*` 或 `w4a8-*` 时会被重新量化：

| 层类型 | 层名模式 | 原始格式 |
|-------|---------|---------|
| MoE Experts | `layers.{i}.ffn.experts.{j}.w1/w2/w3` | MXFP4 |
| Shared Experts | `layers.{i}.ffn.shared_experts.w1/w2/w3` | MXFP4 |
| Attention | `layers.{i}.attn.wq_a/wkv/wo_a/wq_b/wo_b` | FP8 |
| Indexer | `layers.{i}.attn.indexer.wq_b` | FP8 |
| MTP | `mtp.0.*` 相关层 | FP8/MXFP4 |

**量化输出格式：**

| quant_type | 输出格式 |
|-----------|---------|
| w8a8-int | INT8 + scale |
| w8a8-mx | MXFP8 + scale |
| w4a8-mx | MXFP4 + scale |

### 非量化层

以下层仅执行 FP8→BF16 反量化，不重新量化：

| 层类型 | 层名模式 |
|-------|---------|
| 词嵌入 | `embed.weight` |
| 输出头 | `head`, `mtp.0.head` |
| Indexer 权重投影 | `layers.{i}.attn.indexer.weights_proj` |
| Compressor | `layers.{i}.attn.indexer.compressor.wgate/wkv`, `layers.{i}.attn.compressor.wgate/wkv` |

### 权重转换

原始模型权重格式：
- Attention: FP8，每 128×128 权重块共享一个 scale。
- MoE: MXFP4，两个 4bit 值打包存储于一个 uint8，每 32 元素（按列）共享一个 scale。

遍历 safetensor 文件中的权重张量：

1. **缩放因子**（`.scale` 后缀）：遍历时跳过，处理对应权重时按需读取。

2. **FP8/MXFP4 权重**（`.weight` 后缀）：
   - FP8: dtype 为浮点类型，直接读取。
   - MXFP4: dtype=int8，需解包（提取低/高4位，映射到 E2M1 浮点值），元素数量翻倍。
   - 读取对应的 scale，扩展至与 weight 相同维度，逐元素相乘得到 BF16。
   - 若需要量化且层名在量化列表中，重新量化。

3. **其他权重**：转换为 BF16。


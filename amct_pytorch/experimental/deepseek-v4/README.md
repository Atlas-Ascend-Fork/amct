# Model Quantization Converter

deepseekv4模型attention模块采用FP8数据类型，moe模块采用MXFP4数据类型，Atlas A2、Atlas A3系列产品不支持FP8/MXFP4
本特性将FP8/MXFP4转换为int8数据类型，支撑开发者在Atlas A2、Atlas A3系列产品上体验deepseekv4模型的推理效果


## 使用方法

1. 下载deepseekv4模型到本地
2. 依赖安装
```bash
pip install -r requirements.txt
```
3. 执行模型转换脚本
```bash
python convert_model.py \
    --input_fp8_hf_path <FP8模型路径> \
    --output_hf_path <输出路径> \
    --quant_type <量化类型>
```
4. 得到目标类型(e.g. int8)权重和量化参数
5. 基于Atlas A2、Atlas A3系列产品进行推理验证

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
| 非量化层 | 保持原始精度 |

### config.json 新增字段

| 字段 | 说明 |
|------|------|
| `quant_method` | `compressed-tensors` 或 `mxfp8` |
| `format` | `int-quantized` 或 `float-quantized` |
| `ignore` | 跳过量化的层 |
| `kv_cache_scheme` | KV cache 量化配置 |
| `config_groups` | 各层量化参数 |
| `weight_block_size` | 权重块大小（MX 格式） |


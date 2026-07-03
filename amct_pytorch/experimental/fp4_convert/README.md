# FP4 → BF16 → HiFloat4 Conversion Tools (Experimental)

FP4 模型权重离线转换工具集，将 NVFP4 / MXFP4 checkpoint 转为 HiFloat4 格式。

## 转换流程

```
NVFP4 packed   ──nvfp4_to_bf16──►  BF16  ──bf16_to_hif4──►  HiFloat4 (E6M2 官方默认 / E6M0 可选)
MXFP4  packed  ──mxfp4_to_bf16──►  BF16  ──bf16_to_hif4──►  HiFloat4
                    (fp4_formats)              │
                                   ┌─ e6m2 (默认): 调官方 encode_to_hifloat4，与 amct HiF4 体系对齐
                                   └─ e6m0 (可选): 简化变体，≠ 官方 HiF4
```

> ⚠️ **E6M0 ≠ 官方 HiF4(E6M2)**。E6M0 是一个纯指数 scale 的简化变体，仅在 `--hif4-format e6m0` 时使用，
> 其数值输出与官方 HiF4 算子不兼容。默认模式为 **e6m2**，与 `amct_ops/hifloat4_cast/reference.py`
> 完全对齐。

## 目录结构

| 文件 | 职责 |
|------|------|
| `fp4_formats.py` | 格式数学层：FP4 解包/E2M1 解码、FP4→BF16、BF16→HiF4 编解码（纯 tensor，可单测） |
| `ckpt_io.py` | 通用 checkpoint I/O 层：读 index、遍历分片、`ShardCache`、`walk_checkpoint`（格式无关，可复用） |
| `metrics.py` | 诊断/验证层：重建误差、饱和率（事后打分，不在转换路径上） |
| `convert_nvfp4_to_hif4.py` | NVFP4 策略 + CLI：NVFP4→BF16→HiF4 整库转换 |
| `convert_mxfp4_to_hif4.py` | MXFP4 策略 + CLI：MXFP4→BF16→HiF4 转换（含 selftest / convert-pair / list-keys） |

## 依赖

```bash
pip install -r requirements.txt
```

## 用法

### NVFP4 → HiF4（整库转换）

```bash
python -m amct_pytorch.experimental.fp4_convert.convert_nvfp4_to_hif4 \
    --nvfp4-dir /path/to/nvfp4_checkpoint/ \
    --bf16-index /path/to/model.safetensors.index.json \
    --out-dir /path/to/output/ \
    --hif4-format e6m2
```

参数：

- `--nvfp4-dir`：NVFP4 TRT-LLM checkpoint 目录（含 `*.safetensors` 分片）。
- `--bf16-index`：BF16 模型的 `model.safetensors.index.json`（作为目标分片计划）。
- `--out-dir`：输出目录。
- `--hif4-format`：`e6m2`（官方 E6M2，默认）或 `e6m0`（简化变体，≠ 官方 HiF4）。
- `--dry-run`：仅打印转换决策，不实际输出。
- `--verbose`：逐 key 打印转换决策。

### MXFP4 → HiF4

```bash
# 自测（随机数据往返 + 误差）
python -m amct_pytorch.experimental.fp4_convert.convert_mxfp4_to_hif4 selftest

# 列 safetensors 中的 key
python -m amct_pytorch.experimental.fp4_convert.convert_mxfp4_to_hif4 list-keys \
    --file model-00001.safetensors --filter experts

# 转换单对张量
python -m amct_pytorch.experimental.fp4_convert.convert_mxfp4_to_hif4 convert-pair \
    --packed-spec model.safetensors:model.layers.0.mlp.weight \
    --scale-spec model.safetensors:model.layers.0.mlp.weight_scale \
    --out out.pt

# 整库转换
python -m amct_pytorch.experimental.fp4_convert.convert_mxfp4_to_hif4 convert-dir \
    --mxfp4-dir /path/to/mxfp4_checkpoint/ \
    --bf16-index /path/to/model.safetensors.index.json \
    --out-dir /path/to/output/ \
    --hif4-format e6m2
```

## 格式模式说明

| 模式 | 行为 | 实现 | 与官方对齐 |
|------|------|------|-----------|
| `e6m2`（默认） | FP4 → BF16 → 官方 HiF4(E6M2) | 调用 `reference.py:encode_to_hifloat4` | ✅ 是 |
| `e6m0` | FP4 → BF16 → HiF4_E6M0 简化变体 | 本地 `bf16_to_hif4_e6m0` | ❌ 否（≠ 官方 HiF4） |

## 架构说明

四层分离：

1. **格式数学**（`fp4_formats.py`）：纯 tensor→tensor 转换，无 I/O，可单独 import 和单测。
2. **通用 I/O**（`ckpt_io.py`）：safetensors 分片遍历/读写，与格式无关，可被其他转换器复用。
3. **诊断**（`metrics.py`）：事后打分，不在转换数据路径上。
4. **策略 + CLI**（`convert_*.py`）：薄层，各自定义"丢哪些 key / 取哪个 scale / 如何 dequant"，
   共享底下的"计算器 + 仓库 + 质检台"。

跨文件依赖方向（只能从上往下调）：

```
convert_*.py  →  fp4_formats / ckpt_io / metrics
metrics / ckpt_io / fp4_formats  之间互不依赖
```

## 测试

```bash
# MXFP4 selftest（随机数据往返验证）
python -m amct_pytorch.experimental.fp4_convert.convert_mxfp4_to_hif4 selftest

# NVFP4 dry-run（仅检查 key 匹配，不写入）
python -m amct_pytorch.experimental.fp4_convert.convert_nvfp4_to_hif4 \
    --nvfp4-dir /path/to/nvfp4_ckpt \
    --bf16-index /path/to/bf16_index.json \
    --out-dir /tmp/test_out \
    --dry-run --verbose
```

# FP4 → BF16 → HiFloat4 Conversion Tools (Experimental)

纯 PyTorch CPU 实现，无 NPU 算子依赖。将 NVFP4 / MXFP4 checkpoint 转为 HiFloat4 格式。

## 转换流程

```
NVFP4 packed  ──nvfp4_to_bf16──►  BF16  ──bf16_to_hif4──►  HiFloat4 (E6M2 / E6M0)
MXFP4 packed  ──mxfp4_to_bf16──►  BF16  ──bf16_to_hif4──►  HiFloat4
                   (fp4_formats)              │
                                  ┌─ e6m2 (默认): 官方 HiF4 E6M2, bit-exact with spec
                                  └─ e6m0 (可选): 简化变体, ≠ 官方 HiF4
```

> ⚠️ **E6M0 ≠ 官方 HiF4(E6M2)**。E6M0 仅在 `--hif4-format e6m0` 时使用。默认模式为 **e6m2**。

## 目录结构

| 文件 | 职责 |
|------|------|
| `fp4_formats.py` | 格式数学层：FP4 解包、FP4→BF16、BF16→HiF4 编解码（纯 tensor，可单测） |
| `ckpt_io.py` | 通用 checkpoint I/O 层：读 index、遍历分片、`walk_checkpoint`（格式无关） |
| `metrics.py` | 诊断/验证层：重建误差（事后打分） |
| `convert_nvfp4_to_hif4.py` | NVFP4 策略 + CLI |
| `convert_mxfp4_to_hif4.py` | MXFP4 策略 + CLI（含 selftest / convert-pair / list-keys） |

## 依赖

```bash
pip install -r requirements.txt
```

## 用法

### NVFP4 → HiF4（整库转换）

```bash
python -m amct_pytorch.experimental.fp4_convert.convert_nvfp4_to_hif4 \
    --nvfp4-dir /path/to/nvfp4_ckpt \
    --bf16-index /path/to/model.safetensors.index.json \
    --out-dir /path/to/output \
    --hif4-format e6m2
```

### MXFP4 → HiF4

```bash
# 自测
python -m amct_pytorch.experimental.fp4_convert.convert_mxfp4_to_hif4 selftest

# 列 key
python -m amct_pytorch.experimental.fp4_convert.convert_mxfp4_to_hif4 list-keys \
    --file model.safetensors --filter experts

# 转换单对张量
python -m amct_pytorch.experimental.fp4_convert.convert_mxfp4_to_hif4 convert-pair \
    --packed-spec model.safetensors:weight \
    --scale-spec  model.safetensors:weight_scale --out out.pt

# 整库转换
python -m amct_pytorch.experimental.fp4_convert.convert_mxfp4_to_hif4 convert-dir \
    --mxfp4-dir /path/to/mxfp4_ckpt \
    --bf16-index /path/to/model.safetensors.index.json \
    --out-dir /path/to/output --hif4-format e6m2
```

## 格式模式

| 模式 | 行为 | 与官方 HiF4 对齐 |
|------|------|-----------------|
| `e6m2`（默认） | FP4 → BF16 → HiF4 E6M2 | ✅ 是 |
| `e6m0` | FP4 → BF16 → HiF4_E6M0 简化变体 | ❌ 否 |

## 架构

四层分离：

```
convert_*.py  →  策略 + CLI（薄层）
    ↓ 调用
fp4_formats.py  →  格式数学（tensor→tensor）
ckpt_io.py      →  通用分片 I/O
metrics.py      →  诊断验证
```

跨文件依赖方向（只能从上往下）：

```
convert_*.py  →  fp4_formats / ckpt_io / metrics
metrics / ckpt_io / fp4_formats  之间互不依赖
```

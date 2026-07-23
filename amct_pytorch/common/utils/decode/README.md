# FP4 → BF16 → HiFloat4 Conversion Tools (Experimental)

将 NVFP4 checkpoint 转为经过 HiF4 fake-quant 的 BF16 权重。

## 转换流程

```
NVFP4 packed  ──nvfp4_to_bf16──►  BF16  ──quant_dequant_float──►  BF16 (HiF4 fake-quant)
                    (fp4_formats)              (hifloat4_quant)
```

两步转换：(1) FP4 反量化 → BF16（纯 torch CPU），(2) HiF4 模拟量化 → BF16
（复用 ``amct_pytorch.common.utils.hifloat4_quant.quant_dequant_float``）。

## 目录结构

| 文件 | 职责 |
|------|------|
| `fp4_formats.py` | 格式数学：FP4→BF16 反量化 |
| `ckpt_io.py` | 通用 checkpoint I/O：`walk_checkpoint` |
| `metrics.py` | 诊断：重建误差打分 |
| `convert_nvfp4_to_hif4.py` | NVFP4 转换 CLI |

## 依赖

```bash
pip install -r requirements.txt
```

依赖 ``jian/hifloat4-cast`` 分支的 ``hifloat4_quant`` 模块。

## 用法

### NVFP4 → HiF4

```bash
python -m amct_pytorch.common.utils.fpx_decode.convert_nvfp4_to_hif4 \
    --nvfp4-dir /path/to/nvfp4_ckpt \
    --bf16-index /path/to/model.safetensors.index.json \
    --out-dir /path/to/output
```


## 架构

四层分离，跨文件依赖方向（只能从上往下）：

```
convert_*.py  →  策略 + CLI（薄层）
    ↓ 调用
fp4_formats.py  →  格式数学（FP4→BF16）
ckpt_io.py      →  通用分片 I/O
metrics.py      →  诊断验证
    ↓ 调用
hifloat4_quant     →  BF16→HiF4 fake-quant（jian/hifloat4-cast 已有）
```

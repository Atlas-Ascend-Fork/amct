# FP4/FP8 → BF16 Decode Tools

将外部量化格式（HiF4 / NVFP4 / MXFP4）的 checkpoint **解码为普通 BF16 权重**。

本目录只负责「解码」这一步：把打包的量化字节还原成 BF16。之后的 HiF4 pack（deploy）
或 HiF4 fake-quant（ptq）由 `hifloat4_quant` 在解码后的 BF16 上完成，**不在本目录内**。

## 解码流程

```
HiF4 packed   ──hif4_bit_reconstruct──►  BF16 / FP16
NVFP4 packed  ──nvfp4_to_bf16────────►   BF16
MXFP4 packed  ──mxfp4_to_bf16────────►   BF16
MXFP8 packed  ──weight_dequant───────►   BF16
                    (fp4_formats)
```

ptq / deploy 入口通过 `detect.detect_source_format` 识别输入格式；外部格式先由对应
unpacker 解码为临时 BF16 checkpoint，再走 HiF4。

## 目录结构

| 文件 | 职责 |
|------|------|
| `detect.py` | 格式识别 + `decode_to_bf16` 分发 |
| `fp4_formats.py` | 格式数学：FP4→BF16 反量化 + 可选 HiF4_E6M0 编解码 |
| `unpack_hif4.py` | HiF4 packed → BF16/FP16 整库解码 |
| `unpack_nvfp4.py` | NVFP4 packed → BF16 整库解码 |
| `unpack_mxfp4.py` | MXFP4 packed → BF16 整库解码（含 e6m0 自测/单对诊断） |
| `unpack_mxfp8.py` | MXFP8 packed → BF16 整库解码（自带 `ShardCache`） |

每个 unpacker 自己跑一个小的「逐分片解码循环」，复用 `common/utils/checkpoint_io.py`
的分片 I/O 原语（`resolve_index` / `invert_weight_map` / `save_shard` / `write_index`）。
`checkpoint_io` 只做格式无关的分片读写，不含任何格式知识。

## 用法

一般无需手动调用——`ptq --quant_dtype hifp` / `deploy` 会自动识别并解码。手动整库解码：

```bash
# NVFP4 → BF16
python -m amct_pytorch.common.utils.decode.unpack_nvfp4 \
    --nvfp4-dir /path/to/nvfp4_ckpt --out-dir /path/to/output

# MXFP4 → BF16
python -m amct_pytorch.common.utils.decode.unpack_mxfp4 convert-dir \
    --mxfp4-dir /path/to/mxfp4_ckpt --out-dir /path/to/output

# MXFP8 → BF16
python -m amct_pytorch.common.utils.decode.unpack_mxfp8 \
    --quant_model_dir /path/to/mxfp8_ckpt --output_dir /path/to/output
```

MXFP4 附带的诊断子命令（可选，非解码主路径）：

```bash
# HiF4 e6m2/e6m0 往返自测
python -m amct_pytorch.common.utils.decode.unpack_mxfp4 selftest --hif4-format e6m0

# 单对张量转换（打包权重 + scale → HiF4）
python -m amct_pytorch.common.utils.decode.unpack_mxfp4 convert-pair \
    --packed-spec model.safetensors:weight \
    --scale-spec  model.safetensors:weight_scale --out out.pt
```

## 依赖

```bash
pip install -r requirements.txt
```

依赖 `jian/hifloat4-cast` 分支的 `hifloat4_quant` 模块（HiF4 codec）。

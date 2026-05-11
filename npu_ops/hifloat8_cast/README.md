# HiFloat8 Cast Kernel

HiFloat8 数据类型转换算子，支持 FP16/BF16 ↔ HiFloat8 双向转换。

## 功能

支持 4 种转换模式：

| castMode | 说明 | Kernel |
|----------|------|--------|
| 0 | FP16 → HiFloat8 | `KernelHiFloat8CastLut` (GM-LUT) |
| 1 | BF16 → HiFloat8 | `KernelHiFloat8CastLut` (GM-LUT) |
| 2 | HiFloat8 → FP16 | `KernelHiFloat8CastLut` (UB-LUT) |
| 3 | HiFloat8 → BF16 | `KernelHiFloat8CastLut` (UB-LUT) |

## 接口

### encode_to_hifloat8(x: Tensor) -> Tensor

将 FP16/BF16 张量编码为 HiFloat8 格式。

**参数**：
- `x`: 输入张量，支持 `torch.float16` / `torch.bfloat16`，任意 shape

**返回**：
- `torch.uint8` 张量，shape 与输入相同

**示例**：
```python
y = encode_to_hifloat8(x_fp16)
y = encode_to_hifloat8(x_bf16)
```

---

### decode_from_hifloat8(x: Tensor, dtype: dtype = torch.float16) -> Tensor

将 HiFloat8 格式解码为 FP16/BF16 张量。

**参数**：
- `x`: HiFloat8 编码张量（`torch.uint8`），任意 shape
- `dtype`: 输出类型，默认 `torch.bfloat16`，支持 `float16` / `bfloat16`

**返回**：
- 解码后的张量，shape 与输入相同

**示例**：
```python
z = decode_from_hifloat8(y)                  # 默认输出 BF16
z = decode_from_hifloat8(y, torch.float16)  # 指定输出 FP16
```

## 文件结构

```
hifloat8_cast/
├── op_kernel/
│   ├── hifloat8_cast_kernel.cpp   device 端 kernel 实现（LUT encode + decode）
│   └── hifloat8_cast_tiling.h     Tiling 常量（TILE_LENGTH=8192）+ 结构体
├── op_host/
│   ├── hifloat8_cast_op.cpp       host 端算子实现（Torch binding + CPU 端 LUT 预计算）
│   ├── ops.h                      PyTorch C++ 扩展函数声明
│   └── register.cpp               TORCH_LIBRARY 注册
├── python/
│   └── hifloat8_cast/
│       ├── __init__.py            Python 接口
│       └── ops.py                 算子函数实现
├── scripts/
│   ├── accuracy_compare_cpu_npu.py  CPU vs NPU 精度对比
│   ├── perf_compare_cpu_npu.py      FP16 性能对比（NPU vs CPU）
│   └── perf_bf16_compare.py         BF16 性能对比（NPU vs CPU）
├── CMakeLists.txt
└── README.md
```

## 编译

```bash
cd npu_ops/hifloat8_cast
source $ASCEND_HOME_PATH/set_env.sh
mkdir -p build && cd build && cmake .. && make -j8
```

## 使用

```python
import sys
sys.path.insert(0, 'python')

from npu_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8
import torch

# 编码（支持任意 shape）
x = torch.randn(1024, 256, dtype=torch.bfloat16, device='npu')
y = encode_to_hifloat8(x)      # shape [1024, 256], dtype uint8

# 解码
z = decode_from_hifloat8(y, torch.bfloat16)  # shape [1024, 256], dtype bfloat16
```

## 性能

**测试平台**: Ascend 910B3 (A2)，CANN 9.0.0  
**吞吐量定义**: (输入字节 + 输出字节) / 耗时  
**推荐场景**: 数据量 ≥ 256K 时 NPU 优势显著

### BF16 ↔ HiFloat8（KernelHiFloat8CastLut，GM-LUT / UB-LUT）

| 数据大小 | Encode (ms) | 吞吐 (MB/s) | Decode (ms) | 吞吐 (MB/s) |
|---------|------------|------------|------------|------------|
| 16K     | 0.310      | 159        | 0.312      | 158        |
| 64K     | 0.323      | 610        | 0.316      | 622        |
| 256K    | 0.321      | 2449       | 0.316      | 2487       |
| 1M      | 1.026      | 3065       | 1.035      | 3040       |
| 4M      | 3.911      | 3218       | 3.881      | 3242       |

### NPU vs CPU 加速比（大数据，≥ 256K）

| 模式 | NPU 吞吐 | CPU 吞吐 | 加速比 |
|------|---------|---------|-------|
| BF16 Encode 1M | 3065 MB/s | 798 MB/s  | **3.8x** |
| BF16 Decode 1M | 3040 MB/s | 983 MB/s  | **3.1x** |

> 小数据（< 64K）因 kernel launch overhead 主导，NPU 优势受限。

## 精度

> CPU 参考实现来自 [`amct_pytorch/experimental/hifloat8/hifloat8_cast.cpp`](../../amct_pytorch/experimental/hifloat8/hifloat8_cast.cpp)，编译为 Python 扩展后由测试脚本以 `import hifloat8_cast` 加载。

CPU 与 NPU 精度逐 bit 一致：

| 测试场景 | 结果 |
|---------|------|
| 随机 FP16/BF16 Encode + Decode（100K 样本） | ✓ 逐 bit 一致 |
| 边界值 Encode（FP16 13 个，BF16 13 个） | ✓ 逐个一致 |
| 特殊值：±0 / ±Inf / NaN（FP16 编码） | ✓ 编码字节一致 |
| 全 256 个 HiF8 值解码（FP16 + BF16） | ✓ 完全一致 |

## 测试

```bash
# 精度验证
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
    python3 scripts/accuracy_compare_cpu_npu.py

# FP16 性能对比
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
    python3 scripts/perf_compare_cpu_npu.py

# BF16 性能对比
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
    python3 scripts/perf_bf16_compare.py
```

> `LD_PRELOAD` 解决 libgomp TLS 问题，仅加载 CPU 参考实现时需要。构建 CPU 参考实现见 [`amct_pytorch/experimental/hifloat8/build.sh`](../../amct_pytorch/experimental/hifloat8/build.sh)。

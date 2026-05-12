# HiFloat8 Cast Kernel

HiFloat8 数据类型转换算子，支持 FP16/BF16 ↔ HiFloat8 双向转换。

## 功能

支持 4 种转换模式，底层统一由 `KernelHiFloat8CastLut` 实现：

| castMode | 转换 | 底层 LUT |
|----------|------|---------|
| 0 | FP16 → HiFloat8 | 32768-entry 半空间 UB-LUT（32 KB） |
| 1 | BF16 → HiFloat8 | 32768-entry 半空间 UB-LUT（32 KB） |
| 2 | HiFloat8 → FP16 | 256-entry UB-LUT（512 B） |
| 3 | HiFloat8 → BF16 | 256-entry UB-LUT（512 B） |

> Python 接口根据输入 dtype 自动选择 castMode，无需手动指定。

## 接口

### encode_to_hifloat8(x: Tensor) -> Tensor

将 FP16/BF16 张量编码为 HiFloat8 格式。

**参数**：
- `x`：输入张量（NPU 设备），dtype 为 `torch.float16` 或 `torch.bfloat16`，任意 shape

**返回**：
- `torch.uint8` 张量，shape 与输入相同

**异常**：
- `RuntimeError`：dtype 不是 float16 / bfloat16，或输入不在 NPU 上

**示例**：
```python
y = encode_to_hifloat8(x_fp16)
y = encode_to_hifloat8(x_bf16)
```

---

### decode_from_hifloat8(x: Tensor, dtype: dtype = torch.bfloat16) -> Tensor

将 HiFloat8 格式解码为 FP16/BF16 张量。

**参数**：
- `x`：HiFloat8 编码张量（`torch.uint8`），任意 shape
- `dtype`：输出类型，默认 `torch.bfloat16`，支持 `float16` / `bfloat16`

**返回**：
- 解码后的张量，shape 与输入相同

**异常**：
- `RuntimeError`：输入 dtype 不是 uint8，或输出 dtype 不支持

**示例**：
```python
z = decode_from_hifloat8(y)                   # 默认输出 BF16
z = decode_from_hifloat8(y, torch.float16)    # 指定输出 FP16
```

## 文件结构

```
hifloat8_cast/
├── op_kernel/
│   ├── hifloat8_cast_kernel.cpp   device 端 kernel（LUT encode + decode，半空间 LUT 优化）
│   └── hifloat8_cast_tiling.h     TilingData 结构体定义（tileLength 由 host 运行时填入）
├── op_host/
│   ├── hifloat8_cast_op.cpp       host 端实现（Torch binding，CPU 端 LUT 预计算与缓存）
│   ├── ops.h                      PyTorch C++ 扩展函数声明
│   └── register.cpp               TORCH_LIBRARY 注册
├── python/
│   └── hifloat8_cast/
│       ├── __init__.py            Python 包入口（加载 .so、注册算子）
│       └── ops.py                 Python 接口实现
├── tests/
│   └── test_hifloat8_cast.py      Python API 综合测试
├── docs/
├── CMakeLists.txt
└── README.md
```

## 环境依赖

| SOC | 平台 | 标称 UB / core | CANN 实测可用 UB |
|-----|------|--------------|----------------|
| `ascend910b` | A2（910B1 / 910B2 / 910B3） | 256 KB | ~192 KB |
| `ascend910_93` | A3（910_93） | 512 KB | ~384 KB |
| `ascend950` | A5 | 512 KB+ | — |

> CANN 运行时会保留部分 UB，`tileLength` 在每次调用时由 `GetCoreMemSize()` 查询实际可用量并计算，无需手动指定。

- CANN ≥ 9.0.0
- Python ≥ 3.9
- PyTorch + torch_npu（适配对应 CANN 版本）

## 编译

### 方式一：统一打包（推荐）

```bash
cd amct_ops
bash ops_build.sh [--soc <soc>] [hifloat8_cast]
```

| `--soc` | 平台 | `--npu-arch` | 默认 |
|---------|------|-------------|------|
| `ascend910b` | A2（910B1/B2/B3，UB 256 KB） | `dav-2201` | ✓ |
| `ascend910_93` | A3（910_93，UB 512 KB） | `dav-2201` | |
| `ascend950` | A5（UB 512 KB+） | `dav-3510` | |

> A2/A3 共用同一 ISA（`dav-2201`），编译产物相同；UB 大小差异由运行时平台 API 区分，自动选择最优 `tileLength`。

```bash
bash ops_build.sh                                   # 全部算子，A2
bash ops_build.sh --soc ascend910_93                # 全部算子，A3
bash ops_build.sh hifloat8_cast                     # 仅本算子，A2
bash ops_build.sh --soc ascend950 hifloat8_cast     # 仅本算子，A5
pip install dist/amct_ops-*.whl
```

### 方式二：独立编译（开发调试）

通过 `-DNPU_ARCH` 指定目标平台（默认 A2）：

| 平台 | `-DNPU_ARCH` |
|------|-------------|
| A2（ascend910b，256 KB） | `dav-2201`（默认） |
| A3（ascend910_93，512 KB） | `dav-2201`（同 A2） |
| A5（ascend950） | `dav-3510` |

```bash
cd amct_ops/hifloat8_cast
source $ASCEND_HOME_PATH/set_env.sh
mkdir -p build && cd build

cmake .. -DNPU_ARCH=dav-2201   # A2 / A3（默认）
# cmake .. -DNPU_ARCH=dav-3510  # A5

make -j8
```

编译产物：`build/libhifloat8_cast_ops.so`

## 使用

```python
import torch
import torch_npu
from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8

# 编码（FP16/BF16 → HiFloat8）
x = torch.randn(1024, 256, dtype=torch.bfloat16, device='npu')
y = encode_to_hifloat8(x)      # shape [1024, 256], dtype uint8

# 解码（HiFloat8 → FP16/BF16）
z = decode_from_hifloat8(y, torch.bfloat16)  # shape [1024, 256], dtype bfloat16
```

也可直接通过 `torch.ops.amct` 调用底层算子：

```python
y = torch.ops.amct.encode_to_hifloat8(x)
z = torch.ops.amct.decode_from_hifloat8(y, torch.bfloat16)
```

## 性能

**测试平台**：Ascend 910B3 (A2，ascend910b)，CANN 9.0.0  
**吞吐量定义**：(输入字节 + 输出字节) / 耗时（100 次迭代均值，20 次预热，NPU synchronize 计时）  
**推荐场景**：数据量 ≥ 256K 时 NPU 优势显著

### BF16 ↔ HiFloat8

| 数据大小 | Encode (ms) | 吞吐 (MB/s) | Decode (ms) | 吞吐 (MB/s) |
|---------|------------|------------|------------|------------|
| 4K      | 0.187      | 66         | 0.176      | 70         |
| 16K     | 0.186      | 264        | 0.182      | 270        |
| 64K     | 0.186      | 1060       | 0.180      | 1093       |
| 256K    | 0.266      | 2960       | 0.275      | 2864       |
| 1M      | 0.842      | 3735       | 0.847      | 3714       |
| 4M      | 3.138      | 4009       | 3.103      | 4056       |

### FP16 ↔ HiFloat8

| 数据大小 | Encode (ms) | 吞吐 (MB/s) | Decode (ms) | 吞吐 (MB/s) |
|---------|------------|------------|------------|------------|
| 4K      | 0.203      | 61         | 0.178      | 69         |
| 16K     | 0.200      | 246        | 0.185      | 265        |
| 64K     | 0.199      | 987        | 0.189      | 1043       |
| 256K    | 0.259      | 3031       | 0.266      | 2959       |
| 1M      | 0.837      | 3759       | 0.859      | 3662       |
| 4M      | 3.118      | 4035       | 3.112      | 4044       |

> 小数据（< 256K）吞吐较低，但已通过核数优先策略优化避免过多核加载LUT开销。  
> 大数据（≥ 4M）encode/decode 吞吐均超过 4 GB/s，接近 HBM 带宽上限。  
> tileLength 由运行时根据平台实际 UB 大小和核数动态计算。

## 精度

精度验证通过以下方式进行：
- **Encode 一致性**：随机采样 100K 个 FP16/BF16 值，编码后解码，与 Python 参考实现逐 bit 对比
- **Decode 完备性**：枚举全部 256 个 HiFloat8 编码，分别解码为 FP16 和 BF16，与 Python 参考实现逐 bit 对比
- **边界/特殊值**：±0、±Inf、NaN、最大值、最小非规格化数等边界情况

| 测试场景 | 结果 |
|---------|------|
| 随机 FP16/BF16 Encode（100K 样本） | ✓ 逐 bit 一致 |
| 随机 FP16 Decode（100K 样本） | ✓ 逐 bit 一致 |
| 边界值 Encode（FP16 13 个，BF16 13 个） | ✓ 逐个一致 |
| 特殊值：±0 / ±Inf / NaN（FP16 编码） | ✓ 编码字节一致 |
| 全 256 个 HiF8 值解码（FP16 + BF16） | ✓ 完全一致 |

> 注：FP16 subnormal 编解码（含 HiFloat8 非规格化值映射到 FP16 非规格化值）已全部验证正确。

## 测试

```bash
# Python API 综合测试（类型推断、边界值、特殊值、非规格化数）
python3 scripts/test_python_api.py

# 性能测试（NPU 多核，FP16 + BF16）
python3 scripts/perf_test_multicore.py
```

以下脚本需要额外构建 CPU 参考实现（`amct_pytorch/experimental/hifloat8/build.sh`），用于 CPU vs NPU 对比：

```bash
# 精度验证（CPU vs NPU 逐 bit 对比）
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
    python3 scripts/accuracy_compare_cpu_npu.py

# FP16 性能对比（NPU vs CPU）
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
    python3 scripts/perf_compare_cpu_npu.py

# BF16 性能对比（NPU vs CPU）
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
    python3 scripts/perf_bf16_compare.py
```

> `LD_PRELOAD` 解决 libgomp TLS 问题，仅加载 CPU 参考实现时需要。

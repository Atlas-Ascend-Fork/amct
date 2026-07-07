# hifloat8_cast — HiFloat8 数据类型转换算子

HiFloat8 数据类型转换算子，支持 FP16/BF16 ↔ HiFloat8 双向转换。

## 功能介绍

支持 4 种转换模式，底层统一由 `KernelHiFloat8CastLut` 实现：

| castMode | 转换 | 底层 LUT |
|----------|------|---------|
| 0 | FP16 → HiFloat8 | 32768-entry 半空间 UB-LUT（32 KB） |
| 1 | BF16 → HiFloat8 | 32768-entry 半空间 UB-LUT（32 KB） |
| 2 | HiFloat8 → FP16 | 256-entry UB-LUT（512 B） |
| 3 | HiFloat8 → BF16 | 256-entry UB-LUT（512 B） |

> Python 接口根据输入 dtype 自动选择 castMode，无需手动指定。

## 接口说明

### encode_to_hifloat8(x: Tensor) -> Tensor

将 FP16/BF16 张量编码为 HiFloat8 格式。

**参数**：
- `x`：输入张量（NPU 设备），dtype 为 `torch.float16` 或 `torch.bfloat16`，任意非空 shape

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
- `x`：HiFloat8 编码张量（`torch.uint8`），任意非空 shape
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

## 目录结构

```
hifloat8_cast/
├── op_kernel/
│   ├── hifloat8_cast_kernel.cpp   # device 端 kernel（LUT encode + decode，半空间 LUT 优化）
│   └── hifloat8_cast_tiling.h     # TilingData 结构体定义（tileLength 由 host 运行时填入）
├── op_extension/
│   ├── hifloat8_cast_torch.cpp    # PyTorch host 实现（CPU 端 LUT 预计算与缓存，调用 ASC 生成的 host stub）
│   ├── ops.h                      # PyTorch C++ 扩展函数声明
│   └── register.cpp               # TORCH_LIBRARY 注册
├── python/
│   └── hifloat8_cast/
│       ├── __init__.py            # Python 包入口（加载 .so、注册算子）
│       └── ops.py                 # Python 接口实现
├── CMakeLists.txt                 # CMake 构建配置
└── README.md                      # 算子说明文档
```

## 环境依赖

| SOC | 平台 | 标称 UB / core | CANN 实测可用 UB |
|-----|------|--------------|----------------|
| `ascend910b` | Atlas A2系列产品 | 256 KB | ~192 KB |
| `ascend910_93` |Atlas A3系列产品 | 512 KB | ~384 KB |
| `ascend950` | Ascend950PR/Ascend950DT | 512 KB+ | — |

> CANN 运行时会保留部分 UB，`tileLength` 在每次调用时由 `GetCoreMemSize()` 查询实际可用量并计算，无需手动指定。

- CANN 9.0.0
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
> A5 构建要求当前 CANN 编译器支持 `dav-3510`。如果 `bisheng` 报 `Unsupported NPU architecture or soc`，需要更换支持 A5 目标的 CANN 编译环境。

```bash
bash ops_build.sh                                   # 全部算子，默认平台
bash ops_build.sh --soc ascend910_93                # 全部算子，指定平台
bash ops_build.sh hifloat8_cast                     # 指定算子，默认平台
bash ops_build.sh --soc ascend950 hifloat8_cast     # 指定算子，指定平台
pip install dist/amct_ops-*.whl
```

### 方式二：单算子独立编译（开发调试）

针对单个算子进行独立编译，适用于开发调试场景。

通过 `-DNPU_ARCH` 指定目标平台（默认 A2）：

| 平台 | `-DNPU_ARCH` |
|------|-------------|
| Atlas A2系列产品 | `dav-2201`（默认） |
| Atlas A3系列产品 | `dav-2201`（同 A2） |
| Ascend950PR/Ascend950DT | `dav-3510` |

```bash
cd amct_ops/hifloat8_cast
source $ASCEND_HOME_PATH/set_env.sh
mkdir -p build && cd build

cmake .. -DNPU_ARCH=dav-2201   # A2 / A3（默认）
# cmake .. -DNPU_ARCH=dav-3510  # A5

make -j8
```

编译产物：`build/libhifloat8_cast_ops.so`

#### 已知 CMake 告警

使用 pip 安装的 PyTorch 构建时，`find_package(Torch)` 可能输出如下告警：

```text
static library kineto_LIBRARY-NOTFOUND not found.
```

该告警来自 PyTorch 自带的 `TorchConfig.cmake`，表示未找到 Kineto profiler 的静态库。`hifloat8_cast` 不依赖 PyTorch profiler / Kineto 能力；只要 CMake configure、编译和链接成功，该告警可以忽略。

## 使用示例

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

## 性能验证

**测试平台**：Ascend 910B3 (A2，ascend910b)，CANN 9.0.0  
**吞吐量定义**：(输入字节 + 输出字节) / 耗时（100 次迭代均值，10 次预热，NPU synchronize 计时）  
**推荐场景**：数据量 ≥ 256K 时 NPU 优势显著

### BF16 ↔ HiFloat8

| 数据大小 | Encode (ms) | 吞吐 (MB/s) | Decode (ms) | 吞吐 (MB/s) |
|---------|------------|------------|------------|------------|
| 1K      | 0.207      | 14.9       | 0.228      | 13.5       |
| 4K      | 0.215      | 57.2       | 0.219      | 56.0       |
| 16K     | 0.215      | 228.4      | 0.255      | 192.9      |
| 64K     | 0.215      | 914.0      | 0.290      | 677.6      |
| 256K    | 0.289      | 2723.9     | 0.293      | 2679.8     |
| 1M      | 0.867      | 3627.2     | 0.863      | 3645.7     |
| 4M      | 3.133      | 4016.7     | 3.111      | 4044.3     |
| 16M     | 12.236     | 4113.4     | 12.152     | 4141.7     |

### FP16 ↔ HiFloat8

| 数据大小 | Encode (ms) | 吞吐 (MB/s) | Decode (ms) | 吞吐 (MB/s) |
|---------|------------|------------|------------|------------|
| 1K      | 0.208      | 14.8       | 0.224      | 13.7       |
| 4K      | 0.216      | 56.9       | 0.223      | 55.1       |
| 16K     | 0.215      | 228.6      | 0.257      | 191.6      |
| 64K     | 0.217      | 904.1      | 0.289      | 680.9      |
| 256K    | 0.288      | 2734.8     | 0.293      | 2687.8     |
| 1M      | 0.867      | 3628.1     | 0.863      | 3645.2     |
| 4M      | 3.130      | 4019.6     | 3.121      | 4032.2     |
| 16M     | 12.236     | 4113.4     | 12.148     | 4143.1     |

### Roundtrip（Encode + Decode）

| 数据大小 | FP16 (ms) | 吞吐 (MB/s) | BF16 (ms) | 吞吐 (MB/s) |
|---------|----------|------------|----------|------------|
| 1K      | 0.389    | 10.5       | 0.392    | 10.5       |
| 4K      | 0.392    | 41.8       | 0.393    | 41.7       |
| 16K     | 0.394    | 166.2      | 0.390    | 168.1      |
| 64K     | 0.388    | 674.8      | 0.389    | 673.7      |
| 256K    | 0.548    | 1911.9     | 0.554    | 1891.6     |
| 1M      | 1.714    | 2446.7     | 1.716    | 2444.9     |
| 4M      | 6.246    | 2686.2     | 6.249    | 2684.8     |
| 16M     | 24.393   | 2751.1     | 24.401   | 2750.3     |

> 小数据（< 256K）吞吐较低，但已通过核数优先策略优化避免过多核加载LUT开销。  
> 大数据（≥ 4M）encode/decode 吞吐均超过 4 GB/s，接近 HBM 带宽上限。  
> tileLength 由运行时根据平台实际 UB 大小和核数动态计算。

## 精度验证

精度验证通过以下方式进行：
- **NPU API 综合测试**：类型推断、边界值、特殊值、非规格化数、全 256 个 HiFloat8 decode 完备性，见 `tests/amct_ops/test_hifloat8_cast.py`
- **边界/特殊值**：±0、±Inf、NaN、最大值、最小非规格化数等边界情况

| 测试场景 | 结果 |
|---------|------|
| 随机 FP16/BF16 roundtrip | ✓ 相对误差在 HiFloat8 精度范围内 |
| 边界值 roundtrip | ✓ 覆盖 0、正负数、小数、较大值 |
| 特殊值：±0 / ±Inf / NaN | ✓ 编码字节和解码语义符合规范 |
| 全 256 个 HiF8 值解码（FP16 + BF16） | ✓ 仅 0x80 为 NaN 编码 |

> 注：FP16 subnormal 编解码（含 HiFloat8 非规格化值映射到 FP16 非规格化值）已全部验证正确。

## 测试方法

```bash
# 在仓库根目录执行。先构建 amct_ops，再通过 staging 运行测试。
bash amct_ops/ops_build.sh hifloat8_cast

# Python API 综合测试（类型推断、边界值、特殊值、非规格化数）
PYTHONPATH=amct_ops/staging python3 -m unittest tests.amct_ops.test_hifloat8_cast
```

也可以先安装 wheel 后再执行测试：

```bash
pip install amct_ops/dist/amct_ops-*.whl
python3 -m unittest tests.amct_ops.test_hifloat8_cast
```

更多测试执行说明见 `tests/amct_ops/README.md`。

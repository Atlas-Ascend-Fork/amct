# hifloat4_cast — HiFloat4 数据类型转换算子

HiFloat4 数据类型转换算子，支持 FP16/BF16 → HiFloat4 → FP16/BF16 fake-quant 仿真（64 元素块缩放，S1P2 尾数）。

## 功能介绍

HiFloat4（hifx4）是 4-bit 块缩放浮点格式：每元素 **S1P2**（1 符号 + 2-bit 尾数，8 档幅值），配**每 64 元素共享的 3 级 scale**（L1 `scale_factor` E6M2 覆盖 64 块、L2 覆盖每 8、L3 覆盖每 4）。

底层由 `hifx_kernel`（fp32）/ `hifx_kernel_bf16`（bf16）实现一次 FP→HiF4→FP 仿真往返。打包/解包由 Python 侧 `hif4_pack` / `hif4_unpack`（`hifp_impl.py`）完成，fake-quant 和 pack 共享同一套编解码逻辑，保证评估路径与部署权重永远一致。

## 接口说明

### hifloat4_fake_quant(x: Tensor, qdim: int = -1) -> Tensor

将 FP16/BF16 张量沿 `qdim` 维度分块（每 64 元素一个共享 scale 块），做一次 FP→HiF4→FP 仿真往返。

**参数**：
- `x`：输入张量（NPU 设备），dtype 为 `torch.float16` 或 `torch.bfloat16`，任意非空 shape
- `qdim`：分块维度，默认 `-1`（最后一维）

**返回**：
- 同 shape/dtype 的张量（注入了 HiF4 量化误差）

**异常**：
- `RuntimeError`：dtype 不是 float16 / bfloat16，或输入不在 NPU 上

**示例**：
```python
from amct_ops.hifloat4_cast import hifloat4_fake_quant
y = hifloat4_fake_quant(x)                      # 沿最后一维分块
y = hifloat4_fake_quant(w, qdim=1)              # Linear 权重 [out, in] 沿 in
y = torch.ops.amct.hifloat4_fake_quant(x, -1)   # torch.ops 等价调用
```

host 会把 `qdim` 移到最后一维、补零到 64 的整数倍（补零不影响块内 max）、调 kernel，再切回、还原维序。

## 目录结构

```
hifloat4_cast/
├── op_kernel/
│   ├── hifloat4_cast_kernel.cpp   # device 端 kernel（FP→HiF4→FP 仿真往返）
│   └── tensorutils.h              # 张量工具函数（补零、切回、维度变换）
├── op_extension/
│   ├── hifloat4_cast_torch.cpp    # PyTorch host 实现（qdim 预处理，调用 kernel）
│   ├── ops.h                      # PyTorch C++ 扩展函数声明
│   └── register.cpp               # TORCH_LIBRARY 注册
├── python/
│   └── hifloat4_cast/
│       ├── __init__.py            # Python 包入口（加载 .so、注册算子）
│       └── ops.py                 # Python 接口实现
├── CMakeLists.txt                 # CMake 构建配置
├── README.md                      # 算子说明文档（中文）
└── README_en.md                   # 算子说明文档（英文）
```

## 打包格式

```
scale : uint8 [M, N // 64, 4]     每 64-group 一条 4-byte 记录
    byte 0 : E6M2                  level-1 scale factor，E6M2 = 8 bits
    byte 1 : E1_8  (8 x 1 bit)    per-8-subgroup exponent bit
    byte 2 : E1_16 high 8 bits    per-4-subgroup exponent bit，块 0..7
    byte 3 : E1_16 low  8 bits    块 8..15
value : uint8 [M, N // 2]         相邻两个 4-bit code 打包为一个 byte
    每 nibble = (sign << 3) | code,  code = round(|in_grp| * 4) in 0..7
    低位优先：元素0放在低4位，元素1放在高4位
```

## 环境依赖

| SOC | 平台 | 标称 UB / core | CANN 实测可用 UB |
|-----|------|--------------|----------------|
| `ascend910b` | Atlas A2系列产品 | 256 KB | ~192 KB |
| `ascend910_93` | Atlas A3系列产品 | 512 KB | ~384 KB |
| `ascend950` | Ascend950PR/Ascend950DT | 512 KB+ | — |

> CANN 运行时会保留部分 UB，`tileLength` 在每次调用时由 `GetCoreMemSize()` 查询实际可用量并计算，无需手动指定。

- CANN 9.0.0
- Python ≥ 3.9
- PyTorch + torch_npu（适配对应 CANN 版本）

## 编译

### 方式一：统一打包（推荐）

```bash
cd amct_ops
bash ops_build.sh [--soc <soc>] [hifloat4_cast]
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
bash ops_build.sh hifloat4_cast                     # 指定算子，默认平台
bash ops_build.sh --soc ascend950 hifloat4_cast     # 指定算子，指定平台
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
cd amct_ops/hifloat4_cast
source $ASCEND_HOME_PATH/set_env.sh
mkdir -p build && cd build

cmake .. -DNPU_ARCH=dav-2201   # A2 / A3（默认）
# cmake .. -DNPU_ARCH=dav-3510  # A5

make -j8
```

编译产物：`build/libhifloat4_cast_ops.so`

#### 已知 CMake 告警

使用 pip 安装的 PyTorch 构建时，`find_package(Torch)` 可能输出如下告警：

```text
static library kineto_LIBRARY-NOTFOUND not found.
```

该告警来自 PyTorch 自带的 `TorchConfig.cmake`，表示未找到 Kineto profiler 的静态库。`hifloat4_cast` 不依赖 PyTorch profiler / Kineto 能力；只要 CMake configure、编译和链接成功，该告警可以忽略。

## 使用示例

```python
import torch
import torch_npu
from amct_ops.hifloat4_cast import hifloat4_fake_quant

# Fake-quant 仿真（FP16/BF16 → HiF4 → FP16/BF16）
x = torch.randn(1024, 256, dtype=torch.bfloat16, device='npu')
y = hifloat4_fake_quant(x)                     # shape [1024, 256], dtype bfloat16

# 指定分块维度
w = torch.randn(4096, 1024, dtype=torch.float16, device='npu')
y = hifloat4_fake_quant(w, qdim=1)             # 沿 in_features 分块
```

也可直接通过 `torch.ops.amct` 调用底层算子：

```python
y = torch.ops.amct.hifloat4_fake_quant(x, -1)
```

## 精度验证

精度验证通过以下方式进行：
- **NPU API 综合测试**：类型推断、边界值、特殊值、非规格化数，见 `tests/amct_ops/test_hifloat4_cast.py`
- **kernel vs CPU 参考**：逐比特比较 NPU kernel 与 Python CPU 实现（`hif4_pack` / `hif4_unpack`），误差 ~0

| 测试场景 | 结果 |
|---------|------|
| 随机 FP16/BF16 fake-quant roundtrip | ✓ 相对误差在 HiFloat4 精度范围内 |
| pack/unpack 往返 | ✓ shape 一致，逐比特恢复 |
| NPU kernel vs CPU fallback | ✓ 逐比特一致（max abs diff < 1e-6） |

## 测试方法

```bash
# 在仓库根目录执行。先构建 amct_ops，再通过 staging 运行测试。
bash amct_ops/ops_build.sh hifloat4_cast

# Python API 综合测试（类型推断、边界值、特殊值、非规格化数）
PYTHONPATH=amct_ops/staging python3 -m unittest tests.amct_ops.test_hifloat4_cast
```

也可以先安装 wheel 后再执行测试：

```bash
pip install amct_ops/dist/amct_ops-*.whl
python3 -m unittest tests.amct_ops.test_hifloat4_cast
```

更多测试执行说明见 `tests/amct_ops/README.md`。

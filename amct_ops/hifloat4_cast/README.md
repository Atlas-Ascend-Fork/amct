# hifloat4_cast — HiFloat4 数据类型转换算子

HiFloat4 数据类型转换算子，支持 FP16/BF16 → HiFloat4 → FP16/BF16 fake-quant 仿真（64 元素块缩放，S1P2 尾数）。

## 功能介绍

HiFloat4（HiF4）是一种 4-bit 块缩放浮点格式：每个元素采用 S1P2 表示（1-bit 符号位 + 2-bit 尾数，共 8 档幅值），并使用每 64 个元素共享的三级 scale（L1 `scale_factor` 采用 E6M2，覆盖 64 块；L2 每 8 个元素一组；L3 每 4 个元素一组）。

该算子仅实现 fake-quant 仿真，即执行一次 FP → HiF4 → FP 的量化与反量化过程。输出为与输入具有相同 shape 和 dtype 的浮点张量。

根据输入 dtype 调用对应的底层 kernel，两个 kernel 内部均使用 fp32 进行计算：

| 输入 dtype | 底层 kernel          | 内部计算                                                              |
| -------- | ------------------ | ----------------------------------------------------------------- |
| BF16     | `hifx_kernel_bf16` | 输入加载后转换为 fp32，仅在指定计算位置（scale factor、倒数、组内值）进行 bf16 舍入，输出时转换回 bf16 |
| FP16     | `hifx_kernel`      | Host 侧先转换为 fp32，计算完成后将输出转换回 fp16                                  |

## 接口说明

### hifloat4_fake_quant(x: Tensor, qdim: int = -1) -> Tensor

将 FP16/BF16 张量沿 `qdim` 维度进行分块，每 64 个元素共享一个 scale 块，并执行一次 FP → HiF4 → FP 的 fake-quant 仿真。

参数：

* `x`：输入张量，位于 NPU 设备上，dtype 为 `torch.float16` 或 `torch.bfloat16`，支持任意非空 shape
* `qdim`：分块维度，默认值为 `-1`，即最后一维

返回：

* 与输入具有相同 shape 和 dtype 的张量，其中包含 HiF4 量化误差

异常：

* `RuntimeError`：输入 dtype 不是 float16 / bfloat16，或输入张量不在 NPU 设备上
* `RuntimeError`：`qdim` 维度长度不是 64 的倍数

示例：

```python
from amct_ops.hifloat4_cast import hifloat4_fake_quant

y = hifloat4_fake_quant(x)                      # 沿最后一维分块
y = hifloat4_fake_quant(w, qdim=1)              # Linear 权重 [out, in] 沿 in
y = torch.ops.amct.hifloat4_fake_quant(x, -1)   # torch.ops 等价调用
```

Host 侧会先将 `qdim` 移至最后一维；该维长度必须是 64 的倍数（否则报错）。随后补零至 512 的整数倍（补零不会影响块内 max）并调用 kernel，计算完成后截回原始长度并恢复原有维度顺序。

## 目录结构

```text
hifloat4_cast/
├── op_kernel/
│   ├── hifloat4_cast_kernel.cpp   # Device 端 kernel（FP→HiF4→FP 仿真往返）
│   └── tensorutils.h              # Ascend C 工具函数（kernel 侧张量操作）
├── op_extension/
│   ├── hifloat4_cast_torch.cpp    # PyTorch Host 实现（qdim 预处理并调用 kernel）
│   ├── ops.h                      # PyTorch C++ 扩展函数声明
│   └── register.cpp               # TORCH_LIBRARY 注册
├── python/
│   └── hifloat4_cast/
│       ├── __init__.py            # Python 包入口（加载 .so 并注册算子）
│       └── ops.py                 # Python 接口实现
├── CMakeLists.txt                 # CMake 构建配置
├── README.md                      # 算子说明文档（中文）
└── README_en.md                   # 算子说明文档（英文）
```

## 环境依赖

* CANN 9.0.0
* Python ≥ 3.9
* PyTorch + torch_npu（需适配对应的 CANN 版本）

## 编译

### 方式一：统一打包（推荐）

```bash
cd amct_ops
bash ops_build.sh [--soc <soc>] [hifloat4_cast]
```

| `--soc`        | 平台                        | `--npu-arch` | 默认 |
| -------------- | ------------------------- | ------------ | -- |
| `ascend910b`   | A2（910B1/B2/B3，UB 256 KB） | `dav-2201`   | ✓  |
| `ascend910_93` | A3（910_93，UB 512 KB）      | `dav-2201`   |    |
| `ascend950`    | A5（UB 512 KB+）            | `dav-3510`   |    |

> A2/A3 使用相同的 ISA（`dav-2201`），因此编译产物相同。
> A5 构建要求当前 CANN 编译器支持 `dav-3510`。如果 `bisheng` 报错 `Unsupported NPU architecture or soc`，需要更换为支持 A5 目标的 CANN 编译环境。

```bash
bash ops_build.sh                                   # 全部算子，默认平台
bash ops_build.sh --soc ascend910_93                # 全部算子，指定平台
bash ops_build.sh hifloat4_cast                     # 指定算子，默认平台
bash ops_build.sh --soc ascend950 hifloat4_cast     # 指定算子，指定平台
pip install dist/amct_ops-*.whl
```

### 方式二：单算子独立编译（开发调试）

可对单个算子进行独立编译，适用于开发和调试场景。

通过 `-DNPU_ARCH` 指定目标平台，默认使用 A2：

| 平台                      | `-DNPU_ARCH`     |
| ----------------------- | ---------------- |
| Atlas A2 系列产品           | `dav-2201`（默认）   |
| Atlas A3 系列产品           | `dav-2201`（同 A2） |
| Ascend950PR/Ascend950DT | `dav-3510`       |

```bash
cd amct_ops/hifloat4_cast
source $ASCEND_HOME_PATH/set_env.sh
mkdir -p build && cd build

cmake .. -DNPU_ARCH=dav-2201   # A2 / A3（默认）
# cmake .. -DNPU_ARCH=dav-3510  # A5

make -j8
```

编译产物位于：

```text
build/libhifloat4_cast_ops.so
```

#### 已知 CMake 告警

使用 pip 安装的 PyTorch 进行构建时，`find_package(Torch)` 可能输出以下告警：

```text
static library kineto_LIBRARY-NOTFOUND not found.
```

该告警来自 PyTorch 自带的 `TorchConfig.cmake`，表示未找到 Kineto profiler 的静态库。`hifloat4_cast` 不依赖 PyTorch profiler / Kineto 功能，因此只要 CMake 配置、编译和链接均成功，该告警可以忽略。

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

也可以直接通过 `torch.ops.amct` 调用底层算子：

```python
y = torch.ops.amct.hifloat4_fake_quant(x, -1)
```

## 精度验证

精度验证方式如下，具体见 `tests/amct_ops/test_hifloat4_cast.py`：

| 测试场景                         | 结果                                     |
| ---------------------------- | -------------------------------------- |
| 随机 BF16/FP16 fake-quant 往返仿真 | ✓ 与 CPU 参考实现逐元素一致（max abs diff < 1e-6） |
| 非 64 对齐维度                    | ✓ 抛 RuntimeError（fake-quant 与 pack 行为一致）      |
| 零输入                          | ✓ 无 NaN，输出全 0                          |

## 测试方法

```bash
# 在仓库根目录执行。先构建 amct_ops，再通过 staging 运行测试。
bash amct_ops/ops_build.sh hifloat4_cast

# NPU kernel 与 CPU 参考实现逐元素一致性验证 + shape 回归测试
PYTHONPATH=amct_ops/staging python3 -m unittest tests.amct_ops.test_hifloat4_cast
```

也可以先安装 wheel，再执行测试：

```bash
pip install amct_ops/dist/amct_ops-*.whl
python3 -m unittest tests.amct_ops.test_hifloat4_cast
```

更多测试执行说明见 `tests/amct_ops/README.md`。

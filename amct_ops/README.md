# amct_ops — AMCT NPU 自定义算子

本目录存放昇腾 NPU 自定义算子，补齐 PyTorch / CANN 尚未原生支持的模型压缩相关算子（量化、稀疏、低比特类型转换等）。

> **命名说明**：`amct` = AMCT 项目归属，`ops` = 算子，完整语义为「AMCT 项目的 NPU 算子」。

## 背景与定位

AMCT 的主体能力在 `amct_pytorch/` 中，那里更关注量化算法、压缩流程编排和模型结构改写。随着低比特量化能力扩展，部分数据类型转换和量化辅助计算需要直接落到 NPU 上执行，而 PyTorch / torch_npu 的通用算子并不总是能覆盖这些场景。`amct_ops` 因此作为 AMCT 的 NPU 自定义算子层存在，用来承载这类更贴近硬件实现的能力。

把这些算子独立放在 `amct_ops` 中，可以让算法流程和硬件实现保持清晰边界。`amct_pytorch` 只需要通过 Python 接口或 `torch.ops.amct` 调用算子，不需要关心 Ascend C kernel、PyTorch C++ extension、CMake 编译和 wheel 打包细节；算子本身也可以按独立模块开发、构建和测试。这样后续新增低比特类型、补充量化辅助算子或调整 NPU 实现时，不会把主算法目录变成混合了算法、kernel 和构建逻辑的复杂目录。

在使用方式上，`amct_ops` 提供的是可独立安装的 PyTorch 扩展包。每个算子在 C++ 侧注册到统一的 `amct` 命名空间，在 Python 侧提供 `amct_ops.<op>` 包装接口。构建脚本会把 Python 接口和编译得到的 `.so` 汇集成平台相关 wheel，供 AMCT 流程或单独的 PyTorch 脚本直接导入使用。

## 支持的算子

| 算子 | 说明 | Python 接口 |
|------|------|-------------|
| [`hifloat8_cast`](./hifloat8_cast/) | FP16 / BF16 ↔ HiFloat8 双向转换 | `encode_to_hifloat8(x)`<br>`decode_from_hifloat8(x, dtype)` |

## 与 `amct_pytorch/` 的分工

| 维度 | `amct_pytorch/` | `amct_ops/` |
|------|-----------------|------------|
| 关注点 | 压缩算法与流程编排 | 单算子的硬件实现 |
| 语言 | Python | Ascend C kernel + C++ binding + Python 接口 |
| 产物 | Python 包 | `.so`（PyTorch 扩展）|
| 复用性 | 绑定 AMCT 流程 | 独立的 PyTorch 扩展，不强依赖 AMCT 主流程 |

## 目录结构

| 路径 | 说明 |
|------|------|
| `hifloat8_cast/` | HiFloat8 转换算子源码，包含 kernel、PyTorch binding、Python 接口和算子说明 |
| `ops_build.sh` | 统一构建入口，负责编译算子、汇集 Python 包并生成 wheel |
| `setup.py` | `amct_ops` wheel 打包配置，基于构建脚本生成的 `staging/` 目录打包 |
| `ops_init.py` | 打包时复制为 `amct_ops/__init__.py`，提供统一包入口 |

构建过程中会生成 `build/`、`dist/`、`staging/`、`<op>/build/` 等目录，这些是本地构建产物，不需要提交。

## 构建与打包

所有算子通过根目录的统一构建脚本一次性编译并打包为 wheel：

依赖要求：

- CANN 9.0.0
- Python ≥ 3.9
- PyTorch + torch_npu（适配对应 CANN 版本）

```bash
cd amct_ops/
bash ops_build.sh [--soc <soc>] [<算子>]

# --soc 指定目标平台（默认 ascend910b）：
#   ascend910b    A2（910B1/B2/B3）
#   ascend910_93  A3（910_93）
#   ascend950     A5（需当前 CANN 编译器支持 dav-3510）

bash ops_build.sh                                # 全部算子，A2
bash ops_build.sh --soc ascend910_93             # 全部算子，指定A3
bash ops_build.sh hifloat8_cast                  # 指定算子，A2
bash ops_build.sh --soc ascend950 hifloat8_cast  # 指定算子，指定A5
```

产物：`dist/amct_ops-1.0.0-cp*-cp*-linux_<arch>.whl`，其中 `<arch>` 随构建主机自动生成为 `x86_64` 或 `aarch64`，包含所有算子的 Python 包及编译 `.so`。

wheel 文件名中的两个 `cp*` 分别表示 Python 实现/版本标签和 ABI 标签。例如，`cp311-cp311` 表示该 wheel 面向 CPython 3.11，并依赖 CPython 3.11 ABI；`linux_<arch>` 表示构建主机平台架构。

### 已知 CMake 告警

使用 pip 安装的 PyTorch 构建时，`find_package(Torch)` 可能输出如下告警：

```text
static library kineto_LIBRARY-NOTFOUND not found.
```

该告警来自 PyTorch 自带的 `TorchConfig.cmake`，表示未找到 Kineto profiler 的静态库。当前 `amct_ops` 算子不依赖 PyTorch profiler / Kineto 能力；只要 CMake configure、编译和链接成功，该告警可以忽略。

安装：

```bash
pip install dist/amct_ops-*.whl
```

安装后按统一命名空间导入，或通过 `torch.ops.amct` 直接调用：

```python
# 方式一：模块导入（有 IDE 补全和文档字符串）
from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8

y = encode_to_hifloat8(x)                    # FP16/BF16 → uint8
z = decode_from_hifloat8(y)                  # → bfloat16（默认）
z = decode_from_hifloat8(y, torch.float16)   # → float16

# 方式二：torch.ops.amct（与其他 NPU 算子风格一致）
import amct_ops.hifloat8_cast                 # 触发 .so 加载
torch.ops.amct.encode_to_hifloat8(x)
torch.ops.amct.decode_from_hifloat8(y, torch.float16)
```

Python 内省：

```python
import amct_ops
help(amct_ops)                               # 查看所有子模块及接口列表

import amct_ops.hifloat8_cast
help(amct_ops.hifloat8_cast.encode_to_hifloat8)   # 查看单个函数签名和文档
```

## 新增算子

每个算子独占一个子目录，结构参照 `hifloat8_cast/`：

```
<新算子>/
├── op_kernel/              Ascend C kernel（.cpp + tiling.h）
├── op_extension/           PyTorch C++ binding（host stub 调用 + TORCH_LIBRARY 注册）
├── python/<pkg>/           Python 接口（__init__.py）
├── CMakeLists.txt          独立编译入口
└── README.md
```

新增后无需修改 `ops_build.sh` 或 `setup.py`，构建脚本会自动发现 `<op>/python/<pkg>/` 目录并打包。正式测试放在 `tests/amct_ops/` 下，避免把构建产物、性能脚本或依赖额外参考实现的对比工具放入算子源码目录。

## 命名空间约束

**所有算子必须注册到 `amct` 命名空间**——与 `amct_ops` 包名一致，便于调用方区分 AMCT 自定义算子与 `torch_npu` 上游算子。

- C++ 侧：`TORCH_LIBRARY_FRAGMENT(amct, m)` + `TORCH_LIBRARY_IMPL(amct, PrivateUse1, m)`
- Python 侧：`torch.ops.amct.<算子名>` 或模块导入 `from amct_ops.xxx import ...`
- 算子名在 `amct` 内须唯一，新增前先检索 `torch.ops.amct` 是否已存在同名算子。

## 参考
[cann/ops-nn算子开发指导](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/develop/torch_extension_develop_guide.md)

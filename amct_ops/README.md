# amct_ops — AMCT NPU 自定义算子

## 简介

**定位**：
- `amct_ops` 是 AMCT 的 NPU 自定义算子层，负责承载 PyTorch / torch_npu 尚未覆盖的低比特量化、数据类型转换等硬件级算子。
- 与 `amct_pytorch/` 聚焦的量化算法、压缩流程编排不同，`amct_ops` 更贴近底层硬件实现。

**独立优势**：
- **职责清晰**：`amct_pytorch` 通过 Python 接口或 `torch.ops.amct` 调用算子，无需关注 Ascend C kernel、C++ extension、CMake 编译等细节
- **独立开发**：算子可按独立模块开发、构建和测试，避免主算法目录混杂 kernel 和构建逻辑
- **灵活扩展**：新增低比特类型、量化辅助算子或调整 NPU 实现时，互不干扰

**与 `amct_pytorch/` 的分工**：

| 维度 | `amct_pytorch/` | `amct_ops/` |
|------|-----------------|------------|
| 关注点 | 压缩算法与流程编排 | 算子底层实现 |
| 语言 | Python | Ascend C kernel & C++ binding & Python 接口 |
| 产物 | .tar.gz 包（源码压缩包）| wheel 包（含 .so & Python 接口）|
| 复用性 | 绑定 AMCT 流程 | 独立的 PyTorch 扩展，不强依赖 AMCT 主流程 |

**使用方式**：
- 可独立安装的 PyTorch 扩展包（wheel 格式）
- 支持 `amct_ops.<op>` 和 `torch.ops.amct.<op>` 两种接口

## 支持的算子

| 算子 | 说明 | Python 接口 |
|------|------|-------------|
| [`hifloat8_cast`](./hifloat8_cast/) | FP16 / BF16 ↔ HiFloat8 双向转换 | `encode_to_hifloat8(x)`<br>`decode_from_hifloat8(x, dtype)` |

## 目录结构

```
amct_ops/
├── hifloat8_cast/          # HiFloat8 转换算子源码（kernel + binding + Python 接口）
├── ops_build.sh            # 统一构建入口
├── setup.py                # wheel 打包配置
└── ops_init.py             # 打包时复制为 __init__.py，提供包接口和文档
```

构建过程中会生成 `build/`、`dist/`、`staging/`、`<op>/build/` 等目录，这些是本地构建产物，不需要提交。

## 构建与打包

所有算子通过amct_ops根目录的统一构建脚本一次性编译并打包为 wheel。

### 依赖要求

| 依赖 | 版本 |
|------|------|
| Python | >=3.9 |
| PyTorch | 2.7.1 或 2.1.0（需配套 `torch_npu`） |
| GCC / CMake | ≥ 7.3 / ≥ 3.16（推荐 3.20） |
| CANN（Toolkit & Ops） | ≥ 9.0.0（需提前安装 NPU 驱动 / 固件） |

完整环境部署请参见 [快速安装](../docs/quick_install.md)

### 构建命令

```bash
cd amct_ops/
bash ops_build.sh [--soc <soc>] [<算子>]

# --soc 指定目标平台（默认 ascend910b）：
#   ascend910b    A2（Ascend 910B1/B2/B3）
#   ascend910_93  A3（Ascend 910_93）
#   ascend950     A5（Ascend 950，需 CANN 编译器支持 dav-3510）

bash ops_build.sh                                # 全部算子，默认平台
bash ops_build.sh --soc ascend910_93             # 全部算子，指定平台
bash ops_build.sh hifloat8_cast                  # 指定算子，默认平台
bash ops_build.sh --soc ascend950 hifloat8_cast  # 指定算子，指定平台
```

### 构建产物

构建产物位于 `dist/amct_ops-1.0.0-cp*-cp*-linux_<arch>.whl`，其中 `<arch>` 随构建主机自动生成为 `x86_64` 或 `aarch64`，包含所有算子的 Python 包及编译 `.so`。

wheel 文件名中的两个 `cp*` 分别表示 Python 实现/版本标签和 ABI 标签。例如，`cp311-cp311` 表示该 wheel 面向 CPython 3.11，并依赖 CPython 3.11 ABI；`linux_<arch>` 表示构建主机平台架构。

### 安装

```bash
pip install dist/amct_ops-*.whl
```

### 使用示例

安装后两种导入方式：

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

#### Python 内省

```python
import amct_ops
help(amct_ops)                               # 查看所有子模块及接口列表

import amct_ops.hifloat8_cast
help(amct_ops.hifloat8_cast.encode_to_hifloat8)   # 查看单个函数签名和文档
```

#### 已知 CMake 告警

使用 pip 安装的 PyTorch 构建时，`find_package(Torch)` 可能输出如下告警：

```text
static library kineto_LIBRARY-NOTFOUND not found.
```

该告警来自 PyTorch 自带的 `TorchConfig.cmake`，表示未找到 Kineto profiler 的静态库。当前 `amct_ops` 算子不依赖 PyTorch profiler / Kineto 能力；只要 CMake configure、编译和链接成功，该告警可以忽略。

## 新增算子

### 算子目录结构

每个算子独占一个子目录，结构参照 `hifloat8_cast/`：

```
<新算子>/
├── op_kernel/              # Ascend C kernel（.cpp + tiling.h）
├── op_extension/           # PyTorch C++ binding（host stub 调用 + TORCH_LIBRARY 注册）
├── python/<pkg>/           # Python 接口（__init__.py）
├── CMakeLists.txt          # 独立编译入口
└── README.md               # 算子说明文档
```

新增后无需修改 `ops_build.sh` 或 `setup.py`，构建脚本会自动发现 `<op>/python/<pkg>/` 目录并打包。正式测试放在 `tests/amct_ops/` 下，避免把构建产物、性能脚本或依赖额外参考实现的对比工具放入算子源码目录。

### 命名空间约束

**所有算子必须注册到 `amct` 命名空间**——与 `amct_ops` 包名一致，便于调用方区分 AMCT 自定义算子与 `torch_npu` 上游算子。

- C++ 侧：`TORCH_LIBRARY_FRAGMENT(amct, m)` + `TORCH_LIBRARY_IMPL(amct, PrivateUse1, m)`
- Python 侧：`torch.ops.amct.<算子名>` 或模块导入 `from amct_ops.xxx import ...`
- 算子名在 `amct` 内须唯一，新增前先检索 `torch.ops.amct` 是否已存在同名算子。

## 参考资料
[cann/ops-nn算子开发指导](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/develop/torch_extension_develop_guide.md)

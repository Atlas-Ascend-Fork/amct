# amct_ops — AMCT NPU 自定义算子

本目录存放昇腾 NPU 自定义算子，补齐 PyTorch / CANN 尚未原生支持的模型压缩相关算子（量化、稀疏、低比特类型转换等）。

> **命名说明**：`amct` = AMCT 项目归属，`ops` = 算子，完整语义为「AMCT 项目的 NPU 算子」。

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

## 构建与打包

所有算子通过根目录的统一构建脚本一次性编译并打包为 wheel：

```bash
cd amct_ops/
bash ops_build.sh [--soc <soc>] [<算子>]

# --soc 指定目标平台（默认 ascend910b）：
#   ascend910b    A2（910B1/B2/B3）
#   ascend910_93  A3（910_93）
#   ascend950     A5

bash ops_build.sh                                # 全部算子，A2
bash ops_build.sh --soc ascend910_93             # 全部算子，指定A3
bash ops_build.sh hifloat8_cast                  # 指定算子，A2
bash ops_build.sh --soc ascend950 hifloat8_cast  # 指定算子，指定A5
```

产物：`dist/amct_ops-1.0.0-cp*-cp*-linux_aarch64.whl`，包含所有算子的 Python 包及编译 `.so`。

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
├── op_host/                PyTorch C++ binding（TORCH_LIBRARY 注册）
├── python/<pkg>/           Python 接口（__init__.py）
├── tests/                  测试与示例
├── CMakeLists.txt          独立编译入口
└── README.md
```

新增后无需修改 `ops_build.sh` 或 `setup.py`，构建脚本会自动发现 `<op>/python/<pkg>/` 目录并打包。

## 命名空间约束

**所有算子必须注册到 `amct` 命名空间**——与 `amct_ops` 包名一致，便于调用方区分 AMCT 自定义算子与 `torch_npu` 上游算子。

- C++ 侧：`TORCH_LIBRARY_FRAGMENT(amct, m)` + `TORCH_LIBRARY_IMPL(amct, PrivateUse1, m)`
- Python 侧：`torch.ops.amct.<算子名>` 或模块导入 `from amct_ops.xxx import ...`
- 算子名在 `amct` 内须唯一，新增前先检索 `torch.ops.amct` 是否已存在同名算子。

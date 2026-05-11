# npu_ops

AMCT 自定义算子目录，支持 PyTorch / CANN 尚未原生支持的模型压缩相关算子（量化、稀疏、低比特类型转换等）。

```
amct_pytorch/      模型压缩框架（PTQ/QAT、稀疏、图分析等）
     │ 依赖
  npu_ops/         自定义算子（PyTorch C++ Ext + Ascend kernel）  ← 本目录
     │ 依赖
CANN / Ascend NPU
```

## 支持的算子

| 算子 | 说明 | Python 接口 |
|------|------|-------------|
| [`hifloat8_cast`](./hifloat8_cast/) | FP16 / BF16 ↔ HiFloat8 双向转换 | `encode_to_hifloat8(x)`<br>`decode_from_hifloat8(x, dtype)` |

## 与 `amct_pytorch/` 的分工

| 维度 | `amct_pytorch/` | `npu_ops/` |
|------|-----------------|------------|
| 关注点 | 压缩算法与流程编排 | 单算子的硬件实现 |
| 语言 | Python | Ascend C kernel + C++ binding + Python 接口 |
| 产物 | Python 包 | `.so`（PyTorch 扩展）|
| 复用性 | 绑定 AMCT 流程 | 独立的 PyTorch 扩展，不强依赖 AMCT 主流程 |

## 构建与打包

所有算子通过根目录的统一构建脚本一次性编译并打包为 wheel：

```bash
cd npu_ops/
bash ops_build.sh               # 编译全部算子 → 生成 dist/npu_ops-*.whl
bash ops_build.sh hifloat8_cast # 仅编译指定算子
```

产物：`dist/npu_ops-1.0.0-cp*-cp*-linux_aarch64.whl`，包含所有算子的 Python 包及编译 `.so`。

安装：

```bash
pip install dist/npu_ops-*.whl
```

安装后按统一命名空间导入，或通过 `torch.ops.npu` 直接调用：

```python
# 方式一：模块导入（有 IDE 补全和文档字符串）
from npu_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8

y = encode_to_hifloat8(x)                    # FP16/BF16 → uint8
z = decode_from_hifloat8(y)                  # → bfloat16（默认）
z = decode_from_hifloat8(y, torch.float16)   # → float16

# 方式二：torch.ops.npu（与其他 NPU 算子风格一致）
import npu_ops.hifloat8_cast                 # 触发 .so 加载
torch.ops.npu.encode_to_hifloat8(x)
torch.ops.npu.decode_from_hifloat8(y, torch.float16)
```

Python 内省：

```python
import npu_ops
help(npu_ops)                               # 查看所有子模块及接口列表

import npu_ops.hifloat8_cast
help(npu_ops.hifloat8_cast.encode_to_hifloat8)   # 查看单个函数签名和文档
```

## 新增算子

每个算子独占一个子目录，结构参照 `hifloat8_cast/`：

```
<新算子>/
├── op_kernel/              Ascend C kernel（.cpp + tiling.h）
├── op_host/                PyTorch C++ binding（TORCH_LIBRARY 注册）
├── python/<pkg>/           Python 接口（__init__.py）
├── scripts/                测试与示例
├── CMakeLists.txt          独立编译入口
└── README.md
```

新增后无需修改 `ops_build.sh` 或 `setup.py`，构建脚本会自动发现 `<op>/python/<pkg>/` 目录并打包。

## 命名空间约束

**所有算子必须注册到 `npu` 命名空间**——与昇腾上游 `torch_npu` 共用，调用方无需关心算子来源。

- C++ 侧：`TORCH_LIBRARY_FRAGMENT(npu, m)` + `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)`
- Python 侧：`torch.ops.npu.<算子名>`
- 算子名在 `npu` 内须唯一，新增前先检索 `torch.ops.npu` 是否已存在同名算子。

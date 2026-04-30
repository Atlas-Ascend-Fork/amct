# npu_ops

AMCT 自定义算子目录，补齐 PyTorch / CANN 尚未原生支持的模型压缩相关算子（量化、稀疏、低比特类型转换等）。

```
amct_pytorch/      模型压缩框架（PTQ/QAT、稀疏、图分析等）
     │ 依赖
  npu_ops/         自定义算子（PyTorch C++ Ext + Ascend kernel）  ← 本目录
     │ 依赖
CANN / Ascend NPU
```

## 支持的算子

| 算子 | 说明 | Python 接口 |
|------|------|------|
| [`hifloat8_cast`](./hifloat8_cast/) | FP32 / FP16 / BF16 ↔ HiFloat8 双向转换 | `encode_to_hifloat8(x)`<br>`decode_from_hifloat8(x, dtype)` |

## 与 `amct_pytorch/` 的分工

| 维度 | `amct_pytorch/` | `npu_ops/` |
|---|---|---|
| 关注点 | 压缩算法与流程编排 | 单算子的硬件实现 |
| 语言 | Python | Ascend C kernel + C++ binding + Python 接口 |
| 产物 | Python 包 | `.so`（PyTorch 扩展）|
| 复用性 | 绑定 AMCT 流程 | 独立的 PyTorch 扩展，不强依赖 AMCT 主流程 |

## 新增算子骨架

每个算子独占一个子目录，结构参照 `hifloat8_cast/`：

```
<新算子>/
├── op_kernel/         Ascend kernel（.asc / tiling）
├── op_extension/      PyTorch C++ binding（TORCH_LIBRARY 注册）
├── python/<pkg>/      Python 接口
├── scripts/           测试与示例
├── CMakeLists.txt     独立编译入口
└── README.md
```

## 命名空间约束

**所有 `npu_ops/` 下的算子必须注册到 `npu` 命名空间**——与昇腾上游 `torch_npu` 共用，调用方无需关心算子来源。

- 注册与调用：C++ 侧 `TORCH_LIBRARY_FRAGMENT(npu, m)` 与 `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)`，Python 侧 `torch.ops.npu.<算子名>`。
- 算子名在 `npu` 内须唯一，参考 `torch_npu` 上游惯例采用描述性短语（如 `quantize`、`quant_matmul`、`hifloat8_cast`），段数不限但表意清晰；新增前先检索 `torch.ops.npu` 是否已存在同名算子。

## 构建

所有算子构建方式一致（在 `npu_ops/` 目录下执行）：

```bash
source $ASCEND_HOME_PATH/set_env.sh
cd <算子>
mkdir -p build && cd build && cmake .. && make -j
```

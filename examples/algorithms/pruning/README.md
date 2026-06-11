# AMCT 结构化剪枝样例

本样例演示 `amct_pytorch.pruning` 的结构化剪枝接口在三种域上的调用方式：
**dense FFN（剪中间维）/ CNN（剪通道）/ MoE（剪专家）**。样例使用随机初始化的微型模型，
**不下载任何权重、纯 CPU 即可运行**，便于在无联网/无 NPU 环境快速上手。

> 算法说明、压缩比与精度结论详见剪枝包文档
> [`amct_pytorch/pruning/README.md`](../../../amct_pytorch/pruning/README.md)。

## 1 剪枝前提

### 1.1 安装依赖

本样例的依赖见 [requirements.txt](requirements.txt)。仅需 `torch`；在 NPU 上运行时另需与
Python/torch 版本匹配的 `torch_npu` 及已安装的 CANN 包。剪枝接口本身随 amct_pytorch 工具包提供。

### 1.2 模型与数据准备

样例模型与数据由 [src/utils.py](src/utils.py) 用固定随机种子构造（`MiniMLP`/`MiniCNN`/`MiniMoE`），
**无需下载、无需联网**。把它们替换成真实模型与校准数据即可用于实际场景。

### 1.3 剪枝配置

剪枝以 dict 配置直接传入 `prune()`（与 `amct.quantize` 同风格），按域选择方法：

| 域 | 方法 | 说明 |
|:--|:--|:--|
| dense | `low_variance` | 按激活方差剪 FFN 中间维（自动避开注意力投影） |
| dense | `reconstruct` | 剪后用最小二乘重构补偿，recovery ∈ {none, bias, ls} |
| cnn | `variance_channel` | 按激活方差朴素切片通道 |
| cnn | `reconstruct` | 输出重构补偿的通道剪枝（破坏更低） |
| moe | `activation_count` | 按专家激活频次剪专家，同步收缩 gate |
| moe | `mass_variance` | 按专家质量方差剪专家 |

只给 `tolerance` 时则走自动剪枝：在 `ratio_grid` 上二分查找满足容差的最大剪枝率。
守卫式 HYBRID（`guarded_recovery_hybrid_prune` / `guarded_variance_hybrid_prune`）在一个独立
小验证集上对 MENU 实测择优，**保证不劣于 fallback**。

## 2 剪枝示例

### 2.1 使用接口方式调用

在当前目录执行以下命令运行样例（纯 CPU 可跑）：

```bash
python3 src/run_dense_samples.py   # dense：固定率 / 容差自动 / recovery-HYBRID / 剪后量化 / evaluator·diagnose
python3 src/run_cnn_samples.py     # cnn：variance vs reconstruct 通道剪枝 / recovery-HYBRID
python3 src/run_moe_samples.py     # moe：activation_count vs mass_variance 专家剪枝 / variance-HYBRID
```

每个样例打印剪枝前后参数量、削减比例并做一次前向校验。

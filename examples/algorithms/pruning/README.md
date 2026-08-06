# AMCT 结构化剪枝样例

演示 `amct_pytorch.pruning` 接口在三种域上的调用：dense FFN（剪中间维）/ CNN（剪通道）/
MoE（剪专家）。样例使用随机初始化的微型模型，纯 CPU 运行，不下载任何权重。

> 接口详见 [`amct_pytorch/pruning/README.md`](../../../amct_pytorch/pruning/README.md)。

## 1 剪枝前提

### 1.1 安装依赖

依赖见 [requirements.txt](requirements.txt)：`torch` 与 `transformers`（随 amct_pytorch 导入链需要）。
在 NPU 上运行时另需与 Python/torch 版本匹配的 `torch_npu` 及已安装的 CANN 包。

### 1.2 模型与数据准备

样例模型与数据由 [src/utils.py](src/utils.py) 用固定随机种子构造（`MiniMLP`/`MiniCNN`/`MiniMoE`），
无需下载、无需联网。替换成真实模型与校准数据即可用于实际场景。

### 1.3 剪枝配置

以 dict 配置直接传入 `prune()`（与 `amct.quantize` 同风格），按域选择方法：

| 域 | 方法 | 说明 |
|:--|:--|:--|
| dense | `low_variance` | 按激活方差剪 FFN 中间维（自动避开注意力投影） |
| dense | `reconstruct` | 剪后用最小二乘重构补偿，recovery ∈ {none, bias, ls} |
| cnn | `variance_channel` | 按激活方差朴素切片通道 |
| cnn | `reconstruct` | 输出重构补偿的通道剪枝 |
| moe | `activation_count` | 按专家激活频次剪专家，同步收缩 gate |
| moe | `mass_variance` | 按专家质量方差剪专家 |

只给 `tolerance` 时走自动剪枝：在 `ratio_grid` 上二分查找满足容差的最大剪枝率。传入 menu 配置
（`MOE_VARIANCE_MENU_CFG` / `DENSE_RECOVERY_MENU_CFG` / `CNN_RECOVERY_MENU_CFG`）时，`prune` 改走 MENU 择优：
在 `eval_data` 指定的独立小验证集上实测每个候选，择优应用。

## 2 剪枝示例

### 2.1 使用接口方式调用

在当前目录执行以下命令运行样例（纯 CPU 可跑）：

```bash
python3 src/run_dense_samples.py   # dense：固定率 / 容差自动 / recovery-menu / 剪后量化 / evaluator
python3 src/run_cnn_samples.py     # cnn：variance vs reconstruct 通道剪枝 / recovery-menu
python3 src/run_moe_samples.py     # moe：activation_count vs mass_variance 专家剪枝 / variance-menu
```

每个样例打印剪枝前后参数量、削减比例并做一次前向校验。

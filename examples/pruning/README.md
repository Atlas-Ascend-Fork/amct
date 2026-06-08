# experimental.pruning 实用样例

`amct_pytorch.experimental.pruning` 各功能的**可直接运行**样例。所有模型均为随机初始化的
微型网络，**不下载任何权重**，纯 CPU 即可跑（CI/离线/无 NPU 环境友好）。

## 运行

```bash
# 任选其一（脚本用相对 import 共用 _models.py，需在本目录下运行）
cd examples/pruning
python 01_fixed_ratio_dense.py
python 02_tolerance_auto_prune.py
python 03_cnn_depgraph_importance.py
python 04_moe_expert_prune.py
python 05_prune_then_quantize.py      # 量化部分缺环境时自动跳过
python 06_evaluator_and_diagnose.py
```

## 样例与覆盖的功能

| 脚本 | 演示功能 | 对应 API |
|------|----------|----------|
| `01_fixed_ratio_dense.py` | 固定剪枝率、dict 配置、in-place、报告 | `prune` / `get_last_report` |
| `02_tolerance_auto_prune.py` | 容差驱动 + **二分查找**（紧容差不剪/放宽选率） | `prune(tolerance=)` / `get_last_auto_prune` |
| `03_cnn_depgraph_importance.py` | 现代 CNN 依赖图剪枝、5 种 importance、预设 | `prune` + `CNN_DEPGRAPH_PRUNE_CFG` |
| `04_moe_expert_prune.py` | MoE 专家剪枝（activation_count / mass_variance） | `prune`（moe 域） |
| `05_prune_then_quantize.py` | 剪枝→量化→转换；函数式 & PrunePass 两条路径 | `prune` / `PrunePass` + `amct.quantize/convert` |
| `06_evaluator_and_diagnose.py` | 自定义 evaluator / quality_fn / 可行性诊断 | `accuracy_based_auto_prune` / `diagnose` |

## 三个剪枝域速查

| 域 | 触发结构 | 方法名 |
|----|----------|--------|
| dense | LLM FFN（gate/up/down 或相邻两 Linear） | `low_variance` |
| cnn | 卷积通道（残差/下采样用依赖图） | `variance_channel` / `depgraph` |
| moe | gate + ModuleList experts / 现代融合专家 | `activation_count` / `mass_variance` |

## 备注

- **量化联合顺序**：先剪枝再量化——量化须在剪枝后的形状上解析 `layer_config`，
  故不能在同一个 `ModelOptimizer` 里把 `PrunePass` 与 `InsertQuantizeModulePass` 无脑串接
  （见 `05_*` 的 path_b 注释）。
- **evaluator 复用**：与量化侧 `accuracy_based_auto_calibration` 共用同一精度抽象，
  只需对象有 `.evaluate(model, iterations) -> float`。注意 graph 版内置 `ModelEvaluator.evaluate`
  **不返回值**（是 calibration 数据馈送器），需用返回 metric 的实现（见 `06_*`）。
- 真实精度评估请把样例里的代理质量函数换成"在验证集上跑模型并返回准确率/mAP"。

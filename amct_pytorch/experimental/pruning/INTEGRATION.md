# 与既有 graph 剪枝子系统的融合（不破坏既有，自然纳入）

本文件说明 `experimental.pruning`（eager 结构化剪枝）如何**与既有
`classic/graph_based` 剪枝子系统对齐、并最终自然纳入**，且全程**不修改/不破坏**既有代码。

## 1. 定位：互补而非替代

| 维度 | 既有 graph 子系统 | 新 experimental.pruning |
|------|-------------------|--------------------------|
| 表达 | ONNX 图（export→parse） | eager `torch.nn.Module` 直接遍历 |
| 配置 | proto `.cfg` 文件 | in-memory `dict`/`PruneConfig` |
| 覆盖 | CNN 卷积通道 + retrain | **LLM dense FFN / MoE 专家** + 现代 CNN |
| 部署 | Ascend 性能收益感知、cfg 产物 | 纯结构裁剪、in-place |
| 入口 | `auto_channel_prune_search`、`create_prune_retrain_model` | `prune`、`accuracy_based_auto_prune` |

→ 既有擅长 *CNN+retrain+Ascend 部署*；新模块补齐 *Transformer/MoE/无标签* 这块既有不覆盖的场景。
二者是同一"剪枝家族"的两条路径，**不互相覆盖命名、不互改代码**。

## 2. 已落地的"对齐"（让新模块说既有的语言）

这些是让新模块**可被自然纳入**的关键，均已实现：

1. **共用精度抽象**（与量化 `accuracy_based_auto_calibration` 同一个）：
   `accuracy_based_auto_prune(evaluator=...)` 直接接受 `AutoCalibrationEvaluatorBase`
   子类（`evaluate(model)->metric`）。同时兼容 `evaluate(model, iterations)`。
   误传"返回 None"的 graph 版 `ModelEvaluator`（数据馈送器）会明确报错。
2. **共用敏感度词汇**：新模块的依赖图剪枝暴露 `importance="taylor"`（一阶 `|w·grad|`），
   与既有 `auto_channel_prune_search` 的 `TaylorLossSensitivity` **同源思路**，便于将来统一。
3. **预算顺序衔接**：`accuracy_based_auto_prune_with_budget(total_tolerance, prune_fraction)`
   产出 `remaining_tolerance`，直接作为后续量化 `accuracy_based_auto_calibration` 的接受预算 ——
   prune→quant 的联合预算"分割并顺序测量"。
4. **Pass 协议一致**：`PrunePass.run(model)` 符合 `ModelOptimizer` 的 `run(model)` 协议，
   可与 `InsertQuantizeModulePass` 同处一个优化流水线（注意量化 layer_config 须在剪枝后解析）。

## 3. 命名对齐表（纳入时的对应关系）

| 既有（graph） | 新（eager） | 语义关系 |
|----------------|--------------|----------|
| `auto_channel_prune_search`（输出 cfg） | `accuracy_based_auto_prune`（in-place） | 自动搜索；后者 + `_with_budget` 接预算 |
| `TaylorLossSensitivity` | `importance="taylor"` | 一阶泰勒敏感度 |
| `ModelEvaluator`（馈送器，return None） | — | **不可**用于精度搜索 |
| `AutoCalibrationEvaluatorBase.evaluate`→metric | `evaluator=` | **共用**精度抽象 |
| `create_prune_retrain_model` | （暂无 retrain） | 既有独有，新模块不涉足 |

## 4. 纳入路线（建议，零破坏）

- **现状**：新模块隔离在 `experimental/`，纯 additive（既有文件 0 改动）。
- **第一步（已做）**：对齐评估器/敏感度/预算/Pass 词汇，使两侧概念可互通。
- **第二步（待定，需共识）**：在既有 `auto_channel_prune_search` 增加一个 *eager 后端选项*
  （如 `backend="eager"`），把本模块作为其在 Transformer/MoE 上的实现，**对外仍是一个入口**；
  既有 graph 后端保持默认、行为不变。
- **第三步（待定）**：文档合并到 `docs/zh/api/`，与 `auto_channel_prune_search.md` 并列成一节。

> 第二/三步涉及改既有公共 API，需先 Issue 讨论达成共识后再做（AGENTS.md 流程）。
> 在此之前，新模块以 experimental 形态零风险共存。

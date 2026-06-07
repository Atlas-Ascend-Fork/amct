# 结构化剪枝（experimental.pruning）

对已实例化的 `torch.nn.Module` 做结构化剪枝（dense FFN 中间维 / CNN 通道 / MoE 专家），
façade 与 `amct.quantize` 对齐：in-place 修改模型，统计经 `get_last_report()` 获取。
库内部不下载模型，调用方负责实例化（如 `from_pretrained(local_path)`）。

## 容差驱动自动剪枝（推荐，面向非专家）

不熟悉剪枝率/层结构也能用：**只说“最多可接受多少精度损失”**，自动搜索并应用
满足容差的最大剪枝率。与量化侧的 [`accuracy_based_auto_calibration`](../../../docs/api/accuracy_based_auto_calibration.md)
（精度优先 + BinarySearch）同理，是其剪枝版本。

```python
import amct_pytorch.experimental.pruning as amct

# 最多损失 2%，其余全自动
amct.prune(model, data=calib, tolerance=0.02)
print(amct.get_last_auto_prune().summary())
# [auto-prune] 已应用剪枝率=0.20：权重 607,808 → 568,640 (削减 6.4%)，质量损失 0.018 ≤ 容差 0.020。
```

- 默认**保真度指标无需标签**：以原模型在校准数据上的 top-1 预测为参考，质量 = 剪枝后预测一致率；
  也可传 `quality_fn`（与 `tolerance` 同量纲，如自定义准确率）。
- `safe_skip_attention=True`（默认）：自动跳过 `self_attn/attention/attn`，避免 dense 域把并列
  q/k/v 投影误判为两层 FFN 而破坏 LLM。
- 在 `ratio_grid` 上二分查找；任一剪枝率试探中剪枝/前向失败都判为不可接受，不抛异常。
  非单调时偏**保守**（少剪），对用户是安全方向。

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | 是 | - | 待剪枝 `torch.nn.Module`（in-place） |
| `data` | 视方法 | `None` | 校准数据；方差类方法需要，也作默认评估集 |
| `tolerance` | 是 | `0.02` | 可接受的精度损失上界（与 `quality_fn` 同量纲） |
| `quality_fn` | 否 | top-1 保真度 | 精度评估函数，越大越好 |
| `eval_data` | 否 | 回退到 `data` | 默认保真度指标的评估批 |
| `ratio_grid` | 否 | `0.1..0.8` | 候选剪枝率（升序） |
| `safe_skip_attention` | 否 | `True` | 自动保护注意力投影 |

> 显式 API：`accuracy_based_auto_prune(model, config=None, *, data, tolerance, ...) -> AutoTuneResult`。
> `amct.prune(..., tolerance=...)` 是其在主入口上的等价封装。

## 固定剪枝率

```python
cfg = {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}}
amct.prune(model, cfg, data=calib)         # dense 域自动避开注意力，无需手动 skip
print(amct.get_last_report().as_dict())
```

## CNN 通道剪枝（依赖图）

简单“同 parent 兄弟卷积”启发式（`variance_channel`）只适合平面/顺序 CNN；对
ResNet/ConvNext/MobileNet 等把卷积包进独立 block、带残差/下采样捷径的现代结构会
0 目标或维度不一致而崩溃。改用基于*依赖图*的方法 `depgraph`（跟踪真实数据流，
自动成组裁剪耦合层）：

```python
from amct_pytorch.experimental.pruning import CNN_DEPGRAPH_PRUNE_CFG
amct.prune(model, CNN_DEPGRAPH_PRUNE_CFG, data=calib)   # calib 提供样例输入用于图追踪
# 或自定义：{"methods": {"cnn": {"name": "depgraph",
#                                "kwargs": {"prune_ratio": 0.3, "importance": "magnitude"}}}}
```

- 自动忽略输出头(最后一个 Linear)，保持类别数不变；`skip_layers` 命中的层也忽略。
- `importance`：`magnitude`(默认)/`l1`/`lamp`/`bn`/`random`。
- 依赖图实现以 static 方式内置于 `_vendor/torch_pruning`（改编自 MIT 许可的
  [Torch-Pruning](https://github.com/VainF/Torch-Pruning)，版权归原作者，见该目录
  `LICENSE`/`NOTICE.md`），**不引入任何外部依赖**。

## 性能测量

小尺寸真实 HF 网络（随机初始化，transformers 5.6.x，CPU）。**精度 = 与未剪枝模型的
top-1 一致率**（保真度，非任务精度；trained 权重下趋势相同、绝对值更高）。

固定剪枝率：

| 模型（域） | ratio | 权重削减 | top-1 一致率 |
|------------|:-----:|:--------:|:------------:|
| Qwen3（dense FFN） | 0.3 / 0.5 / 0.7 | 9.7% / 16.2% / 22.6% | 95.8% / 91.7% / 88.5% |
| Llama（dense FFN，跳注意力） | 0.3 / 0.5 / 0.7 | 22.5% / 37.4% / 52.3% | 87.5% / 81.2% / 66.7% |
| RegNet（CNN 通道） | 0.3 / 0.5 / 0.7 | 3.8% / 6.3% / 8.9% | 100% / 100% / 100% |

容差驱动（Qwen3 dense，自动选率）：

| 容差 | 选中剪枝率 | 权重削减 | 实测损失 |
|:----:|:----------:|:--------:|:--------:|
| 0.05 | 0.1 | 3.3% | 0.044 |
| 0.10 | 0.2 | 6.4% | 0.070 |
| 0.20 | 0.8 | 25.9% | 0.190 |

> 权重削减幅度取决于目标域占比：Qwen3 注意力较大（head_dim=128），仅剪 FFN 时占比小；
> Llama 注意力小，FFN 占比大、削减更多。

## 与量化联合使用（prune → quantize → convert）

剪枝（结构化、减通道/神经元/专家）与量化（降比特）是**正交**的压缩轴，体积收益
**相乘**、精度损失**近似相加**。推荐顺序：**先剪枝，再量化，最后转换部署模型** ——
剪枝改变张量形状，量化须在剪枝后的权重上做 calibration 才正确（`PrunePass` 已设计为
在 `ModelOptimizer` 中位于 `InsertQuantizeModulePass` 之前）。

```python
import amct_pytorch as amct
from amct_pytorch.experimental.pruning import accuracy_based_auto_prune

accuracy_based_auto_prune(model, data=calib, tolerance=0.02)  # 1) 结构化剪枝
amct.quantize(model, quant_cfg)                                # 2) 在剪枝后的模型上量化
amct.convert(model)                                            # 3) 转换部署模型
```

**联合精度预算**：总预算（如 3%）在两步间**分割并顺序测量** —— 剪枝先用掉实测的一部分，
量化搜索在*剪枝后的模型*上用剩余预算。这与量化侧
[`accuracy_based_auto_calibration`](../../../docs/api/accuracy_based_auto_calibration.md) 天然衔接。

**共用同一精度抽象**：把量化用的 `ModelEvaluator`（带 `evaluate(model, iterations)`）直接传给
剪枝的 `evaluator=`，**同一个评估器**即可驱动剪枝与量化两侧的精度搜索，无需定义两套度量：

```python
accuracy_based_auto_prune(model, data=calib, tolerance=0.02, evaluator=model_evaluator)
amct.accuracy_based_auto_calibration(model, model_evaluator, quant_cfg, ...)
```

> 优先级：`quality_fn` > `evaluator` > 默认 top-1 保真度。

## 当前支持边界（以当前 transformers 实测）

| 域 | 支持 | 说明 |
|----|------|------|
| dense | ✅ | LLM 的 FFN（`gate/up/down_proj` 或相邻两层 Linear）。dense 域结构化排除并列的注意力投影（q/k/v/o 共享输入、且不满足 in→inter→in 残差维度复原），Llama/Qwen2/Mistral/Qwen3 **无需手动 `skip_layers`** 即可安全剪枝。 |
| cnn | ✅ | `variance_channel`(启发式) 适合平面/顺序 CNN；**`depgraph`(依赖图) 支持 ResNet/ConvNext/MobileNetV2/RegNet 等残差/下采样/独立 block 结构**（实测 ~50% 通道剪枝、前向正常）。 |
| moe | ✅ | 同时支持 `nn.ModuleList` experts + `nn.Linear` gate 的老式结构，**与现代 transformers 的融合(批处理)专家**（`MixtralExperts`/`Qwen3MoeExperts` + `*TopKRouter`，沿专家维切片批张量）。 |

回归基线见 `tests/amct_pytorch/test_pruning_real_hf_models.py`、容差功能见
`tests/amct_pytorch/test_pruning_auto_prune.py`（需 transformers，用 `.venv` 运行）。

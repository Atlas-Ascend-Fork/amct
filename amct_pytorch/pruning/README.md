# 结构化剪枝（amct_pytorch.pruning）

对已实例化的 `torch.nn.Module` 做结构化剪枝（dense FFN 中间维 / CNN 通道 / MoE 专家），
façade 与 `amct.quantize` 对齐：in-place 修改模型，无进程级全局状态——统计经可选的 `report=PruneReport()` sink 获取，搜索结果由 `accuracy_based_auto_prune` / `size_budget_prune` 的返回值获取。
（与 `amct.quantize` 不同：结构化剪枝会原地改写模型的 `config` 维度（`intermediate_size` / `num_experts`），使剪枝后的模型能正确保存/重载。）
库内部不下载模型，调用方负责实例化（如 `from_pretrained(local_path)`）。

# 剪枝特性

## 容差驱动自动剪枝（推荐，面向非专家）

不熟悉剪枝率/层结构也能用：**只说“最多可接受多少精度损失”**，自动搜索并应用
满足容差的最大剪枝率。与量化侧的 [`accuracy_based_auto_calibration`](../../docs/zh/api/accuracy_based_auto_calibration.md)
（精度优先 + BinarySearch）同理，是其剪枝版本。

```python
import amct_pytorch as amct

# 最多损失 2%，其余全自动
res = amct.accuracy_based_auto_prune(model, data=calib, tolerance=0.02)
print(res.summary())
# [auto-prune] applied prune_ratio=0.20: weights 607,808 → 568,640 (cut 6.4%), quality loss 0.018 ≤ tolerance 0.020.
```

### 搜索 + 剪枝后微调恢复（重要）

一次性结构剪枝会让精度**崩到接近随机**；若搜索按"剪枝直后"评估，几乎选不出任何率（过度保守）。
给 `finetune_fn` 后，搜索对**每个候选率先做(轻量)微调恢复再评估**，于是选出的是"剪枝+恢复后"
真正可用的更激进的率：

```python
from amct_pytorch.pruning import prune_finetune
amct.prune(model, cfg, data=calib, tolerance=0.05, evaluator=test_acc,
           finetune_fn=lambda m: prune_finetune(m, train_data, steps=300))
```

> 实测(MNIST, **held-out val acc** 门控, 容差 5%)：**无 `finetune_fn` 选不出任何率(0% 剪枝)；
> 有 `finetune_fn` 选出 50% 通道剪枝**（恢复后 val acc 仍在容差内；baseline 0.857）。代价：每个候选都要微调一次。

- 默认**保真度指标无需标签**：以原模型在校准数据上的 top-1 预测为参考，质量 = 剪枝后预测一致率；
  也可传 `evaluator`（与 `tolerance` 同量纲，如自定义准确率）。
- `safe_skip_attention=True`（默认）：自动跳过 `self_attn/attention/attn`，避免 dense 域把并列
  q/k/v 投影误判为两层 FFN 而破坏 LLM。
- 在 `ratio_grid` 上二分查找；任一剪枝率试探中剪枝/前向失败都判为不可接受，不抛异常。
  非单调时偏**保守**（少剪），对用户是安全方向。

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | 是 | - | 待剪枝 `torch.nn.Module`（in-place） |
| `data` | 视方法 | `None` | 校准数据；方差类方法需要，也作默认评估集 |
| `tolerance` | 是 | `0.02` | 可接受的精度损失上界（与 `evaluator` 同量纲） |
| `evaluator` | 否 | top-1 保真度 | 精度来源：`callable(model)->float` 或 `AutoCalibrationEvaluatorBase`（与量化共享） |
| `eval_data` | 否 | 回退到 `data` | 默认保真度指标的评估批 |
| `ratio_grid` | 否 | `0.1..0.8` | 候选剪枝率（升序） |
| `safe_skip_attention` | 否 | `True` | 自动保护注意力投影 |

> 显式 API：`accuracy_based_auto_prune(model, config=None, *, data, tolerance, ...) -> AutoTuneResult`。
> `amct.prune(..., tolerance=...)` 是其在主入口上的等价封装。

## 固定剪枝率

```python
from amct_pytorch.pruning import PruneReport
cfg = {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}}
rep = PruneReport()
amct.prune(model, cfg, data=calib, report=rep)   # dense 域自动避开注意力，无需手动 skip
print(rep.as_dict())
```

**部分配置会按默认剪其余域。** `methods` 只是*覆盖*各域默认；未指定的域仍按默认率被剪
（cnn 0.30 / dense 0.50 / moe 0.50）。要**只剪**某域，把其余域 pin 到 `prune_ratio: 0.0`。
这对多域模型尤为关键——例如专家内部含 dense FFN 的 MoE：只写 moe 配置会把那层 FFN 也一并剪掉。

## CNN 通道剪枝（输出保留重构）

简单“同 parent 兄弟卷积”启发式（`variance_channel`）按通道激活方差朴素切片，适合平面/顺序 CNN。
推荐 `reconstruct`：在同一 cnn 域（识别 producer conv → bn → consumer 顺序耦合）上，剪掉低显著度
通道后用 **im2col 最小二乘重构 consumer 卷积权重**，把被移除通道的贡献吸收进保留通道，使输出尽量
保持（ThiNet/CP 式补偿，无需微调；自研实现（LS 重构思想源自 ThiNet/Channel Pruning, ICCV'17），无第三方依赖）：

```python
from amct_pytorch.pruning import CNN_RECONSTRUCT_PRUNE_CFG
amct.prune(model, CNN_RECONSTRUCT_PRUNE_CFG, data=calib)   # calib 提供样例输入用于激活统计
# 或自定义：{"methods": {"cnn": {"name": "reconstruct",
#                                "kwargs": {"prune_ratio": 0.3, "ridge": 1e-2}}}}
```

- 自动忽略输出头(最后一个 Linear)，保持类别数不变；`skip_layers` 命中的层也忽略。
- `reconstruct` 在已训练模型上比朴素 `variance_channel` 切片的破坏更低（同参数量）。
- 校准须有代表性且多样（行数 ≥ keep×k×k 才能解最小二乘）；欠定或 consumer 非 Conv2d 时**明确告警**
  并退回朴素切片（不静默伪装为“已重构”）。
- 在**残差（ResNet）** block 中，*内部* conv→conv 耦合**会被剪枝**；仅把馈入残差 `add` 的那个 conv 自动排除（从而保持残差宽度、避免破坏）——并非整体跳过。Concat（Inception）consumer 与 分组/depthwise 卷积仍为 **0 目标**（暂缓，见"支持的可剪结构"）

## 守卫式判据选择（MoE 专家剪枝）

> **高级接口（非 __all__ 默认公开）** —— 通过 `from amct_pytorch.pruning import ...` 导入。

方差类路由打分一次校准即可得到（无梯度/最小二乘/专家前向），在*部分* MoE 上优于按激活
频率（mass）的剪枝；但没有普适的"更好的方差"判据（判据与路由结构相关）。**MoE 方差判据 menu**
不赌单一判据：把 `MOE_VARIANCE_MENU_CFG` 预设传给 `accuracy_based_auto_prune`，即可
**一次校准 → 在判据 MENU 上各剪一份副本 → 用小验证集实测择优 → 应用最优者**：

```python
from amct_pytorch.pruning import MOE_VARIANCE_MENU_CFG

res = amct.accuracy_based_auto_prune(
    model, MOE_VARIANCE_MENU_CFG, data=calib, eval_data=val, evaluator=ev)  # val 建议与 calib 不同
print(res.summary())            # 标准 AutoTuneResult；选中的变体记录在返回的 AutoTuneResult.report 中（或传 report= sink）
```

- **MENU**（默认）：`mass`(=`activation_count`，fallback) / `cond_var`(Var(w|sel)) /
  `peak`(max w，识别"始终强权重"的特化专家，cond 的盲点) / `cvxpeak`(Var(w|sel)×max w)。
- **守卫保证**：仅当某方差判据在**验证集**上*严格优于* fallback 时才切换，否则回退 `mass`
  —— 因此 HYBRID 在验证集上不劣于 fallback（与 `accuracy_based_auto_prune` 在剪枝率上
  搜索同构，这里搜的是*判据*；二者复用同一套 deepcopy 试探 + evaluator 守卫 + 就地应用）。
- 增益在 **类/专家使用不均衡 + 尾部敏感指标**（如分组路由的多专家 MoE）下最明显。
- **测得范围**：在**真实预训练 MoE** 上判据**依赖深度** —— *全局*方差判据（每层都用）
  劣于 mass，但只在**深层**用方差（`boundary=K`）在多专家路由上优于 mass，见下文
  **真实 MoE 上的深度分界专家选择**。LLaDA2.1-mini **GSM8K** 上激进*静态*专家剪枝会崩溃
  （需动态 masking）。即真实 MoE 的增益需要**深度分界**而非全局判据；"never worse"
  仍是 **VAL 保证**（granite 的少专家路由会正确回退到 mass）。
- 判据的底层打分由 `mass_variance` 的新 kwarg `variance_score`∈{`cond`(默认),`peak`,`cvxpeak`}
  提供，并按深度 `boundary` 切分：**层 idx ≤ boundary 用 mass，> boundary 用方差**（默认 `10`，在
  LLaDA2.1-mini 上验证；`-1` = 每层都用方差；`"auto"` = 逐层择优；`list[int]` = 仅这些层用方差）；
  默认 `cond` 与既有行为完全一致（向后兼容）。`MOE_VARIANCE_MENU_CFG` 的默认 menu 用全局
  `boundary=-1`；传入自定义 menu（方差项设 `boundary=K`）即可搜索深度分界。

### 真实 MoE 上的深度分界专家选择

专家保留判据**依赖深度**：在**深层**（最后几层）MoE 中，每专家路由权重**方差**比激活**频率**（mass）
更能选出该保留的专家；**浅层**偏好 mass；全方差判据劣于深度分界。`mass_variance` 按深度
`boundary` K 切分（层 ≤ K 用 mass，> K 用方差）。当每专家方差信号不可靠（少专家路由）时，受保护的
menu 搜索会回退到 all-mass，因此结果绝不劣于 mass。真实预训练 MoE 在紧质量预算下的可剪量受微调门控。

## 守卫式恢复选择（dense FFN / CNN 通道剪枝）

> **高级接口（非 __all__ 默认公开）** —— 通过 `from amct_pytorch.pruning import ...` 导入。

对 dense FFN / CNN 通道剪枝，决定精度的杠杆是剪枝后的 **RECOVERY（补偿方式）**，而非选神经元的
判据。**dense/CNN recovery menu** 与方差 menu 同构，但搜的是 recovery MENU：把
`DENSE_RECOVERY_MENU_CFG` 预设传给 `accuracy_based_auto_prune`：

```python
from amct_pytorch.pruning import DENSE_RECOVERY_MENU_CFG

res = amct.accuracy_based_auto_prune(
    model, DENSE_RECOVERY_MENU_CFG, data=calib, eval_data=val, evaluator=ev)  # domain（dense/cnn）在预设里
print(res.summary())            # 标准 AutoTuneResult；选中的变体记录在返回的 AutoTuneResult.report 中（或传 report= sink）
```

- **MENU**（默认）：`none`(朴素切片, fallback) / `bias`(均值折叠到 consumer bias, 无最小二乘) /
  `ls`(最小二乘重构 consumer 权重)。底层由 `reconstruct` 的新 kwarg `recovery`∈{`ls`(默认),`bias`,
  `none`} 提供（dense 与 cnn 都加了，默认 `ls` = 既有行为，向后兼容）。
- **守卫保证（限界要诚实）**：守卫仅当某 recovery 在**验证集**上严格优于 fallback(none) 时才切换 ——
  因此在验证集上不劣于不补偿基线。这是 **VAL 上的保证，不是 test 上的保证**：若 `eval_data`
  不代表部署分布，验证择优可能不迁移到 test（见下小校准条目）。务必用**代表部署分布**的 `eval_data`。
- **校准规模是关键旋钮**：校准充分 → `ls` 最佳；校准 < 中间维 → `ls` 欠定过拟合，`bias`(无最小二乘)更稳。
  守卫会自动随校准规模在 bias↔ls 间切换。
- **小校准安全旋钮（dense）**：`bias` 折叠会过拟合校准域的激活均值，在分布漂移的 test 上可能反而
  变差（这是结构性 val↔test 分布失配，非采样噪声）。要 test 侧也稳健，从菜单里去掉 bias ——
  在 `cfg["methods"]["dense"]["menu"]` 中只保留 `none`/`ls` 两项 —— 以小校准的平均增益换取
  test 侧"绝不更差"。
- **CNN 无此 floor**：CNN 通道剪枝里 im2col 行数 = N×空间，即使少量图也充足，`ls` 不欠定、`bias`
  极少成为验证赢家，full 守卫已 test-side 安全。
- **测得范围**：以上 dense/CNN 数字均为 **no-FT**（部署即用、无重训练）相对比较；recovery 是这一无微调
  场景下的杠杆。若后续做微调恢复，朴素切片也可能追上，那是另一问题。

## 进阶配置

```python
# 跨层敏感度分配（默认 uniform；guard 在劣化时自动回退 uniform 并告警）
amct.prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib)
# config["allocation"] = {"strategy": "sensitivity", "ref_ratio": 0.5,
#                          "min_ratio": 0.05, "max_ratio": 0.9, "guard": "calib_nll"}

# MoE 输出空间专家合并（LS 残差配对 + 丢弃回退；calib NLL 与 activation_count A/B 自动择优）
amct.prune(model, MOE_OUTPUT_MERGE_PRUNE_CFG, data=calib)
# moe kwargs: keep_ratio=0.5, selector='calib_nll'|'none'

# 量化感知显著度：直接复用 AMCT calibrate 同式 scale/offset
cfg["methods"]["dense"]["kwargs"]["quant_cfg"] = {
    "weights_cfg": {"strategy": "channel", "symmetric": True, "dtype": "int8"}}

# 预算口径：budget_scope='total'（默认）|'prunable'；不可达时告警 + report.budget_unreachable
amct.prune(model, data=calib, size_budget=0.7)   # prune to a size budget (keep 70%)
# 完整控制（target_params / budget_scope）用 size_budget_prune（高级，from amct_pytorch.pruning import）
```

# 与量化联合使用（prune → quantize → convert）

剪枝（结构化、减通道/神经元/专家）与量化（降比特）是**正交**的压缩轴，体积收益
**相乘**、精度损失**近似相加**。推荐顺序：**先剪枝，再量化，最后转换部署模型** ——
剪枝改变张量形状，量化须在剪枝后的权重上做 calibration 才正确。

```python
import amct_pytorch as amct
from amct_pytorch.pruning import accuracy_based_auto_prune

accuracy_based_auto_prune(model, data=calib, tolerance=0.02)  # 1) 结构化剪枝
amct.quantize(model, quant_cfg)                                # 2) 在剪枝后的模型上量化
amct.convert(model)                                            # 3) 转换部署模型
```

**联合精度预算**：自己把同一份总预算在两步间**分割并顺序测量** ——
剪枝先用掉实测的一部分，量化搜索在*剪枝后的模型*上用剩余预算：

```python
# 自己把同一份精度预算分配给剪枝 + 量化（无需额外 API）：
total_tolerance = 0.03                                                       # 总预算 3%
res = amct.accuracy_based_auto_prune(model, data=calib,
                               tolerance=total_tolerance * 0.5, evaluator=ev)  # 先给剪枝一半上界
prune_drop = res.quality_drop or 0.0                                          # 剪枝实测的精度损失
remaining = max(0.0, total_tolerance - prune_drop)                            # 留给量化的剩余预算
# 把剩余预算作为量化精度搜索的接受阈值（与量化共用同一 evaluator）
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ...)               # 量化预算 = remaining
```

`remaining = max(0, total_tolerance - 实测剪枝损失)`，与量化侧
[`accuracy_based_auto_calibration`](../../docs/zh/api/accuracy_based_auto_calibration.md) 天然衔接。

**共用同一精度抽象**：剪枝 `evaluator=` 与量化 `accuracy_based_auto_calibration` 共用
**同一个精度评估器** —— 即 `AutoCalibrationEvaluatorBase` 的子类（其 `evaluate(model)`
**返回一个精度指标**，越大越好）。同一个对象即可驱动两侧的精度搜索，无需定义两套度量：

```python
class MyEvaluator(AutoCalibrationEvaluatorBase):
    def evaluate(self, model):              # 返回 top-1 / mAP 等指标
        return my_top1_accuracy(model)

ev = MyEvaluator()
accuracy_based_auto_prune(model, data=calib, tolerance=0.02, evaluator=ev)   # 剪枝
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ...)              # 量化
```

剪枝侧同时兼容 `evaluate(model)` 与 `evaluate(model, iterations)` 两种签名。

> 注意区分：graph 版内置 **`amct.ModelEvaluator`** 是量化 calibration 的**数据馈送器**，
> 其 `evaluate(model, iterations)` **返回 None**，不能用于精度搜索；误传时剪枝侧会给出明确报错。
> 需要精度搜索请用返回指标的 `AutoCalibrationEvaluatorBase` 子类。
>
> 精度：`evaluator`（callable 或 `AutoCalibrationEvaluatorBase`）或默认 top-1 保真度。

# 附录：与既有 graph 子系统 / 支持边界

## 与既有 graph 剪枝子系统的关系（零破坏）

本模块与既有 `classic/graph_based`（CNN 卷积通道 + retrain + Ascend 部署）**互补、不替代**：
面向 eager `torch.nn.Module` 的 LLM dense FFN / MoE 专家 / 现代 CNN，纯 additive（既有文件 0 改动）。
已对齐既有词汇以便将来纳入：共用精度抽象 `AutoCalibrationEvaluatorBase`、预算顺序衔接
（剪枝先用掉实测的一部分，量化 `accuracy_based_auto_calibration` 再在剩余预算上运行）。
当前作为独立的纯增量模块（`amct_pytorch.pruning`）与既有 graph 剪枝零风险共存。

### 支持的可剪结构
结构化剪枝只针对 producer↔consumer 接口可被验证的维度，对其余维度保守跳过（绝不误剪）：
- **Dense FFN** —— 仅剪 *中间* 维（`gate/up.out_features` + `down.in_features`）；隐藏/残差宽度（`down.out_features`）保持不变。注意力投影（q/k/v/o）自动排除。
- **CNN 通道** —— producer `Conv2d`（`groups=1`）→ 可选 `BatchNorm2d` → consumer `Conv2d`/`Linear`，且 `consumer.in_channels == producer.out_channels`；producer-out、BN、consumer-in 三者一起协同改尺寸。在**残差（ResNet）** block 中，*内部* conv→conv 耦合会被剪枝（馈入残差 `add` 的那个 conv 自动排除 → 残差宽度保持不变）。**Concat（Inception）** consumer（`in = Σ 各分支`）与 **分组/depthwise** 卷积**不剪**（0 目标）—— 验证其安全性需要跨分支依赖追踪，暂缓。
- **MoE 专家** —— 整个*路由（routed）*专家被移除；该层的输入/输出（hidden）不变。**共享（始终激活）专家从路由专家剪枝中排除**。

## 当前支持边界（以当前 transformers 实测）

| 域 | 支持 | 说明 |
|----|------|------|
| dense | √ | LLM 的 FFN：`gate/up/down_proj` 三 Linear、**融合 `gate_up_proj`**（Phi-3/GLM-4 风格，前半 gate 后半 up）、相邻两层 Linear/Conv1D（含 Bloom 风格自定义无参激活间隔）。dense 域结构化排除并列的注意力投影（q/k/v/o 共享输入、且不满足 in→inter→in 残差维度复原），Llama/Qwen2/Mistral/Qwen3 **无需手动 `skip_layers`** 即可安全剪枝。 |
| cnn | √ | `variance_channel`(启发式) 朴素切片；`reconstruct`(im2col 最小二乘) 输出保留、破坏更低。ResNet 等独立 block 结构不被同级检测识别（已知边界）。 |
| moe | √ | `nn.ModuleList` experts + `nn.Linear` gate 的老式结构，**与现代 transformers 的融合(批处理)专家**（`MixtralExperts`/`Qwen3MoeExperts` + `*TopKRouter`，沿专家维切片批张量）。另支持：**分组路由**（`n_routed_experts`/`n_group`/`topk_group` —— 同步专家数并把分组塌缩为 1；DeepseekV3 风格）、**兄弟双张量融合专家**（专家拆到并列子模块，如 GraniteMoE 的 `input_linear`/`output_linear`）、**嵌套路由偏置**切片（如 Ernie4.5 的 `moe_statics.e_score_correction_bias`）。由 `test_pruning_real_hf_models.py` 验证（Mixtral、Qwen3Moe、GraniteMoe、DeepseekV3、Ernie4_5_Moe）。 |

**架构覆盖矩阵**（config 缩小为 2 层微型模型实测，权重随机仅验证机械可剪）：覆盖 29 个架构家族 —
llama/llama3/qwen2/qwen3/qwen2_moe/qwen3_moe/mixtral/mistral/gemma/gemma2/gemma3/phi/phi3/glm4/
gpt2/gpt_neox/gpt_neo/gpt_bigcode/starcoder2/opt/xglm/olmo/olmo2/stablelm/smollm3/bloom/
granitemoe/deepseekv3/ernie4_5。

回归基线见 `tests/amct_pytorch/test_pruning_real_hf_models.py`、容差功能见
`tests/amct_pytorch/test_pruning_auto_prune.py`（需 transformers）。

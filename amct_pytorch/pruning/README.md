# 结构化剪枝 (amct_pytorch.pruning)

对已实例化的 `torch.nn.Module` 进行结构化剪枝（稠密 FFN 中间维 / CNN 通道 / MoE 专家）。
模型被原地修改；统计信息通过可选的 `report=PruneReport()` 出口以及
`accuracy_based_auto_prune` / `size_budget_prune` 的返回值返回。剪枝会原地改写模型
`config` 的维度（`intermediate_size` / `num_experts`），使剪枝后的模型能正确保存/重载。
本库从不下载模型；由调用方负责实例化。

## 工作原理（先看这张图）

结构化剪枝按四步走：**打分 → 剪枝 → 恢复 →（可选）量化**。

<img src="figures/overview.svg" alt="结构化剪枝总流程" width="900">

你给一个**已加载权重**的模型和**少量校准数据**（几批前向即可）；工具在校准数据上前向一遍，
为每个可剪的结构（FFN 中间维通道 / CNN 通道 / MoE 专家）打一个“重要性分数”，移除分数最低的那部分，
必要时再对保留下来的权重做一次轻量补偿（**恢复**），最后可选地接量化。整个过程**原地**修改模型，
并同步改写 `config` 里的维度，因此剪枝后的模型能直接 `save/load`，无需额外转换步骤。

### 三个可剪枝的域

能剪什么、剪完之后什么保持不变，取决于“域”。工具只剪**生产者→消费者接口能对得上**的维度，
对不确定的一律保守跳过，保证张量形状始终自洽：

<img src="figures/domains.svg" alt="三个可剪枝的域" width="900">

- **稠密 FFN** —— 只收窄中间维：`gate/up_proj` 的**输出通道**和 `down_proj` 的**输入通道**一起变小；
  隐藏维 / 残差宽度保持不动，注意力 q/k/v/o 投影自动排除。
- **CNN 通道** —— 沿“生产者卷积输出通道 →（可选 BatchNorm）→ 消费者输入通道”这条链一起改尺寸；
  喂入残差 `add` 的卷积、Concat（Inception）消费方、分组 / 深度可分离卷积都不剪。
- **MoE 专家** —— 整块地移除被路由的专家，并同步收缩路由器；共享（始终激活）专家保留，隐藏 in/out 不变。

> 各域支持的具体网络结构见文末 [支持的可剪枝结构](#支持的可剪枝结构)。

## 容差驱动的自动剪枝

只需指定可接受的精度损失；在容差范围内搜索（在 `ratio_grid` 上二分查找）并应用满足容差的最大剪枝率。

<img src="figures/tolerance_search.svg" alt="容差驱动的二分搜索" width="560">

你不用猜该剪多少——只给一个能接受的精度损失 `tolerance`，工具就在 `ratio_grid` 上**二分查找**：
每试一个剪枝率 `r`，就在副本上剪一遍、（可选微调后）量一下精度掉了多少；掉得在容差内，说明还能再狠一点
（往更大的 `r` 试），超了就往更小的 `r` 退。最终应用**满足容差的最大剪枝率**。任何剪枝 / 前向失败都被
当作不可接受（不抛异常），于是在非单调的情况下搜索会自动保守少剪。

```python
import amct_pytorch as amct

res = amct.accuracy_based_auto_prune(model, data=calib, tolerance=0.02)
print(res.summary())
```

当提供 `finetune_fn` 时，每个候选剪枝率在评估前都会先微调，因此被选中的剪枝率是剪枝 + 恢复后
可用的那个：

```python
from amct_pytorch.pruning import prune_finetune
amct.prune(model, cfg, data=calib, tolerance=0.05, evaluator=test_acc,
           finetune_fn=lambda m: prune_finetune(m, train_data, steps=300))
```

- 默认保真度指标无需标签：质量 = 在校准数据上与原始模型 top-1 预测的一致率。也可改为传入
  `evaluator`（`callable(model)->float` 或 `AutoCalibrationEvaluatorBase`）。
- `safe_skip_attention=True`（默认）：跳过 `self_attn/attention/attn`。
- 探测某个剪枝率时的任何剪枝/前向失败都被视为不可接受（不抛异常）。在非单调情况下，搜索会少剪。

### 参数

| 参数 | 是否必填 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | 是 | - | 待剪枝的 `torch.nn.Module`（原地） |
| `data` | 视情况 | `None` | 校准数据；基于方差的方法、默认评估集需要 |
| `tolerance` | 否 | `0.02` | 可接受精度损失的上界（与 `evaluator` 同量纲）；默认 `0.02` |
| `evaluator` | 否 | top-1 保真度 | `callable(model)->float` 或 `AutoCalibrationEvaluatorBase` |
| `eval_data` | 否 | 回退到 `data` | 默认保真度指标使用的评估批 |
| `ratio_grid` | 否 | `0.1..0.8` | 候选剪枝率（升序） |
| `safe_skip_attention` | 否 | `True` | 跳过注意力投影 |

函数签名：`accuracy_based_auto_prune(model, config=None, *, data, tolerance, ...) -> AutoTuneResult`。
`amct.prune(..., tolerance=...)` 是等价封装。

## 固定剪枝率

```python
from amct_pytorch.pruning import PruneReport
cfg = {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}}
rep = PruneReport()
amct.prune(model, cfg, data=calib, report=rep)
print(rep.as_dict())
```

`methods` 只覆盖各域的默认值；未列出的域仍按默认率剪枝
（cnn 0.30 / dense 0.50 / moe 0.50）。若只想剪一个域，将其余域固定为 `prune_ratio: 0.0`。

## CNN 通道剪枝

`variance_channel` 按激活方差切分通道（同父兄弟卷积启发式）。
`reconstruct` 先移除低显著度通道，再通过 im2col 最小二乘重建消费方卷积权重以保持输出（无需微调）：

```python
from amct_pytorch.pruning import CNN_RECONSTRUCT_PRUNE_CFG
amct.prune(model, CNN_RECONSTRUCT_PRUNE_CFG, data=calib)
# custom: {"methods": {"cnn": {"name": "reconstruct",
#                              "kwargs": {"prune_ratio": 0.3, "ridge": 1e-2}}}}
```

- 输出头（最后一个 Linear）及匹配 `skip_layers` 的层会被自动忽略。
- 校准必须提供足够多的行（>= keep x k x k）以求解最小二乘；当方程欠定或消费方不是 Conv2d 时，
  会发出警告并改用朴素切片。
- 在残差（ResNet）块中，内部 conv->conv 耦合会被剪枝；馈入残差 `add` 的卷积会被自动排除。
  Concat（Inception）消费方以及分组/深度可分离卷积不剪枝。

## MoE 专家剪枝（方差准则菜单）

> 进阶（不在 `__all__` 中）-- 通过 `from amct_pytorch.pruning import ...` 导入。

将 `MOE_VARIANCE_MENU_CFG` 传给 `accuracy_based_auto_prune`，即可运行一次校准、按菜单中的每个准则
各剪枝一份副本、在小验证集上挑出最佳并应用胜者：

<img src="figures/menu_select.svg" alt="一次校准、多方案择优" width="560">

“菜单”类配置的套路是**一次校准、多方案择优**：共用同一遍校准，把菜单里每个候选各剪出一份副本，
在一个**独立的小验证集**（`eval_data`）上实测，谁**严格赢过安全回退项**才用谁；打平或没赢就保留回退项。
这样即使某个花哨方案在校准集上好看、到验证集上并不稳，也不会被误选。MoE 方差菜单里，各候选就是不同的
重要性打分准则（`mass` / `cond_var` / `peak` / `cvxpeak`），其中 `mass` 是那个安全回退项。

```python
from amct_pytorch.pruning import MOE_VARIANCE_MENU_CFG
res = amct.accuracy_based_auto_prune(
    model, MOE_VARIANCE_MENU_CFG, data=calib, eval_data=val, evaluator=ev)  # keep val != calib
print(res.summary())
```

- 菜单：`mass`（= `activation_count`，回退）/ `cond_var` / `peak` / `cvxpeak`。
- 方差准则仅在验证集上严格胜出时才替换 `mass`，否则回退到 `mass`。
- 各准则的分数是 `mass_variance` 的 kwarg `variance_score`，取值 {`cond`（默认）, `peak`, `cvxpeak`}，
  深度切分 `boundary`：层 <= boundary 用 mass，层 > boundary 用 variance（默认 `10`；
  `-1` = 全用 variance；`"auto"` = 逐层；`list[int]` = 指定这些层）。`MOE_VARIANCE_MENU_CFG` 默认
  `boundary=-1`；传入带 `boundary=K` 的自定义菜单可搜索深度切分。

## 恢复菜单（稠密 FFN / CNN 通道剪枝）

> 进阶（不在 `__all__` 中）-- 通过 `from amct_pytorch.pruning import ...` 导入。

将 `DENSE_RECOVERY_MENU_CFG` 传入以搜索剪枝后的恢复菜单：

恢复菜单沿用上一节**完全相同的“择优菜单”机制**（见 [上图](#moe-专家剪枝方差准则菜单)），
只是候选换成了剪枝后的补偿方式——同样是一次校准、在独立验证集上择优、只有严格胜出才替换回退项。

```python
from amct_pytorch.pruning import DENSE_RECOVERY_MENU_CFG
res = amct.accuracy_based_auto_prune(
    model, DENSE_RECOVERY_MENU_CFG, data=calib, eval_data=val, evaluator=ev)
print(res.summary())
```

- 菜单：`none`（朴素切片，回退）/ `bias`（均值折叠进消费方 bias）/ `ls`（最小二乘重建）。
  由 `reconstruct` 的 kwarg `recovery` 支撑，取值 {`ls`（默认）, `bias`, `none`}。
- 仅在验证集上严格胜出时才替换回退项；请使用能代表部署场景的 `eval_data`。
- 校准充足时倾向 `ls`；校准量 < 中间维宽度会使 `ls` 欠定，此时 `bias` 更稳健。
- `bias` 折叠可能在偏移的测试集上过拟合校准均值；为保留测试侧下限，可从
  `cfg["methods"]["dense"]["menu"]` 中去掉 `bias`。CNN 的 im2col 行数充足，因此 `ls` 在那里不会欠定。
- 适用范围：无微调场景；若剪枝后进行微调，朴素切片可能追平。

## 进阶配置

```python
import amct_pytorch as amct
from amct_pytorch.pruning import (
    SENSITIVITY_ALLOC_PRUNE_CFG,
    MOE_OUTPUT_MERGE_PRUNE_CFG,
)

# cross-layer sensitivity allocation (default uniform)
amct.prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib)
# config["allocation"] = {"strategy": "sensitivity", "ref_ratio": 0.5,
#                          "min_ratio": 0.05, "max_ratio": 0.9, "guard": "calib_nll"}

# MoE output-space expert merge (LS residual pairing + drop fallback)
amct.prune(model, MOE_OUTPUT_MERGE_PRUNE_CFG, data=calib)
# moe kwargs: keep_ratio=0.5, selector='calib_nll'|'none'

# quantization-aware saliency
cfg["methods"]["dense"]["kwargs"]["quant_cfg"] = {
    "weights_cfg": {"strategy": "channel", "symmetric": True, "dtype": "int8"}}

# size budget (keep 70%); budget_scope='total' (default) | 'prunable'
amct.prune(model, data=calib, size_budget=0.7)
# full control (target_params / budget_scope): size_budget_prune
```

## 与量化结合使用

先剪枝，再量化，最后转换（剪枝会改变张量形状，因此量化校准只有在剪枝后的权重上才正确）：

<img src="figures/prune_then_quantize.svg" alt="剪枝与量化在同一精度预算下串联" width="820">

```python
import amct_pytorch as amct
from amct_pytorch.pruning import accuracy_based_auto_prune

accuracy_based_auto_prune(model, data=calib, tolerance=0.02)  # 1) structured pruning
amct.quantize(model, quant_cfg)                                # 2) quantize on the pruned model
amct.convert(model)                                            # 3) convert the deployment model
```

若要把一份精度预算拆分到两个步骤：给剪枝一部分预算，测量其损失，再用剩余预算在剪枝后的模型上
运行量化搜索（如上图）：

```python
total_tolerance = 0.03
res = amct.accuracy_based_auto_prune(model, data=calib,
                               tolerance=total_tolerance * 0.5, evaluator=ev)
prune_drop = res.quality_drop or 0.0
remaining = max(0.0, total_tolerance - prune_drop)
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ...)   # quant budget = remaining
```

剪枝的 `evaluator=` 与量化的 `accuracy_based_auto_calibration` 共用同一个评估器：一个
`AutoCalibrationEvaluatorBase` 子类，其 `evaluate(model)` 返回精度指标（越高越好）。

```python
class MyEvaluator(AutoCalibrationEvaluatorBase):
    def evaluate(self, model):
        return my_top1_accuracy(model)

ev = MyEvaluator()
accuracy_based_auto_prune(model, data=calib, tolerance=0.02, evaluator=ev)
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ...)
```

> 内置的 `amct.ModelEvaluator` 是量化校准的数据馈送器；其 `evaluate(model, iterations)` 返回 None，
> 不能用于精度搜索。精度搜索请使用返回指标的 `AutoCalibrationEvaluatorBase` 子类。剪枝侧同时接受
> `evaluate(model)` 和 `evaluate(model, iterations)`。

## 支持的可剪枝结构

只有生产方<->消费方接口可被验证的维度才会被剪枝；其余跳过。

- **稠密 FFN** -- 仅剪枝中间维（`gate/up.out_features` + `down.in_features`）；隐藏/残差宽度保持不变。
  注意力投影（q/k/v/o）被自动排除。
- **CNN 通道** -- 生产方 `Conv2d`（`groups=1`）-> 可选 `BatchNorm2d` -> 通道匹配的消费方
  `Conv2d`/`Linear`，一并改尺寸。馈入残差 `add` 的卷积自动排除；Concat 消费方以及分组/深度可分离卷积不剪枝。
- **MoE 专家** -- 整个被路由的专家被移除；隐藏 in/out 不变。共享（始终激活）专家被排除。

| 域 | 是否支持 | 备注 |
|----|------|------|
| dense | 是 | 三 Linear `gate/up/down_proj`、融合 `gate_up_proj`（Phi-3/GLM-4）、相邻两 Linear/Conv1D（含 Bloom 风格）。Llama/Qwen2/Mistral/Qwen3 无需手动 `skip_layers` 即可剪枝。 |
| cnn | 是 | `variance_channel`（启发式切片）；`reconstruct`（im2col 最小二乘）。同层检测无法识别 ResNet 风格的独立块。 |
| moe | 是 | 传统 `nn.ModuleList` + `nn.Linear` 门控、融合批处理专家（`MixtralExperts`/`Qwen3MoeExperts` + `*TopKRouter`）、分组路由器（DeepseekV3 风格）、兄弟双张量融合专家（GraniteMoE）、嵌套路由器 bias（Ernie4.5）。 |

机械可剪枝性在多种常见架构家族（Llama/Qwen/Mixtral/GLM/Phi/GPT/DeepseekV3/GraniteMoE/Ernie4.5 等）
的 2 层微模型上有覆盖测试，见 `tests/amct_pytorch/test_pruning_real_hf_models.py` 与
`tests/amct_pytorch/test_pruning_auto_prune.py`。

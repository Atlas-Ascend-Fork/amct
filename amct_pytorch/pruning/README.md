# 结构化剪枝 (amct_pytorch.pruning)

对已实例化的 `torch.nn.Module` 进行结构化剪枝（稠密 FFN 中间维 / CNN 通道 / MoE 专家）。
模型被原地修改；统计信息通过可选的 `report=PruneReport()` 出口返回
（不保留任何进程级全局状态）。剪枝会原地改写模型
`config` 的维度（`intermediate_size` / `num_experts`），使剪枝后的模型能正确保存/重载。
本库从不下载模型；由调用方负责实例化。

## 工作原理

结构化剪枝按四步走：**打分 → 剪枝 → 恢复 →（可选）量化**。

工具接收一个**已加载权重**的模型和**少量校准数据**（几批前向即可），在校准数据上前向一遍，
为每个可剪结构（FFN 中间维通道 / CNN 通道 / MoE 专家）计算“重要性分数”，移除分数最低的部分，
必要时对保留下来的权重做一次轻量补偿（**恢复**），随后可选地接续量化。整个过程**原地**修改模型，
并同步改写 `config` 中的维度，因此剪枝后的模型可直接 `save/load`，无需额外转换步骤。

### 三个可剪枝的域

能剪什么、剪完之后什么保持不变，取决于“域”。工具只剪**生产者→消费者接口能对得上**的维度，
对不确定的一律保守跳过，保证张量形状始终自洽：

- **稠密 FFN** —— 只收窄中间维：`gate/up_proj` 的**输出通道**和 `down_proj` 的**输入通道**一起变小；
  隐藏维 / 残差宽度保持不动，注意力 q/k/v/o 投影自动排除。
- **CNN 通道** —— 沿“生产者卷积输出通道 →（可选 BatchNorm）→ 消费者输入通道”这条链一起改尺寸；
  喂入残差 `add` 的卷积、Concat（Inception）消费方、分组 / 深度可分离卷积都不剪。
- **MoE 专家** —— 整块地移除被路由的专家，并同步收缩路由器；共享（始终激活）专家保留，隐藏 in/out 不变。

> 各域支持的具体网络结构见文末 [支持的可剪枝结构](#支持的可剪枝结构)。

## 容差驱动的自动剪枝

只需指定一个可接受的精度损失 `tolerance`，工具即在 `ratio_grid` 上**二分查找**：
每试一个剪枝率 `r`，在副本上剪一遍、（可选微调后）测量精度下降量；下降在容差内则搜索转向更大的 `r`，
超出则退回更小的 `r`。最终应用**满足容差的最大剪枝率**。探测某一剪枝率时的任何剪枝 / 前向失败均
视为不可接受（不抛异常），因此在非单调情形下搜索倾向于少剪。

> ⚠️ 每个候选剪枝率都在**整模型副本**上试剪，因此搜索期间峰值内存约为模型的两倍；`size_budget`、
> 菜单择优、敏感度分配同理。固定剪枝率模式不复制。在 CPU 上剪大模型时需据此预留内存。

```python
import amct_pytorch as amct

amct.prune(model, data=calib, tolerance=0.02)   # 在容差内搜索并应用最大剪枝率，原地生效
```

当提供 `finetune_fn` 时，每个候选剪枝率在评估前都会先微调，因此被选中的剪枝率是剪枝 + 恢复后
可用的那个：

```python
from amct_pytorch.pruning import prune_finetune
amct.prune(model, cfg, data=calib, tolerance=0.05, evaluator=test_acc,
           finetune_fn=lambda m: prune_finetune(m, train_data, steps=300))
```

- 默认保真度指标无需标签：质量 = 在校准数据上与原始模型 top-1 预测的一致率。也可改为传入
  `evaluator`（`callable(model)->float`，或任何暴露 `.evaluate(model) -> float` 的对象）。

### 参数

| 参数 | 是否必填 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | 是 | - | 待剪枝的 `torch.nn.Module`（原地） |
| `data` | 视情况 | `None` | 校准数据；基于方差的方法、默认评估集需要 |
| `tolerance` | 否 | `None` | 可接受精度损失的上界（与 `evaluator` 同量纲）；传入即进入容差搜索 |
| `evaluator` | 否 | top-1 保真度 | `callable(model)->float`，或暴露 `.evaluate(model) -> float` 的对象 |
| `eval_data` | 否 | 回退到 `data` | 默认保真度指标使用的评估批 |
| `ratio_grid` | 否 | `0.1..0.8` | 候选剪枝率（升序） |
| `report` | 否 | `None` | 传入 `PruneReport()` 作为出口以取回统计信息 |

注意力投影默认被跳过（由 `skip_layers` 配置控制，见[进阶配置](#进阶配置)）。
`evaluator` / `eval_data` / `ratio_grid` / `finetune_fn` / `quant_fn` 仅在搜索模式下生效；
固定剪枝率下传入会报 `ValueError`，不会被静默忽略。

## 固定剪枝率

```python
from amct_pytorch.pruning import PruneReport
cfg = {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}}
rep = PruneReport()
amct.prune(model, cfg, data=calib, report=rep)
print(rep.as_dict())
```

**`methods` 里列出哪些域，就只剪哪些域。** 未列出的域一律固定为 `prune_ratio: 0.0`，不会按默认率
被顺带剪掉；固定剪枝率、容差搜索、size budget、菜单择优、诊断都遵循这一条。
`config=None`（一个域都不指定）时才回到三域默认值 cnn 0.30 / dense 0.50 / moe 0.50。

`kwargs` 只接受该方法实际读取的键。写错的键（例如把 `prune_ratio` 写成 `prune_rate`）会抛
`ValueError` 并列出该方法可用的键，而不是被静默忽略、回落到默认剪枝率。

### `PruneReport` 结构

`prune(..., report=rep)` 就地填充 `rep`（无进程级全局状态），通过 `rep.as_dict()` 读取：

| 字段 | 含义 |
|---|---|
| `backend` | 识别到的模型后端（`huggingface` / `modelscope` / `pretrained-module` / `torch`） |
| `params_before` / `params_after` | 剪枝前 / 后的参数量 |
| `prunable_fraction` | 位于可剪结构中的参数占比（仅 `size_budget` 模式下填充） |
| `per_layer_sparsity` | `{module_path: 稀疏率}` —— 各被剪层移除的比例 |
| `warnings` | 非致命提示（跳过的融合专家、非均匀宽度等） |
| `budget_unreachable` | 若 `size_budget` 目标无法达成则为 `True` |
| `allocation_choice` | 采用的跨层分配策略（均匀 / 敏感度） |
| `events` | 按 (域, 方法, 模块) 记录剪了什么 |

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

将 `MOE_VARIANCE_MENU_CFG` 传给 `prune`，即可运行一次校准、按菜单中的每个准则
各剪枝一份副本、在小验证集上挑出最佳并应用胜者。`prune` 检测到配置带 `menu` 就自动改走菜单择优，
无需再传 `tolerance`：

“菜单”类配置的套路是**一次校准、多方案择优**：共用同一遍校准，把菜单里每个候选各剪出一份副本，
在一个**独立的小验证集**（`eval_data`）上实测，谁**严格赢过安全回退项**才用谁；打平或没赢就保留回退项。
如此，某个在校准集上好看、但在验证集上并不稳定的方案便不会被误选。MoE 方差菜单里，各候选即不同的
重要性打分准则（`mass` / `cond_var` / `peak` / `cvxpeak`），其中 `mass` 是那个安全回退项。

```python
from amct_pytorch.pruning import MOE_VARIANCE_MENU_CFG
amct.prune(
    model, MOE_VARIANCE_MENU_CFG, data=calib, eval_data=val, evaluator=ev)  # keep val != calib
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

恢复菜单沿用上一节**完全相同的“择优菜单”机制**（见 [上一节](#moe-专家剪枝方差准则菜单)），
只是候选换成了剪枝后的补偿方式——同样是一次校准、在独立验证集上择优、只有严格胜出才替换回退项。

```python
from amct_pytorch.pruning import DENSE_RECOVERY_MENU_CFG
amct.prune(
    model, DENSE_RECOVERY_MENU_CFG, data=calib, eval_data=val, evaluator=ev)
```

CNN 通道剪枝用同一套菜单，换成 `CNN_RECOVERY_MENU_CFG`（菜单模式只剪菜单所在的域）：

```python
from amct_pytorch.pruning import CNN_RECOVERY_MENU_CFG
amct.prune(
    model, CNN_RECOVERY_MENU_CFG, data=calib, eval_data=val, evaluator=ev)
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
# 注意：敏感度分配会让各层宽度不一致，而 config 中相应字段只有单个标量，无法描述这种结构，
# 因此计数字段不会被写入，保存后也无法用原 from_pretrained 还原：需要自行保存每层尺寸并在
# 加载时重建结构。需要标准保存/加载时请让各层使用同一剪枝率。详见「剪枝后的 config 同步」。

# MoE output-space expert merge (LS residual pairing + drop fallback)
amct.prune(model, MOE_OUTPUT_MERGE_PRUNE_CFG, data=calib)
# moe kwargs: keep_ratio=0.5, selector='calib_nll'|'none'

# skip_layers: 名称匹配这些子串的层不剪。搜索模式（tolerance / size_budget / menu）会自动
# 并入注意力投影 self_attn/attention/attn；固定剪枝率模式需要自行列出。
# quant_cfg 只被 reconstruct 方法读取，low_variance 会忽略它
cfg = {"methods": {"dense": {"name": "reconstruct", "kwargs": {"prune_ratio": 0.5}}}}
cfg["skip_layers"] = ["self_attn", "lm_head", "shared_expert"]

# quantization-aware saliency
cfg["methods"]["dense"]["kwargs"]["quant_cfg"] = {
    "weights_cfg": {"strategy": "channel", "symmetric": True, "dtype": "int8"}}

# size budget: keep 70% of the parameters
amct.prune(model, cfg, data=calib, size_budget=0.7)
```

搜索模式会复制整个模型（见上文容差一节）。若设备显存不足以容纳两份，可把干净权重留在主机内存，
自行在固定剪枝率上循环——固定剪枝率不复制，设备上始终只有一份工作模型：

```python
import copy
import torch
import amct_pytorch as amct

pristine = model.cpu()                       # 基线留在主机内存，不参与计算
best = None
for prune_ratio in (0.3, 0.4, 0.5):
    trial = copy.deepcopy(pristine).to("npu:0")   # 设备上只有这一份
    cfg = {"methods": {"dense": {"name": "low_variance",
                                 "kwargs": {"prune_ratio": prune_ratio}}}}
    amct.prune(trial, cfg, data=calib)            # 固定剪枝率：不复制
    if my_evaluator(trial) >= threshold:          # 自行评估并保留满足要求的最大剪枝率
        best = prune_ratio
    del trial
    torch.npu.empty_cache()
```

## 与量化结合使用

先剪枝，再量化，最后转换（剪枝会改变张量形状，因此量化校准只有在剪枝后的权重上才正确）：

```python
import amct_pytorch as amct

amct.prune(model, data=calib, tolerance=0.02)  # 1) structured pruning
amct.quantize(model, quant_cfg)                                # 2) quantize on the pruned model
amct.convert(model)                                            # 3) convert the deployment model
```

### 恢复与量化回调：`finetune_fn` / `quant_fn`

两者都是可选回调，在容差搜索中对每个剪枝副本**评估前**应用，因此被选中的剪枝率反映的是
恢复后 / 量化后的精度：

- `finetune_fn(model)` —— 对剪枝副本做一次轻量恢复（如几步优化）。
- `quant_fn(model)` —— 对剪枝副本施加量化，使搜索计入剪枝+量化的合并损失（只会让搜索剪得更保守，不会更激进）。

```python
from amct_pytorch.pruning import prune_finetune

# 仅恢复：每个候选评估前先微调
amct.prune(model, cfg, data=calib, tolerance=0.05, evaluator=ev,
           finetune_fn=lambda m: prune_finetune(m, train_data, steps=300))

# 剪枝 + 量化共用一个容差：搜索看到的是量化后的精度
amct.prune(model, cfg, data=calib, tolerance=0.03, evaluator=ev,
           quant_fn=lambda m: amct.quantize(m, quant_cfg))
```

剪枝的 `evaluator=` 只要求一个协议：对象暴露 `evaluate(model)` 并返回精度指标（越高越好）。
不依赖任何基类，因此同一个评估器也可直接传给量化的 `accuracy_based_auto_calibration`。

```python
class MyEvaluator:
    def evaluate(self, model):
        return my_top1_accuracy(model)

ev = MyEvaluator()
amct.prune(model, data=calib, tolerance=0.02, evaluator=ev)
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ...)
```

> 内置的 `amct.ModelEvaluator` 是量化校准的数据馈送器；其 `evaluate(model, iterations)` 返回 None，
> 不能用于精度搜索。精度搜索请传入返回指标的评估器。剪枝侧同时接受
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
| moe | 是 | 传统 `nn.ModuleList` + `nn.Linear` 门控、融合批处理专家（`MixtralExperts`/`Qwen3MoeExperts` + `*TopKRouter`）、分组路由器（`n_group`/`topk_group`）、共享专家 + sigmoid 路由（noaux_tc）、兄弟双张量融合专家（GraniteMoE）、嵌套路由器 bias（Ernie4.5）。 |

机械可剪枝性在多种常见架构家族（Llama/Qwen/Mixtral/GLM/Phi/GPT/GraniteMoE/Ernie4.5 等）
的 2 层微模型上有覆盖测试，见 `tests/amct_pytorch/test_pruning_real_hf_models.py` 与
`tests/amct_pytorch/test_pruning_auto_prune.py`。

### 剪枝后的 config 同步

剪枝结束后会把新尺寸写回 `model.config`，使 `save_pretrained` 保存的配置与权重一致：

- 稠密 FFN 中间维 -> `intermediate_size` / `ffn_hidden_size` / `n_inner`
- 专家数 -> `num_local_experts` / `num_experts` / `n_routed_experts` / `n_experts`
- 每 token 专家数 -> `num_experts_per_tok` / `moe_top_k` / `top_k` / `num_selected_experts`
  会被**下调**到不超过剩余专家数（否则路由器会选到已删除的专家，首次前向即报错）

**各层专家数不一致时**：单个标量无法描述非均匀剪枝，因此 `num_experts` 一类的计数字段
**不会被写入**，只打印告警——写入任一层的数值都会与其余层不符。此时 `top_k` 仍会被下调到
**最薄一层**的专家数，保证所有层都能前向。这类模型直接 `save_pretrained` 后无法用原
`from_pretrained` 还原，需要自行保存每层尺寸并在加载时重建结构。若要避免该情况，
请让各层使用同一剪枝率（`prune_ratio`），不要使用会产生非均匀结果的敏感度分配
（`SENSITIVITY_ALLOC_PRUNE_CFG`）。稠密 FFN 各层宽度不一致时同理。

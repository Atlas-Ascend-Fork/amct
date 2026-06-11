# 结构化剪枝（experimental.pruning）

对已实例化的 `torch.nn.Module` 做结构化剪枝（dense FFN 中间维 / CNN 通道 / MoE 专家），
façade 与 `amct.quantize` 对齐：in-place 修改模型，统计经 `get_last_report()` 获取。
库内部不下载模型，调用方负责实例化（如 `from_pretrained(local_path)`）。

## 容差驱动自动剪枝（推荐，面向非专家）

不熟悉剪枝率/层结构也能用：**只说“最多可接受多少精度损失”**，自动搜索并应用
满足容差的最大剪枝率。与量化侧的 [`accuracy_based_auto_calibration`](../../../docs/zh/api/accuracy_based_auto_calibration.md)
（精度优先 + BinarySearch）同理，是其剪枝版本。

```python
import amct_pytorch.experimental.pruning as amct

# 最多损失 2%，其余全自动
amct.prune(model, data=calib, tolerance=0.02)
print(amct.get_last_auto_prune().summary())
# [auto-prune] 已应用剪枝率=0.20：权重 607,808 → 568,640 (削减 6.4%)，质量损失 0.018 ≤ 容差 0.020。
```

### 搜索 + 剪枝后微调恢复（重要）

一次性结构剪枝会让精度**崩到接近随机**；若搜索按"剪枝直后"评估，几乎选不出任何率（过度保守）。
给 `finetune_fn` 后，搜索对**每个候选率先做(轻量)微调恢复再评估**，于是选出的是"剪枝+恢复后"
真正可用的更激进的率：

```python
from amct_pytorch.experimental.pruning import finetune
amct.prune(model, cfg, data=calib, tolerance=0.05, quality_fn=test_acc,
           finetune_fn=lambda m: finetune(m, train_data, steps=300))
```

> 实测(MNIST, 真 test acc, 容差 5%)：**无 `finetune_fn` 选不出任何率(0% 剪枝)；
> 有 `finetune_fn` 选出 40% 通道剪枝**（恢复后 acc 仍在容差内）。代价：每个候选都要微调一次。

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

## CNN 通道剪枝（输出保留重构）

简单“同 parent 兄弟卷积”启发式（`variance_channel`）按通道激活方差朴素切片，适合平面/顺序 CNN。
推荐 `reconstruct`：在同一 cnn 域（识别 producer conv → bn → consumer 顺序耦合）上，剪掉低显著度
通道后用 **im2col 最小二乘重构 consumer 卷积权重**，把被移除通道的贡献吸收进保留通道，使输出尽量
保持（ThiNet/CP 式补偿，无需微调；**自研实现（LS 重构思想源自 ThiNet/Channel Pruning, ICCV'17），无任何外部/内置第三方依赖**）：

```python
from amct_pytorch.experimental.pruning import CNN_RECONSTRUCT_PRUNE_CFG
amct.prune(model, CNN_RECONSTRUCT_PRUNE_CFG, data=calib)   # calib 提供样例输入用于激活统计
# 或自定义：{"methods": {"cnn": {"name": "reconstruct",
#                                "kwargs": {"prune_ratio": 0.3, "ridge": 1e-2}}}}
```

- 自动忽略输出头(最后一个 Linear)，保持类别数不变；`skip_layers` 命中的层也忽略。
- 实测（已训练 MNIST CNN，prune 0.3）：朴素 `variance_channel` acc 0.343 vs `reconstruct` acc 0.574
  （同参数量，+23pp）——重构在已训练模型上显著低破坏。
- 校准须有代表性且多样（行数 ≥ keep×k×k 才能解最小二乘）；欠定或 consumer 非 Conv2d 时**明确告警**
  并退回朴素切片（不静默伪装为“已重构”）。
- ResNet 等把卷积包进独立 block 的结构不被同级检测识别（0 目标，自然跳过、避免破坏）——同级检测的
  已知边界（早期基于 Torch-Pruning 依赖图的实现因 acc 破坏严重已移除，改用同域输出保留重构）。

## 性能测量（真实训练模型 + NPU 实测 int8）

16 个**从头训练**的小模型（CNN@MNIST / Dense-LM·MoE-LM@WikiText）+ 20 个真实预训练 HF（10 架构系），
4 种压缩预算 × AMCT 真 int8（Ascend NPU）。预算驱动指标 = **top-1 保真度**（无标签，对原模型预测的一致率）；
报告任务指标（CNN = test acc↑，LM/MoE/HF = perplexity↓）。

### 结论 1：int8 量化近乎免费，且"量化感知剪枝"在 int8 下无收益

剪枝后再 int8（`amct.quantize→convert`）几乎不掉指标；`quant_aware=True`（用量化后权重打分）剪枝 ≈ 普通剪枝。

| 模型 | 预算 | 剪枝后 | 普通剪枝+int8 | quant_aware+int8 |
|------|------|:------:|:------------:|:----------------:|
| gpt2（dense Conv1D, ppl） | param −13.7% (率0.3) | ppl 91.07 | 91.66 | 91.57 |
| moe_E4_L4（moe 专家, ppl） | param −9.9% (率0.1) | ppl 993.1 | 993.1 | (=普通) |
| cnn_w16_d4（cnn 通道） | param−25% (−27%参数) | acc 0.954 | 0.954 (int8) | (=普通) |

> 注：上表 dense/moe 的剪枝后 ppl 用初始小校准集测得（结论 1 只关心 int8 前后**差值**，与校准集无关）；
> dense LM 用更大校准集（token 数 > FFN 中间维）的剪枝结果更优，见结论 3。

→ **int8 下量化感知剪枝可省略**（差异在小数点末位）。

### 结论 2：敏感度分配高度依赖深度 — 守卫必选（FFN 同等削减下）

逐层实测敏感度（单独剪该层时的 fidelity 下降）做 water-fill 分配：**浅模型（<10 层）6/6 全胜；
深模型（≥10 层）仅 4/19** — 深网 robust 层多剪会破坏可达性。整体 10/25。
（注：此敏感度实验为 opt-350m 纳入前的 **19 HF 子集**；结论 3 的可剪量统计已更新为 20 HF。）
因此 allocation 不设为默认，且 `guard='calib_nll'` 强制 A/B 比较，劣于 uniform 自动回退并告警。

| 模型 (层数) | uniform ppl | sensitivity ppl | 胜 |
|------|:---:|:---:|:---:|
| lm_L6_d160 (6) | 919.8 | **914.0** | sens |
| SmolLM-135M (30) | 120.2 | **117.4** | sens |
| Cerebras-GPT-111M (10) | 134.1 | **128.9** | sens |
| gpt2 (12) | **116.6** | 121.9 | uniform |
| pythia-410m (24) | **132.5** | 157.5 | uniform |

> 须用**实测 fidelity 下降**作信号（raw saliency 跨层不可比、更差）。早期 bench 因 skip 前缀
> 无尾点（layers.1 误隔离 layers.10+）曾高估收益至 23/25；修正后为上表。

### 结论 3：1% 质量预算下 dense LM 的可剪量——无微调≈0，+1-epoch 微调后中位 27%（最高 61%）

**1% 预算 max 剪枝率搜索**：每个候选率做 `prune→[finetune]→held-ppl`，二分查找"上报 ppl 损失 ≤1%"
的**最大**率（=总参数 max 削减）。用 ppl 本身门控（非 fidelity 代理）。20 个真实预训练 HF dense-LM
（10 架构系，WikiText-2 held 评估；微调用 WikiText-2 train 的 *distinct* 切片单遍 1-epoch，与评估集分离）：

| | **no-FT** 1% 预算内 max 剪 | **+1-epoch 微调** 1% 预算内 max 剪 |
|------|:---:|:---:|
| 全 20 HF（范围 / 中位） | **0~14%**（多数 0%） | **14~61% / 中位 27%** |
| santacoder (GPTBigCode) | 14.3% | **60.9%** |
| phi-1_5 (Phi) | 0% | **48.3%** |
| opt-350m (OPT) | 3.0% | **36.5%** |
| SmolLM-135M (Llama) | 0% | **23.7%** |
| Qwen2.5-1.5B (Qwen2) | 0% | **15.0%** |

**诚实结论**：训练充分的预训练 LM 上，*无微调* 的 dense FFN 剪枝在 1% 预算下几乎无可用空间
（多数模型 max 剪 0%——结构化剪枝直后损伤即超 1%；fidelity 对 LM 不敏感、ppl 却敏感）。
**1-epoch 微调**（lr 1e-5 + warmup、单遍 distinct 数据）把 max 可剪提到 **14~61%（中位 27%）**——
即"用 1% 质量换 14~61% 参数"（而非 1% 换 1%），且微调后多数模型 ppl 反优于基线（WikiText 域适应）。
这非方法失败而是结构化剪枝常态：CNN reconstruct（输出重构吸收损伤）与 MoE 专家剪枝/合并免微调，
dense LM 需微调。微调超参须为 LM 量级：lr 1e-5 + warmup（1e-4 发散）、单遍 distinct 数据
（少量批多轮 epoch 会记忆训练集、held 爆炸）。

> **opt-350m 何以从早期 19 统计缺席**：其 `project_in/project_out`（词嵌入↔隐藏维投影，注册顺序与
> 前向顺序相反）曾被误配为两层 FFN 而剪坏、前向崩溃，致其被静默排除。该误配已在 dense 域按命名排除修复
> （`domains/dense.py` 的 `_looks_like_embed_proj`），opt-350m 重新纳入 → **20 HF**，1% 预算可剪 36.5%。
> 跨域对照（同 1% 预算、+微调 max 剪）：CNN@MNIST(test acc) **31~74%**；toy MoE@WikiText **8~29%**；
> toy Dense-LM 仅 **2~7%**（embedding 主导、FFN 占比小——结构限制非质量限制）。
> 复现：`python examples/pruning/e2e_dense_finetune_recovery.py`（默认 pythia-160m，`AMCT_PRUNE_LM` 可换模型）。

### 边界
- **toy Dense-LM**（from-scratch、词表 embedding 占比极高）即便 +微调，仅剪 FFN 也难超 ~7% 总参——
  结构限制（FFN 占比太小）非质量限制；真实预训练 HF（FFN 占比正常）+微调可达 **14~61%**（见结论 3）。
- 紧质量预算（**−1%**）下 *无微调* 对预训练 LM 几乎无可行率（剪一点 ppl 即超阈值）——需微调恢复。

## 进阶配置（v0.4 新增）

```python
# 跨层敏感度分配（默认 uniform；guard 在劣化时自动回退 uniform 并告警）
P.prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib)
# config["allocation"] = {"strategy": "sensitivity", "ref_ratio": 0.5,
#                          "min_ratio": 0.05, "max_ratio": 0.9, "guard": "calib_nll"}

# MoE 输出空间专家合并（LS 残差配对 + 丢弃回退；calib NLL 与 activation_count A/B 自动择优）
P.prune(model, MOE_OUTPUT_MERGE_PRUNE_CFG, data=calib)
# moe kwargs: keep_ratio=0.5, selector='calib_nll'|'none'

# 量化感知显著度：直接复用 AMCT calibrate 同式 scale/offset
cfg["methods"]["dense"]["kwargs"]["quant_cfg"] = {
    "weights_cfg": {"strategy": "channel", "symmetric": True, "dtype": "int8"}}

# 预算口径：budget_scope='total'（默认）|'prunable'；不可达时告警 + report.budget_unreachable
size_budget_prune(model, data=calib, target_keep_ratio=0.7, budget_scope="prunable")
```

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

**联合精度预算**：`accuracy_based_auto_prune_with_budget` 把总预算在两步间**分割并顺序测量** ——
剪枝先用掉实测的一部分，量化搜索在*剪枝后的模型*上用剩余预算：

```python
from amct_pytorch.experimental.pruning import accuracy_based_auto_prune_with_budget

jb = accuracy_based_auto_prune_with_budget(
    model, data=calib, total_tolerance=0.03, prune_fraction=0.5)  # 总预算 3%，先给剪枝一半上界
print(jb.summary())                       # 剪枝实测用掉 jb.prune_drop
# 把剩余预算作为量化精度搜索的接受阈值（与量化共用同一 evaluator）
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ..., )   # 预算 = jb.remaining_tolerance
```

`jb.remaining_tolerance = max(0, total_tolerance - 实测剪枝损失)`，与量化侧
[`accuracy_based_auto_calibration`](../../../docs/zh/api/accuracy_based_auto_calibration.md) 天然衔接。

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

> ⚠️ 注意区分：graph 版内置 **`amct.ModelEvaluator`** 是量化 calibration 的**数据馈送器**，
> 其 `evaluate(model, iterations)` **返回 None**，不能用于精度搜索；误传时剪枝侧会给出明确报错。
> 需要精度搜索请用返回指标的 `AutoCalibrationEvaluatorBase` 子类。
>
> 优先级：`quality_fn` > `evaluator` > 默认 top-1 保真度。

## 与既有 graph 剪枝子系统的关系（零破坏）

本模块与既有 `classic/graph_based`（CNN 卷积通道 + retrain + Ascend 部署）**互补、不替代**：
面向 eager `torch.nn.Module` 的 LLM dense FFN / MoE 专家 / 现代 CNN，纯 additive（既有文件 0 改动）。
已对齐既有词汇以便将来纳入：共用精度抽象 `AutoCalibrationEvaluatorBase`、预算顺序衔接
（`accuracy_based_auto_prune_with_budget` → 量化 `accuracy_based_auto_calibration`）、
`PrunePass.run(model)` 与 `ModelOptimizer` 协议一致。若纳入，可作为既有入口的 `backend="eager"` 选项
（涉及改公共 API，须先按 AGENTS.md 走 Issue 共识）；在此之前以 experimental 形态零风险共存。

## 当前支持边界（以当前 transformers 实测）

| 域 | 支持 | 说明 |
|----|------|------|
| dense | ✅ | LLM 的 FFN：`gate/up/down_proj` 三 Linear、**融合 `gate_up_proj`**（Phi-3/GLM-4 风格，前半 gate 后半 up）、相邻两层 Linear/Conv1D（含 Bloom 风格自定义无参激活间隔）。dense 域结构化排除并列的注意力投影（q/k/v/o 共享输入、且不满足 in→inter→in 残差维度复原），Llama/Qwen2/Mistral/Qwen3 **无需手动 `skip_layers`** 即可安全剪枝。 |
| cnn | ✅ | `variance_channel`(启发式) 朴素切片；**`reconstruct`(im2col 最小二乘) 输出保留、低破坏**（实测已训练 MNIST CNN @prune0.3：variance 0.343 → reconstruct 0.574）。ResNet 等独立 block 结构不被同级检测识别（已知边界）。 |
| moe | ✅ | 同时支持 `nn.ModuleList` experts + `nn.Linear` gate 的老式结构，**与现代 transformers 的融合(批处理)专家**（`MixtralExperts`/`Qwen3MoeExperts` + `*TopKRouter`，沿专家维切片批张量）。 |

**架构覆盖矩阵**（config 缩小为 2 层微型模型实测，权重随机仅验证机械可剪）：26/29 家族 OK —
llama/llama3/qwen2/qwen3/qwen2_moe/mixtral/mistral/gemma/gemma2/gemma3/phi/phi3/glm4/falcon-naming(bloom)/
gpt2/gpt_neox/gpt_neo/gpt_bigcode/starcoder2/opt/xglm/olmo/olmo2/stablelm/smollm3/bloom；
不支持：mamba（SSM 无 FFN）、granite MoE 变体、falcon（fused QKV 形状）。

回归基线见 `tests/amct_pytorch/test_pruning_real_hf_models.py`、容差功能见
`tests/amct_pytorch/test_pruning_auto_prune.py`（需 transformers，用 `.venv` 运行）。

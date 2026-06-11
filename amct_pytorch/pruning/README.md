# 结构化剪枝（amct_pytorch.pruning）

对已实例化的 `torch.nn.Module` 做结构化剪枝（dense FFN 中间维 / CNN 通道 / MoE 专家），
façade 与 `amct.quantize` 对齐：in-place 修改模型，统计经 `get_last_report()` 获取。
库内部不下载模型，调用方负责实例化（如 `from_pretrained(local_path)`）。

## 容差驱动自动剪枝（推荐，面向非专家）

不熟悉剪枝率/层结构也能用：**只说“最多可接受多少精度损失”**，自动搜索并应用
满足容差的最大剪枝率。与量化侧的 [`accuracy_based_auto_calibration`](../../docs/zh/api/accuracy_based_auto_calibration.md)
（精度优先 + BinarySearch）同理，是其剪枝版本。

```python
import amct_pytorch as amct

# 最多损失 2%，其余全自动
amct.prune(model, data=calib, tolerance=0.02)
print(amct.get_last_auto_prune().summary())
# [auto-prune] applied prune_ratio=0.20: weights 607,808 → 568,640 (cut 6.4%), quality loss 0.018 ≤ tolerance 0.020.
```

### 搜索 + 剪枝后微调恢复（重要）

一次性结构剪枝会让精度**崩到接近随机**；若搜索按"剪枝直后"评估，几乎选不出任何率（过度保守）。
给 `finetune_fn` 后，搜索对**每个候选率先做(轻量)微调恢复再评估**，于是选出的是"剪枝+恢复后"
真正可用的更激进的率：

```python
from amct_pytorch.pruning import prune_finetune
amct.prune(model, cfg, data=calib, tolerance=0.05, quality_fn=test_acc,
           finetune_fn=lambda m: prune_finetune(m, train_data, steps=300))
```

> 实测(MNIST, **held-out val acc** 门控, 容差 5%)：**无 `finetune_fn` 选不出任何率(0% 剪枝)；
> 有 `finetune_fn` 选出 50% 通道剪枝**（恢复后 val acc 仍在容差内；baseline 0.857）。代价：每个候选都要微调一次。

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

> **部分配置会按默认剪其余域。** `methods` 只是*覆盖*各域默认；未指定的域仍按默认率被剪
> （cnn 0.30 / dense 0.50 / moe 0.50）。要**只剪**某域，把其余域 pin 到 `prune_ratio: 0.0`。
> 这对多域模型尤为关键——例如专家内部含 dense FFN 的 MoE：只写 moe 配置会把那层 FFN 也一并剪掉。

## CNN 通道剪枝（输出保留重构）

简单“同 parent 兄弟卷积”启发式（`variance_channel`）按通道激活方差朴素切片，适合平面/顺序 CNN。
推荐 `reconstruct`：在同一 cnn 域（识别 producer conv → bn → consumer 顺序耦合）上，剪掉低显著度
通道后用 **im2col 最小二乘重构 consumer 卷积权重**，把被移除通道的贡献吸收进保留通道，使输出尽量
保持（ThiNet/CP 式补偿，无需微调；**自研实现（LS 重构思想源自 ThiNet/Channel Pruning, ICCV'17），无任何外部/内置第三方依赖**）：

```python
from amct_pytorch.pruning import CNN_RECONSTRUCT_PRUNE_CFG
amct.prune(model, CNN_RECONSTRUCT_PRUNE_CFG, data=calib)   # calib 提供样例输入用于激活统计
# 或自定义：{"methods": {"cnn": {"name": "reconstruct",
#                                "kwargs": {"prune_ratio": 0.3, "ridge": 1e-2}}}}
```

- 自动忽略输出头(最后一个 Linear)，保持类别数不变；`skip_layers` 命中的层也忽略。
- 实测（已训练 MNIST CNN，prune 0.3）：朴素 `variance_channel` acc 0.343 vs `reconstruct` acc 0.574
  （同参数量，+23pp）——重构在已训练模型上显著低破坏。
- 校准须有代表性且多样（行数 ≥ keep×k×k 才能解最小二乘）；欠定或 consumer 非 Conv2d 时**明确告警**
  并退回朴素切片（不静默伪装为“已重构”）。
- 在**残差（ResNet）** block 中，*内部* conv→conv 耦合**会被剪枝**；仅把馈入残差 `add` 的那个 conv 自动排除（从而保持残差宽度、避免破坏）——并非整体跳过。Concat（Inception）consumer 与 分组/depthwise 卷积仍为 **0 目标**（暂缓，见"支持的可剪结构"）

## 性能测量（真实训练模型 + NPU 实测 int8）

16 个**从头训练**的小模型（CNN@MNIST / Dense-LM·MoE-LM@WikiText）+ 20 个真实预训练 HF（10 架构系），
4 种压缩预算 × AMCT 真 int8（Ascend NPU）。预算驱动指标 = **top-1 保真度**（无标签，对原模型预测的一致率）；
报告任务指标（CNN = test acc↑，LM/MoE/HF = perplexity↓）。

> **数值为参考实测**，本服务包不附带其基准脚本/原始数据；可运行的用法样例见 [`examples/algorithms/pruning/`](../../examples/algorithms/pruning/README.md)。

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
因此 allocation 不设为默认，且 `guard='calib_nll'` 强制 A/B 比较，劣于 uniform 自动回退并告警。

| 模型 (层数) | uniform ppl | sensitivity ppl | 胜 |
|------|:---:|:---:|:---:|
| lm_L6_d160 (6) | 919.8 | **914.0** | sens |
| SmolLM-135M (30) | 120.2 | **117.4** | sens |
| Cerebras-GPT-111M (10) | 134.1 | **128.9** | sens |
| gpt2 (12) | **116.6** | 121.9 | uniform |
| pythia-410m (24) | **132.5** | 157.5 | uniform |

> 须用**实测 fidelity 下降**作信号（raw saliency 跨层不可比、更差）。

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

> 下表把 1% 预算扫描扩展到 **recovery 菜单** {low_variance | reconstruct none/bias/ls | 守卫 HYBRID}（+1-epoch 微调）：

| 模型（+1-epoch 微调） | low_variance | none | bias | ls | **HYBRID** |
|------|:---:|:---:|:---:|:---:|:---:|
| pythia-160m | 7.0% | 14.0% | 14.0% | **17.5%** | 17.5% |
| gpt-neo-125M | 9.1% | 18.1% | 22.6% | **27.2%** | 27.2% |
| SmolLM-135M | 11.8% | 17.8% | 17.8% | **23.7%** | 23.7% |
| distilgpt2 (Conv1D) | **0%** | 20.8% | 20.8% | **20.8%** | 20.8% |

> no-FT 下每个模型×每种方法均为 **0%**（与头表一致）。微调后 recovery `ls` ≈ **low_variance 的 2~3 倍**；
> distilgpt2 的 Conv1D FFN 上 `low_variance` 无 `nn.Linear` 目标（**0%**）而 `reconstruct` 仍剪 **20.8%**——
> 菜单的覆盖面优势。**HYBRID** = 守卫在 val 上从 {none,bias,ls} 择优、绝不劣于 `none`（此处选中 `ls`）。

**诚实结论**：训练充分的预训练 LM 上，*无微调* 的 dense FFN 剪枝在 1% 预算下几乎无可用空间
（多数模型 max 剪 0%——结构化剪枝直后损伤即超 1%；fidelity 对 LM 不敏感、ppl 却敏感）。
**1-epoch 微调**（lr 1e-5 + warmup、单遍 distinct 数据）把 max 可剪提到 **14~61%（中位 27%）**——
即"用 1% 质量换 14~61% 参数"（而非 1% 换 1%），且微调后多数模型 ppl 反优于基线（WikiText 域适应）。
这非方法失败而是结构化剪枝常态：CNN reconstruct（输出重构吸收损伤）免微调，MoE 专家剪枝/合并在
toy MoE-LM 上免微调——但真实预训练 MoE 同样需要微调（granite-1b-a400m：无微调约 0% 在预算内、
单 epoch 微调 **17%**），与 dense LM 一致。dense LM 需微调。微调超参须为 LM 量级：lr 1e-5 + warmup（1e-4 发散）、单遍 distinct 数据
（少量批多轮 epoch 会记忆训练集、held 爆炸）。

> dense 域按命名自动排除词嵌入投影（如 OPT 的 `project_in/project_out`，注册顺序与前向相反），避免误当两层 FFN 剪坏；OPT 系模型可安全剪枝（opt-350m 1% 预算可剪 36.5%）。
> 跨域对照（同 1% 预算、+微调 max 剪）：CNN@MNIST(test acc) **31~74%**；toy MoE@WikiText **8~29%**；
> **真实预训练 MoE granite-1b-a400m 17%**（mass + 单 epoch 微调，held ppl 18.17→15.79）；
> toy Dense-LM 仅 **2~7%**（embedding 主导、FFN 占比小——结构限制非质量限制）。
> 剪枝 **接口用法**参见 [`examples/algorithms/pruning/`](../../examples/algorithms/pruning/README.md) 下的样例。

### 边界
- **toy Dense-LM**（from-scratch、词表 embedding 占比极高）即便 +微调，仅剪 FFN 也难超 ~7% 总参——
  结构限制（FFN 占比太小）非质量限制；真实预训练 HF（FFN 占比正常）+微调可达 **14~61%**（见结论 3）。
- 紧质量预算（**−1%**）下 *无微调* 对预训练 LM 几乎无可行率（剪一点 ppl 即超阈值）——需微调恢复。

### 结论 4：两个守卫式 HYBRID 均在小型确定性 demo 上实测（表 + 结论内联）

- **MoE variance-HYBRID**（合成特化路由 MoE 分类器、balanced-acc、ratio 0.5）：不均衡时方差判据胜 mass
  （均衡时也略胜）。2-config 演示（不均衡 1 与 20）实测 —— **mass 基线为
  MoE-only**（cnn/dense 置 0，与 hybrid 同域 → 公平对比；不置 0 会按默认 0.5 连专家内部 dense FFN 一起剪、虚高增益）：

  | imbalance | base | mass (MoE-only) | hybrid | pick | gain | 判据 / 选择 |
  |---|---|---|---|---|---|---|
  | 1 | 0.991 | 0.903 | 0.919 | peak | +0.017 | +0.017 / +0.000 |
  | 20 | 0.967 | 0.672 | **0.791** | cond_var | **+0.119** | +0.120 / −0.001 |

  > 单判据 TEST（无菜单）= hybrid 选中判据 → 增益来自**判据本身**，非菜单选择（选择增益 ≈ 0）。

- **Dense recovery-HYBRID** —— **真实 LM** 的证据是**结论 3 的菜单表**（预训练 LM、1% 预算、+微调：
  守卫 HYBRID 选中 `ls`、达 `low_variance` 的 ≈2~3 倍）。下表是**合成隔离阶梯演示**（训练后玩具分类器、
  激进剪 FFN、**免微调**——*不可* 与 1%-预算 LM 数值比较），在干净确定的设定下展示阶梯 none(崩塌)≪bias<ls
  与守卫"绝不劣于 none"的 val 择优：

  | ratio | base | none | bias | ls | hybrid_pick | weight↓% |
  |---|---|---|---|---|---|---|
  | 0.50 | 0.908 | 0.830 | 0.880 | **0.907** | ls | 43.9% |
  | 0.75 | 0.908 | 0.461 | 0.846 | **0.906** | ls | 65.9% |

  二者共同印证：dense FFN 剪枝的精度杠杆是**剪后 recovery**（最小二乘重构），而非选神经元的判据。
- **真·端到端（NPU，量化实发火）**：在真实 SmolLM-135M(fp16) 上 recovery-HYBRID 剪 FFN → **真实 int8 量化
  发火**（插入 `NpuWeightQuantizedLinear`）→ convert。实测 ppl base 1.649 → 剪枝(−17.8%) **1.699** → +int8
  **1.704**（量化近乎免费 Δ+0.006，与结论 1 一致）；守卫**按 ppl 选中 `bias`**（小校准 < FFN 宽 → 无最小二乘的 bias 更稳，+0.016 over none），−17.8% 参数下近乎无损。即 HYBRID 剪枝后的模型能走完整部署流水线。
  CPU 组合（prune→quantize→convert 跑通，量化在 CPU 为 no-op）见 `tests/amct_pytorch/test_pruning_quantize_compose.py`。
- 上面是 CPU **代表性**实测（合成、确定性、秒级）。更大规模验证（多架构 × 不均衡 × 校准规模）
  结论一致：方差增益随不均衡单调上升；recovery 是 dense/CNN 的杠杆，且随校准规模在 bias↔ls 间切换。

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
size_budget_prune(model, data=calib, target_keep_ratio=0.7, budget_scope="prunable")
```

## 守卫式 variance-HYBRID（MoE 专家剪枝）

方差类路由打分一次校准即可得到（无梯度/最小二乘/专家前向），在*部分* MoE 上优于按激活
频率（mass）的剪枝；但没有普适的"更好的方差"判据（判据与路由结构相关）。`guarded_variance_hybrid_prune`
不赌单一判据，而是**一次校准 → 在判据 MENU 上各剪一份副本 → 用小验证集实测择优 → 应用最优者**：

```python
from amct_pytorch.pruning import guarded_variance_hybrid_prune

res = guarded_variance_hybrid_prune(
    model, data=calib, prune_ratio=0.5, top_k=8, eval_data=val)  # val 建议与 calib 不同
print(res.summary())            # 选中的判据 + 相对 fallback(mass) 的验证增益 + 削减比例
# res.chosen_criterion / res.criterion_quality / res.gain_over_fallback(守卫保证 >= 0)
```

- **MENU**（默认）：`mass`(=`activation_count`，fallback) / `cond_var`(Var(w|sel)) /
  `peak`(max w，识别"始终强权重"的特化专家，cond 的盲点) / `cvxpeak`(Var(w|sel)×max w)。
- **守卫保证**：仅当某方差判据在**验证集**上*严格优于* fallback 时才切换，否则回退 `mass`
  —— 因此 HYBRID 在验证集上**绝不劣于** fallback（与 `accuracy_based_auto_prune` 在剪枝率上
  搜索同构，这里搜的是*判据*；二者复用同一套 deepcopy 试探 + quality_fn 守卫 + 就地应用）。
- 增益在 **类/专家使用不均衡 + 尾部敏感指标**（如分组路由的多专家 MoE）下最明显。
- **测得范围**：主要证据是结论 4 的 2-config 演示（均衡 +0.017、不均衡 20 时
  **+0.119**，MoE-only 基线）。在**真实预训练 MoE** 上判据**依赖深度** —— *全局*方差判据（每层都用）
  劣于 mass，但只在**深层**用方差（`boundary=K`）在多专家路由上优于 mass（OLMoE held ppl −25%、
  LLaDA KL 更低），见下文 **真实 MoE 上的深度分界专家选择**。**60 个 mini-suite 配置**上平均 **+6pp**
  （合成初始化专家、随不均衡上升）为 **[外部佐证，未在本包复现]**；LLaDA2.1-mini **GSM8K** 上激进*静态*
  专家剪枝会崩溃（需动态 masking）。即真实 MoE 的增益需要**深度分界**而非全局判据；"never worse"
  仍是 **VAL 保证**（granite 的少专家路由会正确回退到 mass）。
- 判据的底层打分由 `mass_variance` 的新 kwarg `variance_score`∈{`cond`(默认),`peak`,`cvxpeak`}
  提供，并按深度 `boundary` 切分：**层 idx ≤ boundary 用 mass，> boundary 用方差**（默认 `10`，在
  LLaDA2.1-mini 上验证；`-1` = 每层都用方差；`"auto"` = 逐层择优；`list[int]` = 仅这些层用方差）；
  默认 `cond` 与既有行为完全一致（向后兼容）。`guarded_variance_hybrid_prune` 的 `DEFAULT_MENU` 用全局
  `boundary=-1`；传入自定义 `menu=`（方差项设 `boundary=K`）即可搜索深度分界。

### 真实 MoE 上的深度分界专家选择

专家保留判据**依赖深度**：在**深层**（最后几层）MoE 中，每专家路由权重**方差**比激活**频率**（mass）
更能选出该保留的专家；**浅层**偏好 mass；**全方差在每个模型上都劣于深度分界**。`mass_variance` 按深度
`boundary` K 切分（层 ≤ K 用 mass，> K 用方差）。在固定剪枝率下，深层方差在真实预训练 MoE 上优于 mass：

| 真实 MoE（路由，MoE 层数） | 指标 ↓ | all-mass | var-deep（最优 K） | all-var |
|---|---|---|---|---|
| OLMoE-1B-7B（softmax，E64，16L） | held ppl @0.30 | 46.9 | **35.3**（var last 1） | 38.1 |
| LLaDA2.1-mini（分组路由，E256，19L） | KL @0.20（扩散） | 0.0392 | **0.0391**（var last 10） | 0.0634 |
| granite-1b-a400m（softmax，E32，24L） | held ppl @0.30 | **94.8** | 94.8（K=L → mass） | 1487 |

granite 的 32 专家 softmax 路由太小、缺乏可靠的每专家方差信号（all-variance ppl 1487 ≫ mass 94.8）→
boundary 搜索**正确回退到 all-mass**（绝不劣于 mass）。增益随专家数增大（OLMoE E64）。在 −1% 质量预算下
可剪量受微调门控（真实预训练 MoE 无微调 ≈ 0）：OLMoE **0% → 23%**（+1-epoch 微调），且 var-deep 在
同等削减下微调到比 mass 更低的 ppl（15%：18.96 vs 19.35；23%：**20.69 vs 21.85**）—— 深层方差的优势在
微调后依然保持。（LLaDA 是扩散模型 —— 仅 KL，无微调。）

## 守卫式 recovery-HYBRID（dense FFN / CNN 通道剪枝）

对 dense FFN / CNN 通道剪枝，决定精度的杠杆是剪枝后的 **RECOVERY（补偿方式）**，而非选神经元的
判据。`guarded_recovery_hybrid_prune` 与 variance-HYBRID 同构，但搜的是 recovery MENU：

```python
from amct_pytorch.pruning import guarded_recovery_hybrid_prune

res = guarded_recovery_hybrid_prune(
    model, data=calib, prune_ratio=0.5, domain="dense", eval_data=val)  # 或 domain="cnn"
print(res.summary())            # 选中的 recovery + 相对 none 的验证增益 + 削减比例
```

- **MENU**（默认）：`none`(朴素切片, fallback) / `bias`(均值折叠到 consumer bias, 无最小二乘) /
  `ls`(最小二乘重构 consumer 权重)。底层由 `reconstruct` 的新 kwarg `recovery`∈{`ls`(默认),`bias`,
  `none`} 提供（dense 与 cnn 都加了，默认 `ls` = 既有行为，向后兼容）。
- **守卫保证（限界要诚实）**：守卫仅当某 recovery 在**验证集**上严格优于 fallback(none) 时才切换 ——
  因此**在验证集上绝不劣于不补偿基线**。这是 **VAL 上的保证，不是 test 上的保证**：若 `eval_data`
  不代表部署分布，验证择优可能不迁移到 test（见下小校准条目）。务必用**代表部署分布**的 `eval_data`。
- **校准规模是关键旋钮**：校准充分 → `ls` 最佳；校准 < 中间维 → `ls` 欠定过拟合，`bias`(无最小二乘)更稳。
  守卫会自动随校准规模在 bias↔ls 间切换。
- **小校准安全旋钮（dense 实测）**：`bias` 折叠会过拟合校准域的激活均值，在分布漂移的 test 上可能反而
  变差。**[外部佐证，未在本包复现]** dense 54 配置 @ r0.5, no-FT：约 **2% 小校准配置在 test 上回退**（最坏 −28% ppl），且
  **更大验证集或更严的 margin 守卫都消不掉**（这是结构性 val↔test 分布失配，非采样噪声）。要 test 侧也
  稳健，传 `recovery_menu=("none", "ls")` 去掉 bias —— 以小校准约 10pp 的平均增益换取 test 侧"绝不更差"。
- **CNN 无此 floor（实测）**：CNN 通道剪枝里 im2col 行数 = N×空间，即使 16 张图也充足，`ls` 不欠定、`bias`
  极少成为验证赢家 —— **[外部佐证，未在本包复现]** 108 配置 **test 侧 0% 回退**，full 守卫已 test-side 安全。
- **测得范围**：以上 dense/CNN 数字均为 **no-FT**（部署即用、无重训练）相对比较；recovery 是这一无微调
  场景下的杠杆。若后续做微调恢复，朴素切片也可能追上，那是另一问题。

## 与量化联合使用（prune → quantize → convert）

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

**联合精度预算**：`accuracy_based_auto_prune_with_budget` 把总预算在两步间**分割并顺序测量** ——
剪枝先用掉实测的一部分，量化搜索在*剪枝后的模型*上用剩余预算：

```python
from amct_pytorch.pruning import accuracy_based_auto_prune_with_budget

jb = accuracy_based_auto_prune_with_budget(
    model, data=calib, total_tolerance=0.03, prune_fraction=0.5)  # 总预算 3%，先给剪枝一半上界
print(jb.summary())                       # 剪枝实测用掉 jb.prune_drop
# 把剩余预算作为量化精度搜索的接受阈值（与量化共用同一 evaluator）
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ..., )   # 预算 = jb.remaining_tolerance
```

`jb.remaining_tolerance = max(0, total_tolerance - 实测剪枝损失)`，与量化侧
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

> ⚠️ 注意区分：graph 版内置 **`amct.ModelEvaluator`** 是量化 calibration 的**数据馈送器**，
> 其 `evaluate(model, iterations)` **返回 None**，不能用于精度搜索；误传时剪枝侧会给出明确报错。
> 需要精度搜索请用返回指标的 `AutoCalibrationEvaluatorBase` 子类。
>
> 优先级：`quality_fn` > `evaluator` > 默认 top-1 保真度。

## 与既有 graph 剪枝子系统的关系（零破坏）

本模块与既有 `classic/graph_based`（CNN 卷积通道 + retrain + Ascend 部署）**互补、不替代**：
面向 eager `torch.nn.Module` 的 LLM dense FFN / MoE 专家 / 现代 CNN，纯 additive（既有文件 0 改动）。
已对齐既有词汇以便将来纳入：共用精度抽象 `AutoCalibrationEvaluatorBase`、预算顺序衔接
（`accuracy_based_auto_prune_with_budget` → 量化 `accuracy_based_auto_calibration`）。
当前作为独立的纯增量模块（`amct_pytorch.pruning`）与既有 graph 剪枝零风险共存。

### 支持的可剪结构
结构化剪枝只针对 producer↔consumer 接口可被验证的维度，对其余维度保守跳过（绝不误剪）：
- **Dense FFN** —— 仅剪 *中间* 维（`gate/up.out_features` + `down.in_features`）；隐藏/残差宽度（`down.out_features`）保持不变。注意力投影（q/k/v/o）自动排除。
- **CNN 通道** —— producer `Conv2d`（`groups=1`）→ 可选 `BatchNorm2d` → consumer `Conv2d`/`Linear`，且 `consumer.in_channels == producer.out_channels`；producer-out、BN、consumer-in 三者一起协同改尺寸。在**残差（ResNet）** block 中，*内部* conv→conv 耦合会被剪枝（馈入残差 `add` 的那个 conv 自动排除 → 残差宽度保持不变）。**Concat（Inception）** consumer（`in = Σ 各分支`）与 **分组/depthwise** 卷积**不剪**（0 目标）—— 验证其安全性需要跨分支依赖追踪，暂缓。
- **MoE 专家** —— 整个*路由（routed）*专家被移除；该层的输入/输出（hidden）不变。**共享（始终激活）专家从路由专家剪枝中排除**。

## 当前支持边界（以当前 transformers 实测）

| 域 | 支持 | 说明 |
|----|------|------|
| dense | ✅ | LLM 的 FFN：`gate/up/down_proj` 三 Linear、**融合 `gate_up_proj`**（Phi-3/GLM-4 风格，前半 gate 后半 up）、相邻两层 Linear/Conv1D（含 Bloom 风格自定义无参激活间隔）。dense 域结构化排除并列的注意力投影（q/k/v/o 共享输入、且不满足 in→inter→in 残差维度复原），Llama/Qwen2/Mistral/Qwen3 **无需手动 `skip_layers`** 即可安全剪枝。 |
| cnn | ✅ | `variance_channel`(启发式) 朴素切片；**`reconstruct`(im2col 最小二乘) 输出保留、低破坏**（实测已训练 MNIST CNN @prune0.3：variance 0.343 → reconstruct 0.574）。ResNet 等独立 block 结构不被同级检测识别（已知边界）。 |
| moe | ✅ | `nn.ModuleList` experts + `nn.Linear` gate 的老式结构，**与现代 transformers 的融合(批处理)专家**（`MixtralExperts`/`Qwen3MoeExperts` + `*TopKRouter`，沿专家维切片批张量）。另支持：**分组路由**（`n_routed_experts`/`n_group`/`topk_group` —— 同步专家数并把分组塌缩为 1；DeepseekV3 风格）、**兄弟双张量融合专家**（专家拆到并列子模块，如 GraniteMoE 的 `input_linear`/`output_linear`）、**嵌套路由偏置**切片（如 Ernie4.5 的 `moe_statics.e_score_correction_bias`）。由 `test_pruning_real_hf_models.py` 验证（Mixtral、Qwen3Moe、GraniteMoe、DeepseekV3、Ernie4_5_Moe）。 |

**架构覆盖矩阵**（config 缩小为 2 层微型模型实测，权重随机仅验证机械可剪）：覆盖 29 个架构家族 —
llama/llama3/qwen2/qwen3/qwen2_moe/qwen3_moe/mixtral/mistral/gemma/gemma2/gemma3/phi/phi3/glm4/
gpt2/gpt_neox/gpt_neo/gpt_bigcode/starcoder2/opt/xglm/olmo/olmo2/stablelm/smollm3/bloom/
granitemoe/deepseekv3/ernie4_5。

回归基线见 `tests/amct_pytorch/test_pruning_real_hf_models.py`、容差功能见
`tests/amct_pytorch/test_pruning_auto_prune.py`（需 transformers）。

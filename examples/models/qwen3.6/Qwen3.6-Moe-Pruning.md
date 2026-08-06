# Qwen3.6-MoE 结构化剪枝（单卡装载）

## 概述

Qwen3.6-35B-A3B 的 BF16 权重约 69 GiB，超过单张 Atlas A3 卡的可用显存（约 61 GiB），无法整模型装载。
本实践用 `amct_pytorch.pruning` 对其 MoE 专家做结构化剪枝：该模型 40 层、每层 256 个路由专家，
专家占总参数的 **92.9%**，因此小比例地移除专家即可把整模型压到单卡以内，同时保持精度基本不掉。

剪枝与量化正交——本样例只做剪枝以换取“装得下”，剪枝后的模型仍可继续走
[量化流程](Qwen3.6-Moe.md)进一步压缩。

---

## 硬件要求

产品型号：Atlas A3 Pod 系列

操作系统：Linux ARM

镜像版本：amct_llm_images:v1

驱动版本：Ascend HDK 25.5.1
> `npu-smi info` 检查 Ascend NPU 固件和驱动是否为 `25.5.1`。若未安装或版本不符，请下载
> [固件和驱动包](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=9.0.0-beta.2&driver=Ascend+HDK+25.5.1)
> 并按[指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)安装。

> **主机内存**：剪枝在 CPU 上进行（`amct.prune` 需要整模型在内存中），请确保主机可用内存 >= 150 GB。
> 剪枝完成后再把剪好的模型搬到 NPU 上评测。

---

## 为什么能装下：剪枝前后的尺寸账

| 剪枝率 | 参数量 | BF16 尺寸 | 单卡（≈61 GiB）装载 |
|---|---|---|---|
| 0%（原始） | 34.66B | 64.56 GiB | ❌ |
| 10% | 31.39B | 58.46 GiB | ✅（余量约 2.5 GiB） |
| 12% | 30.76B | 57.29 GiB | ✅ |
| 15% | 29.88B | 55.65 GiB | ✅ |

> 由于专家占总参数 92.9%，剪枝率几乎等比例地转化为整模型缩减。约 8% 是能装下的临界点，
> 10% 在留出 KV/激活余量的同时最稳妥（A3B 激活量小，KV 占用不高）。仅剪 MoE 专家、
> 共享专家与注意力保持不变。

---

## 快速启动

### 下载源码与安装

参见[环境安装&验证](../../../README.md#安装验证)。剪枝功能随 `amct_pytorch` 主包提供，无需额外开关。

### 下载权重

下载 [Qwen/Qwen3.6-35B-A3B 原始权重](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 到固定路径，
例如 `/data/models/Qwen3.6-35B-A3B`。该权重为 `bfloat16`，无需格式转换。

### 剪枝脚本

原始 35B 单卡装不下，无法整模型前向，因此剪枝在 **CPU** 上进行（`amct.prune` 需要整模型在内存中）。
MoE 专家为**融合张量**（`gate_up_proj`/`down_proj` 以专家为第 0 维），因此使用 `mass_variance`
准则——输出合并（`output_merge`）仅支持 `nn.ModuleList` 专家，对本模型不适用。

> ⚠️ **只剪 MoE 专家时，务必把 dense/cnn 显式固定为 0**。`MOE_MASSVAR_PRUNE_CFG` 只指定了 `moe`，
> 而未列出的域会按默认率剪枝（dense 0.5 / cnn 0.3）。本模型每层还有一个共享专家（shared expert，
> 属于稠密 FFN），若不固定 dense=0，它的中间维会被一并砍半，而 `config` 的 `shared_expert_intermediate_size`
> 不会同步，导致保存后无法重载。

```python
import copy
import torch
import amct_pytorch as amct
from transformers import AutoModelForCausalLM, AutoTokenizer
from amct_pytorch.pruning import MOE_MASSVAR_PRUNE_CFG, prune, PruneReport
from amct_pytorch.common.datasets.preproc import get_wiki_inputs

MODEL = "/data/models/Qwen3.6-35B-A3B"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()   # CPU

# 校准数据：几批前向即可，供方差准则统计各专家的激活分布。
# CPU 前向较慢，样例用较短序列/少量批次即可（剪枝率越保守，对统计精度要求越低）。
calib = get_wiki_inputs(tok, seq_len=512)[:1]

cfg = copy.deepcopy(MOE_MASSVAR_PRUNE_CFG)
cfg["methods"]["moe"]["kwargs"]["prune_ratio"] = 0.10   # 每层剪掉 10% 专家
cfg["methods"]["dense"] = {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}}  # 保护共享专家
cfg["methods"]["cnn"] = {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}}

rep = PruneReport()
amct.prune(model, cfg, data=calib, report=rep)          # 原地剪枝 + 改写 config
print(rep.as_dict())

n = sum(p.numel() for p in model.parameters())
print(f"pruned: {n/1e9:.2f}B  bf16 {n*2/1e9/1.073741824:.2f} GiB")

model.save_pretrained("/data/models/Qwen3.6-35B-A3B-pruned10")
tok.save_pretrained("/data/models/Qwen3.6-35B-A3B-pruned10")
```

> `amct.prune` 会同步改写 `model.config` 的维度（`num_experts` 等），使 `save_pretrained` 的 config
> 与剪后权重一致。**前提是各层同一剪枝率**（本样例即如此）；若改用敏感度分配 `SENSITIVITY_ALLOC`
> 使各层专家数不一，则计数字段无法用单个标量表达——详见
> [剪枝说明](../../../amct_pytorch/pruning/README.md#剪枝后的-config-同步)。

> ⚠️ **VL 封装的 config**：Qwen3.6-35B-A3B 是 `Qwen3_5MoeForConditionalGeneration`，config 用
> `text_config` 嵌套。`AutoModelForCausalLM` 保存出的是扁平（language-only）config，权重 key 仍是
> `model.language_model.*`。若要用 amct eval/量化流程（依赖 `text_config`）加载，请把保存目录里的
> `config.json` 换回原始 VL 结构，并只改 `text_config.num_experts` 为剪后的专家数（本例 230）。

### 精度评测

原始模型单卡装不下，因此用 amct 的 **blockwise 评测**（逐块处理，绕开整模型前向 OOM）。剪枝前后
用同一命令、同一 `seq_len`，掉点可直接比较：

```shell
# 基准（原始，单卡 blockwise）
python -m amct_pytorch.eval --model /data/models/Qwen3.6-35B-A3B \
  --model_name qwen3_6_moe --seq_len 2048 --granularity block \
  --device npu:0 --eval_mode bf16 --bit_config amct_pytorch/configs/bf16.yaml

# 剪枝后（先按上文修正 config.json，再评测）
python -m amct_pytorch.eval --model /data/models/Qwen3.6-35B-A3B-pruned10 \
  --model_name qwen3_6_moe --seq_len 2048 --granularity block \
  --device npu:0 --eval_mode bf16 --bit_config amct_pytorch/configs/bf16.yaml
```

结果（Wikitext2，seq_len=2048，blockwise）：

| 剪枝率 | ppl | 相对基准 |
|---|---|---|
| 0%（基准） | 6.7380 | — |
| 10% | 7.0915 | +5.2% |
| 12% | 7.1636 | +6.3% |
| 15% | 7.3296 | +8.8% |

> 10% 剪枝在不做微调的情况下 ppl 仅上升约 5%，换来的是从“单卡装不下”到“单卡装得下”。
> 基准 ppl 与[量化样例](Qwen3.6-Moe.md#基准测试)使用的 seq_len 不同（此处 2048，量化样例 4096），
> 因此绝对值不可横比；请以本表内部的相对掉点为准。若掉点超出预期，可传入 `finetune_fn`
> 在每个候选剪枝率评估前做一次轻量恢复（见[剪枝说明](../../../amct_pytorch/pruning/README.md#容差驱动的自动剪枝)）。

### 继续量化（可选）

剪枝已换来“单卡装得下”，如需进一步压缩，可在剪枝后的模型上继续走
[A8W4 量化流程](Qwen3.6-Moe.md)。剪枝与量化正交：先剪枝改变形状，再量化。

<details>
<summary>旧：手动保存（现已并入上面的剪枝脚本）</summary>

```python
model = model.to("cpu")
model.save_pretrained("/data/models/Qwen3.6-35B-A3B-pruned10")
tok.save_pretrained("/data/models/Qwen3.6-35B-A3B-pruned10")
```

</details>

---

## 与容差驱动搜索结合（可选）

若不想手动试剪枝率，可用 `accuracy_based_auto_prune` 指定可接受的精度损失，自动搜索满足容差的最大剪枝率：

```python
from amct_pytorch.pruning import accuracy_based_auto_prune

class PplEvaluator:
    def __init__(self, tok, wiki):
        self.tok, self.wiki = tok, wiki
    def evaluate(self, model):
        # 返回越大越好的指标，例如负 ppl
        ...

res = accuracy_based_auto_prune(
    model, MOE_MASSVAR_PRUNE_CFG, data=calib,
    tolerance=0.02, evaluator=PplEvaluator(tok, wiki))
print(res.summary())
```

> `evaluator` 只需暴露 `evaluate(model) -> float`（越大越好），不依赖任何基类。

# Qwen3.6-MoE 结构化剪枝（单卡装载）

## 概述

Qwen3.6-35B-A3B 的 BF16 权重约 64.56 GiB，超过单卡可用显存（约 61 GiB），无法整模型装载。
本实践用 `amct_pytorch.pruning` 对其 MoE 专家做结构化剪枝：该模型 40 层、每层 256 个路由专家，
专家占总参数的 **92.9%**，因此小比例地移除专家即可把整模型压到单卡以内。

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

```python
import copy
import torch
import amct_pytorch as amct
from transformers import AutoModelForCausalLM, AutoTokenizer
from amct_pytorch.pruning import MOE_MASSVAR_PRUNE_CFG, PruneReport
from amct_pytorch.common.datasets.preproc import get_wiki_inputs

MODEL = "/data/models/Qwen3.6-35B-A3B"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()   # CPU

# 校准数据：几批前向即可，供方差准则统计各专家的激活分布。
# CPU 前向较慢，样例用较短序列/少量批次即可（剪枝率越保守，对统计精度要求越低）。
calib = get_wiki_inputs(tok, seq_len=512)[:1]

cfg = copy.deepcopy(MOE_MASSVAR_PRUNE_CFG)   # 只列了 moe，因此只剪 MoE 专家
cfg["methods"]["moe"]["kwargs"]["prune_ratio"] = 0.10   # 每层剪掉 10% 专家

rep = PruneReport()
amct.prune(model, cfg, data=calib, report=rep)          # 原地剪枝 + 改写 config
print(rep.as_dict())

n = sum(p.numel() for p in model.parameters())
print(f"pruned: {n/1e9:.2f}B  bf16 {n*2/1e9/1.073741824:.2f} GiB")

model.save_pretrained("/data/models/Qwen3.6-35B-A3B-pruned10")
tok.save_pretrained("/data/models/Qwen3.6-35B-A3B-pruned10")
```

> `amct.prune` 会同步改写 `model.config` 的维度（`num_experts` 等），使 `save_pretrained` 的 config
> 与剪后权重一致。**前提是各层同一剪枝率**（本样例即如此）；若改用敏感度分配 `SENSITIVITY_ALLOC_PRUNE_CFG`
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
  --model_name qwen3_6_moe --seq_len 4096 --granularity block \
  --device npu:0 --eval_mode bf16 --bit_config amct_pytorch/configs/bf16.yaml

# 剪枝后（先按上文修正 config.json，再评测）
python -m amct_pytorch.eval --model /data/models/Qwen3.6-35B-A3B-pruned10 \
  --model_name qwen3_6_moe --seq_len 4096 --granularity block \
  --device npu:0 --eval_mode bf16 --bit_config amct_pytorch/configs/bf16.yaml
```

基准精度结果：
`Wikitext2-ppl=6.2840`

10% 剪枝后精度结果：
`Wikitext2-ppl=6.6142`

### 剪到能在单卡上训练

上面的 10% 只够推理：推理只需装下权重（2 字节/参数），训练还要同时放下梯度，所需显存翻倍。
本节把剪枝率提到 60%（专家 256 → 102，模型压到 15.27B / 28.44 GiB），并在单卡上做全参数恢复训练。

剪枝脚本同上，只把 `prune_ratio` 改为 `0.60`。

60% 剪枝后精度结果：
`Wikitext2-ppl=13.6053`

剪掉六成专家后 ppl 明显上升，需要恢复训练把精度带回来；所需数据量与步数按实际训练预算规划。
恢复训练用无动量 SGD——优化器状态会额外占显存，无动量时不产生状态：

```python
import torch
from amct_pytorch.pruning import prune_finetune

# 训练数据：prune_finetune 的默认损失接受带 input_ids 的 dict
batches = [{"input_ids": t} for t in get_wiki_inputs(tok, seq_len=512)[:300]]

model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
model.to("npu:0")

opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                      lr=1e-2, momentum=0.0)
prune_finetune(model, batches, steps=300, lr=1e-2,
               optimizer=opt, warmup=20, device="npu:0")
```

> 从 AdamW 换到 SGD 时要一并调大学习率——SGD 不对梯度做归一化，沿用 AdamW 的 `2e-5` 几乎不更新权重。

---

## 与容差驱动搜索结合（可选）

若不想手动试剪枝率，可给 `amct.prune` 传 `tolerance` 指定可接受的精度损失，自动搜索满足容差的最大剪枝率：

```python
import amct_pytorch as amct
from amct_pytorch.pruning import PruneReport

class PplEvaluator:
    def evaluate(self, model):
        # 返回越大越好的指标，例如负 ppl（可复用上文的 blockwise 评测）
        ...

report = PruneReport()
amct.prune(model, MOE_MASSVAR_PRUNE_CFG, data=calib,
           tolerance=0.02, evaluator=PplEvaluator(), report=report)
cut = 1 - report.params_after / report.params_before
print(f"weights {report.params_before:,} -> {report.params_after:,} (cut {100 * cut:.1f}%)")
```

> `evaluator` 只需暴露 `evaluate(model) -> float`（越大越好），不依赖任何基类。

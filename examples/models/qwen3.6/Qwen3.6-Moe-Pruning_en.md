# Qwen3.6-MoE Structured Pruning (single-card fit)

## Overview

The BF16 weights of Qwen3.6-35B-A3B are about 64.56 GiB, which exceeds the usable memory of a single
card (~61 GiB), so the whole model cannot be loaded on one card. This practice uses
`amct_pytorch.pruning` to structurally prune its MoE experts: the model has 40 layers with 256 routed
experts each, and experts account for **92.9%** of the parameters, so removing a small fraction of them
shrinks the whole model below the single-card limit.

Pruning and quantization are orthogonal — this sample only prunes to make the model fit; the pruned
model can still go through the [quantization flow](Qwen3.6-Moe_en.md) for further compression.

---

## Hardware Requirements

Product model: Atlas A3 Pod series

Operating system: Linux ARM

Image version: amct_llm_images:v1

Driver version: Ascend HDK 25.5.1
> Use `npu-smi info` to confirm the Ascend NPU firmware/driver is `25.5.1`. If it is missing or the
> version differs, download the
> [firmware and driver package](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=9.0.0-beta.2&driver=Ascend+HDK+25.5.1)
> and install it per the [guide](https://hiascend.com/document/redirect/CannCommunityInstSoftware).

> **Host memory**: pruning runs on CPU (`amct.prune` needs the whole model in memory), so make sure the
> host has >= 150 GB of RAM available. Only after pruning is the pruned model moved to the NPU for evaluation.

---

## Quick Start

### Get the source and install

See [Installation & Verification](../../../README_en.md#installation--verification). Pruning ships in the main
`amct_pytorch` package; no extra build flag is needed.

### Download the weights

Download the [Qwen/Qwen3.6-35B-A3B original weights](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) to a fixed
path, e.g. `/data/models/Qwen3.6-35B-A3B`. The weights are `bfloat16`, so no format conversion is needed.

### Pruning script

The original 35B does not fit one card, so it cannot be forwarded whole; pruning therefore runs on **CPU**
(`amct.prune` needs the whole model in memory). The MoE experts are **fused tensors** (`gate_up_proj` /
`down_proj` with experts on dim 0), so the `mass_variance` criterion is used — output merge (`output_merge`)
only supports `nn.ModuleList` experts and does not apply here.

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

# Calibration data: a few forward passes are enough for the variance criterion to profile each
# expert's activations. CPU forward is slow, so a short sequence / few batches is
# enough (a conservative prune ratio needs less statistical precision).
calib = get_wiki_inputs(tok, seq_len=512)[:1]

cfg = copy.deepcopy(MOE_MASSVAR_PRUNE_CFG)   # lists only moe, so only the MoE experts are pruned
cfg["methods"]["moe"]["kwargs"]["prune_ratio"] = 0.10   # drop 10% of experts per layer

rep = PruneReport()
amct.prune(model, cfg, data=calib, report=rep)          # in-place prune + config rewrite
print(rep.as_dict())

n = sum(p.numel() for p in model.parameters())
print(f"pruned: {n/1e9:.2f}B  bf16 {n*2/1e9/1.073741824:.2f} GiB")

model.save_pretrained("/data/models/Qwen3.6-35B-A3B-pruned10")
tok.save_pretrained("/data/models/Qwen3.6-35B-A3B-pruned10")
```

> `amct.prune` rewrites the `model.config` dims (`num_experts` etc.) in place so `save_pretrained` writes a
> config that matches the pruned weights. **This holds when every layer uses the same prune ratio** (as here);
> with sensitivity allocation (`SENSITIVITY_ALLOC_PRUNE_CFG`) the per-layer counts cannot be a single scalar — see the
> [pruning notes](../../../amct_pytorch/pruning/README_en.md#config-sync-after-pruning).

> ⚠️ **VL-wrapped config**: Qwen3.6-35B-A3B is a `Qwen3_5MoeForConditionalGeneration` whose config nests a
> `text_config`. `AutoModelForCausalLM` saves a flat (language-only) config, while the weight keys stay
> `model.language_model.*`. To load it through the amct eval / quantization flow (which reads `text_config`),
> put the original VL-structured `config.json` back into the saved directory and only change
> `text_config.num_experts` to the pruned count (230 here).

### Accuracy evaluation

The original model does not fit one card, so ppl is measured with amct's **blockwise eval** (processed block
by block, avoiding a whole-model forward OOM). Baseline and pruned use the same command and same `seq_len`, so
the drop is directly comparable:

```shell
# baseline (original, single-card blockwise)
python -m amct_pytorch.eval --model /data/models/Qwen3.6-35B-A3B \
  --model_name qwen3_6_moe --seq_len 4096 --granularity block \
  --device npu:0 --eval_mode bf16 --bit_config amct_pytorch/configs/bf16.yaml

# pruned (fix config.json as noted above first, then eval)
python -m amct_pytorch.eval --model /data/models/Qwen3.6-35B-A3B-pruned10 \
  --model_name qwen3_6_moe --seq_len 4096 --granularity block \
  --device npu:0 --eval_mode bf16 --bit_config amct_pytorch/configs/bf16.yaml
```

Baseline accuracy:
`Wikitext2-ppl=6.2840`

Accuracy after 10% pruning:
`Wikitext2-ppl=6.6142`

### Pruning until it trains on one card

The 10% above only covers inference: inference has to hold the weights (2 bytes/param), while training also
has to hold the gradients, which doubles the requirement. This section raises the prune ratio to 60%
(experts 256 -> 102, model down to 15.27B / 28.44 GiB) and runs full-parameter recovery training on one card.

The pruning script is the same as above with `prune_ratio` set to `0.60`.

Accuracy after 60% pruning:
`Wikitext2-ppl=13.6053`

Dropping six out of ten experts raises ppl noticeably, so recovery training is needed to bring accuracy back;
size the data and the step count against your own training budget. Recovery uses momentum-free SGD --
optimizer state costs extra memory, and without momentum there is none:

```python
import torch
from amct_pytorch.pruning import prune_finetune

# training data: prune_finetune's default loss accepts dicts carrying input_ids
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

> Raise the learning rate when moving from AdamW to SGD -- SGD does not normalise gradients, so AdamW's `2e-5`
> barely updates the weights.

---

## Combining with tolerance-driven search (optional)

To avoid trying prune ratios by hand, pass `tolerance` to `amct.prune`: it takes an acceptable accuracy
loss and searches for the largest ratio that stays within it:

```python
import amct_pytorch as amct
from amct_pytorch.pruning import PruneReport

class PplEvaluator:
    def evaluate(self, model):
        # return a higher-is-better metric, e.g. negative ppl (reuse the blockwise eval above)
        ...

report = PruneReport()
amct.prune(model, MOE_MASSVAR_PRUNE_CFG, data=calib,
           tolerance=0.02, evaluator=PplEvaluator(), report=report)
cut = 1 - report.params_after / report.params_before
print(f"weights {report.params_before:,} -> {report.params_after:,} (cut {100 * cut:.1f}%)")
```

> The `evaluator` only needs to expose `evaluate(model) -> float` (higher is better); no base class is required.

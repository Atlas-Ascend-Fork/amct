# Qwen3.6-MoE Structured Pruning (single-card fit)

## Overview

The BF16 weights of Qwen3.6-35B-A3B are about 69 GiB, which exceeds the usable memory of a single
Atlas A3 card (~61 GiB), so the whole model cannot be loaded on one card. This practice uses
`amct_pytorch.pruning` to structurally prune its MoE experts: the model has 40 layers with 256 routed
experts each, and experts account for **92.9%** of the parameters, so removing a small fraction of them
shrinks the whole model below the single-card limit while keeping accuracy essentially intact.

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

## Why it fits: the size arithmetic

| Prune ratio | Params | BF16 size | Fits one card (~61 GiB) |
|---|---|---|---|
| 0% (original) | 34.66B | 64.56 GiB | ❌ |
| 10% | 31.39B | 58.46 GiB | ✅ (~2.5 GiB headroom) |
| 12% | 30.76B | 57.29 GiB | ✅ |
| 15% | 29.88B | 55.65 GiB | ✅ |

> Because experts are 92.9% of the parameters, the prune ratio translates almost proportionally into whole-model
> reduction. ~8% is the break-even point for fitting; 10-12% is the safe range once KV/activation headroom is
> accounted for (A3B has small activations, so KV usage is modest).

---

## Quick Start

### Get the source and install

See [Installation & Verification](../../../README_en.md#installation-verification). Pruning ships in the main
`amct_pytorch` package; no extra build flag is needed.

### Download the weights

Download the [Qwen/Qwen3.6-35B-A3B original weights](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) to a fixed
path, e.g. `/data/models/Qwen3.6-35B-A3B`. The weights are `bfloat16`, so no format conversion is needed.

### Pruning script

The original 35B does not fit one card, so it cannot be forwarded whole; pruning therefore runs on **CPU**
(`amct.prune` needs the whole model in memory). The MoE experts are **fused tensors** (`gate_up_proj` /
`down_proj` with experts on dim 0), so the `mass_variance` criterion is used — output merge (`output_merge`)
only supports `nn.ModuleList` experts and does not apply here.

> ⚠️ **To prune only the MoE experts, pin dense/cnn to 0 explicitly.** `MOE_MASSVAR_PRUNE_CFG` specifies only
> `moe`, and unlisted domains are pruned at their defaults (dense 0.5 / cnn 0.3). Each layer here also has a
> shared expert (a dense FFN); without pinning dense=0 its intermediate dim gets halved too, and the config's
> `shared_expert_intermediate_size` is not synced, so the saved model fails to reload.

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

# Calibration data. CPU forward is slow, so a short sequence / few batches is
# enough (a conservative prune ratio needs less statistical precision).
calib = get_wiki_inputs(tok, seq_len=512)[:1]

cfg = copy.deepcopy(MOE_MASSVAR_PRUNE_CFG)
cfg["methods"]["moe"]["kwargs"]["prune_ratio"] = 0.10   # drop 10% of experts per layer
cfg["methods"]["dense"] = {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}}  # protect shared expert
cfg["methods"]["cnn"] = {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}}

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
> with sensitivity allocation (`SENSITIVITY_ALLOC`) the per-layer counts cannot be a single scalar — see the
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
  --model_name qwen3_6_moe --seq_len 2048 --granularity block \
  --device npu:0 --eval_mode bf16 --bit_config amct_pytorch/configs/bf16.yaml

# pruned (fix config.json as noted above first, then eval)
python -m amct_pytorch.eval --model /data/models/Qwen3.6-35B-A3B-pruned10 \
  --model_name qwen3_6_moe --seq_len 2048 --granularity block \
  --device npu:0 --eval_mode bf16 --bit_config amct_pytorch/configs/bf16.yaml
```

Results (Wikitext2, seq_len=2048, blockwise):

| Prune ratio | ppl | vs baseline |
|---|---|---|
| 0% (baseline) | 6.7380 | — |
| 10% | 7.0915 | +5.2% |
| 12% | 7.1636 | +6.3% |
| 15% | 7.3296 | +8.8% |

> At 10%, ppl rises only ~5% with no finetuning, in exchange for going from "does not fit one card" to "fits
> one card". The baseline ppl uses a different seq_len than the [quantization sample](Qwen3.6-Moe_en.md#baseline-test)
> (2048 vs 4096), so absolute values are not cross-comparable — read the relative drop within this table. If
> the drop is larger than acceptable, pass a `finetune_fn` for a light recovery before each candidate ratio is
> evaluated (see [pruning notes](../../../amct_pytorch/pruning/README_en.md#tolerance-driven-auto-pruning)).

### Continue to quantization (optional)

Pruning has bought the single-card fit; for further compression, run the pruned model through the
[A8W4 quantization flow](Qwen3.6-Moe_en.md). Pruning and quantization are orthogonal: prune to change shape
first, then quantize.

---

## Combining with tolerance-driven search (optional)

To avoid trying prune ratios by hand, `accuracy_based_auto_prune` takes an acceptable accuracy loss and searches
for the largest ratio that stays within tolerance:

```python
from amct_pytorch.pruning import accuracy_based_auto_prune

class PplEvaluator:
    def __init__(self, tok, wiki):
        self.tok, self.wiki = tok, wiki
    def evaluate(self, model):
        # return a higher-is-better metric, e.g. negative ppl
        ...

res = accuracy_based_auto_prune(
    model, MOE_MASSVAR_PRUNE_CFG, data=calib,
    tolerance=0.02, evaluator=PplEvaluator(tok, wiki))
print(res.summary())
```

> The `evaluator` only needs to expose `evaluate(model) -> float` (higher is better); no base class is required.

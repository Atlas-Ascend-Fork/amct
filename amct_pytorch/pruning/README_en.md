# Structured Pruning (amct_pytorch.pruning)

Structured pruning of an instantiated `torch.nn.Module` (dense FFN intermediate dim / CNN channels /
MoE experts). The model is modified in-place; statistics are returned via an optional `report=PruneReport()`
sink. Pruning rewrites the
model `config` dims (`intermediate_size` / `num_experts`) in place so the pruned model saves/reloads
correctly. The library never downloads models; the caller instantiates them.

## How it works

Structured pruning runs in four steps: **score -> prune -> recover -> (optional) quantize**.

The tool takes a model with **weights already loaded** and a small amount of **calibration data** (a few
forward batches). It runs one forward pass over that data, assigns an importance score to every prunable
structure (FFN intermediate channels / CNN channels / MoE experts), removes the lowest-scoring part, and
optionally applies a light compensation to the surviving weights (**recovery**), after which quantization
may be chained. All changes are made **in-place** and the `config` dims are rewritten as well, so the pruned
model `save/load`s directly with no extra conversion step.

### The three prunable domains

What can be pruned, and what stays fixed afterwards, depends on the "domain". The tool only prunes
dimensions whose **producer -> consumer interface can be verified**, and conservatively skips anything
uncertain, so tensor shapes always stay self-consistent:

- **Dense FFN** -- shrinks the intermediate dim only: the **out_features** of `gate/up_proj` and the
  **in_features** of `down_proj` shrink together; the hidden / residual width is untouched, and attention
  q/k/v/o projections are auto-excluded.
- **CNN channels** -- resizes along the chain "producer conv out-channels -> (optional BatchNorm) ->
  consumer in-channels"; the conv feeding a residual `add`, Concat (Inception) consumers, and
  grouped / depthwise convs are not pruned.
- **MoE experts** -- removes whole routed experts and shrinks the router accordingly; shared
  (always-active) experts are kept and hidden in/out is unchanged.

> The concrete network structures supported per domain are listed at the end in
> [Supported prunable structures](#supported-prunable-structures).

## Tolerance-driven auto pruning

Only an acceptable accuracy loss `tolerance` is specified; the tool runs a **binary search** over
`ratio_grid`. For each candidate ratio `r` it prunes a copy, optionally finetunes, and measures the
resulting quality drop; if the drop is within tolerance the search moves to a larger `r`, otherwise it backs
off to a smaller `r`. The **largest ratio that satisfies the tolerance** is applied. Any prune/forward
failure while probing a ratio is treated as unacceptable (no exception), so under non-monotonicity the
search prunes less.

> ⚠️ Every candidate ratio is tried on a **full copy of the model**, so peak memory during the search is
> roughly twice the model; the same holds for `size_budget`, menu selection and sensitivity allocation.
> The fixed prune ratio does not copy. Budget for this when pruning a large model on CPU.

```python
import amct_pytorch as amct

amct.prune(model, data=calib, tolerance=0.02)   # search + apply the largest ratio within tolerance, in-place
```

When `finetune_fn` is provided, each candidate ratio is finetuned before evaluation, so the selected ratio
is the one usable after prune + recovery:

```python
from amct_pytorch.pruning import prune_finetune
amct.prune(model, cfg, data=calib, tolerance=0.05, evaluator=test_acc,
           finetune_fn=lambda m: prune_finetune(m, train_data, steps=300))
```

- Default fidelity metric needs no labels: quality = top-1 prediction agreement with the original model on
  the calibration data. An `evaluator` (`callable(model)->float`, or any object exposing `.evaluate(model) -> float`) can be passed instead.

### Parameters

| Parameter | Required | Default | Description |
|------|------|--------|------|
| `model` | yes | - | `torch.nn.Module` to prune (in-place) |
| `data` | depends | `None` | Calibration data; required by variance-based methods, default eval set |
| `tolerance` | no | `None` | Upper bound on acceptable accuracy loss (same scale as `evaluator`); passing it selects the tolerance search |
| `evaluator` | no | top-1 fidelity | `callable(model)->float`, or an object exposing `.evaluate(model) -> float` |
| `eval_data` | no | falls back to `data` | Evaluation batch for the default fidelity metric |
| `ratio_grid` | no | `0.1..0.8` | Candidate prune ratios (ascending) |
| `report` | no | `None` | Pass a `PruneReport()` as the sink to get the statistics back |

Attention projections are skipped by default (driven by the `skip_layers` config; see
[Advanced configuration](#advanced-configuration)). `evaluator` / `eval_data` / `ratio_grid` /
`finetune_fn` / `quant_fn` only take effect in a search mode; passing them with a fixed prune
ratio raises `ValueError` rather than being silently ignored.

## Fixed prune ratio

```python
from amct_pytorch.pruning import PruneReport
cfg = {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}}
rep = PruneReport()
amct.prune(model, cfg, data=calib, report=rep)
print(rep.as_dict())
```

**Whichever domains `methods` lists are the only ones pruned.** Any domain left out is pinned to
`prune_ratio: 0.0` instead of being cut at a default; this holds for the fixed ratio, the tolerance
search, size budget, menu selection and diagnosis alike. Only `config=None` — naming no domain at
all — falls back to the three-domain defaults cnn 0.30 / dense 0.50 / moe 0.50.

`kwargs` accepts only the keys the method actually reads. A misspelled key — `prune_rate` for
`prune_ratio`, say — raises `ValueError` listing the keys that method accepts, instead of being
ignored and leaving the default prune ratio in force.

### The `PruneReport` structure

`prune(..., report=rep)` fills `rep` in place (no process-global state); read it with `rep.as_dict()`:

| Field | Meaning |
|---|---|
| `backend` | detected model backend (`huggingface` / `modelscope` / `pretrained-module` / `torch`) |
| `params_before` / `params_after` | parameter count before / after pruning |
| `prunable_fraction` | share of parameters that lived in prunable structures (populated in `size_budget` mode only) |
| `per_layer_sparsity` | `{module_path: sparsity}` — fraction removed per pruned layer |
| `warnings` | non-fatal notes (skipped fused experts, non-uniform widths, ...) |
| `budget_unreachable` | `True` if a `size_budget` target could not be met |
| `allocation_choice` | which cross-layer allocation was applied (uniform / sensitivity) |
| `events` | per-(domain, method, module) log of what was pruned |

## CNN channel pruning

`variance_channel` slices channels by activation variance (same-parent sibling conv heuristic).
`reconstruct` removes low-saliency channels then reconstructs the consumer conv weights via im2col least
squares to preserve the output (no finetuning):

```python
from amct_pytorch.pruning import CNN_RECONSTRUCT_PRUNE_CFG
amct.prune(model, CNN_RECONSTRUCT_PRUNE_CFG, data=calib)
# custom: {"methods": {"cnn": {"name": "reconstruct",
#                              "kwargs": {"prune_ratio": 0.3, "ridge": 1e-2}}}}
```

- The output head (last Linear) and `skip_layers` matches are ignored automatically.
- Calibration must supply enough rows (>= keep x k x k) to solve the least squares; when underdetermined or
  the consumer is not Conv2d, a warning is emitted and naive slicing is used.
- In a residual (ResNet) block the interior conv->conv coupling is pruned; the conv feeding the residual
  `add` is auto-excluded. Concat (Inception) consumers and grouped/depthwise convs are not pruned.

## MoE expert pruning (variance criterion menu)

> Advanced (not in `__all__`) -- import via `from amct_pytorch.pruning import ...`.

Pass `MOE_VARIANCE_MENU_CFG` to `prune` to run one calibration, prune a copy per criterion in the menu,
pick the best on a small validation set, and apply the winner. `prune` switches to menu selection as soon
as it sees a config carrying a `menu`, so no `tolerance` is needed:

The "menu"-style configs follow one pattern: **calibrate once, then pick the best of several candidates**.
They share a single calibration pass, prune one copy per menu candidate, measure each on a **separate small
validation set** (`eval_data`), and keep a candidate only if it **strictly beats the safe fallback**; on a
tie or a loss the fallback is kept. A candidate that merely looks good on the calibration set but is
unstable on validation is therefore not mis-selected. In the MoE variance menu the candidates are the
different importance-scoring criteria (`mass` / `cond_var` / `peak` / `cvxpeak`), with `mass` as the safe
fallback.

```python
from amct_pytorch.pruning import MOE_VARIANCE_MENU_CFG
amct.prune(
    model, MOE_VARIANCE_MENU_CFG, data=calib, eval_data=val, evaluator=ev)  # keep val != calib
```

- Menu: `mass` (= `activation_count`, fallback) / `cond_var` / `peak` / `cvxpeak`.
- A variance criterion replaces `mass` only on a strict win on the validation set, else falls back to `mass`.
- Per-criterion score is the `mass_variance` kwarg `variance_score` in {`cond` (default), `peak`, `cvxpeak`},
  with depth split `boundary`: mass on layers <= boundary, variance on layers > boundary (default `10`;
  `-1` = all-variance; `"auto"` = per-layer; `list[int]` = those layers). `MOE_VARIANCE_MENU_CFG` defaults to
  `boundary=-1`; pass a custom menu with `boundary=K` to search depth splits.

## Recovery menu (dense FFN / CNN channel pruning)

> Advanced (not in `__all__`) -- import via `from amct_pytorch.pruning import ...`.

Pass `DENSE_RECOVERY_MENU_CFG` to search the post-prune recovery menu:

The recovery menu reuses the **exact same best-of-menu mechanism** as the previous section
(see the [previous section](#moe-expert-pruning-variance-criterion-menu)); only the candidates change to
post-prune compensation strategies -- still one calibration, best-of-menu on a held-out validation set,
replace the fallback only on a strict win.

```python
from amct_pytorch.pruning import DENSE_RECOVERY_MENU_CFG
amct.prune(
    model, DENSE_RECOVERY_MENU_CFG, data=calib, eval_data=val, evaluator=ev)
```

CNN channel pruning uses the same menu via `CNN_RECOVERY_MENU_CFG` (menu mode prunes only the
domain the menu is keyed on):

```python
from amct_pytorch.pruning import CNN_RECOVERY_MENU_CFG
amct.prune(
    model, CNN_RECOVERY_MENU_CFG, data=calib, eval_data=val, evaluator=ev)
```

- Menu: `none` (naive slice, fallback) / `bias` (mean-fold into consumer bias) / `ls` (least-squares
  reconstruction). Backed by the `reconstruct` kwarg `recovery` in {`ls` (default), `bias`, `none`}.
- The fallback is replaced only on a strict win on the validation set; use `eval_data` representative of deployment.
- Ample calib favors `ls`; calib < intermediate width makes `ls` underdetermined, `bias` is more robust.
- `bias`-fold can overfit the calibration mean on a shifted test set; to keep a test-side floor, drop `bias`
  from `cfg["methods"]["dense"]["menu"]`. CNN im2col rows are abundant, so `ls` is not underdetermined there.
- Scope: no-fine-tune regime; with post-prune fine-tuning the naive slice may catch up.

## Advanced configuration

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
# NOTE: sensitivity allocation leaves the layers at different widths, and the matching config
# fields hold a single scalar that cannot describe that, so the count fields are left unwritten
# and the saved model does not come back through from_pretrained: keep the per-layer sizes
# yourself and rebuild the structure on load. Use one prune ratio for every layer when you need
# standard save/load. See "Config sync after pruning".

# MoE output-space expert merge (LS residual pairing + drop fallback)
amct.prune(model, MOE_OUTPUT_MERGE_PRUNE_CFG, data=calib)
# moe kwargs: keep_ratio=0.5, selector='calib_nll'|'none'

# skip_layers: layers whose name contains any of these are left untouched. The search modes
# (tolerance / size_budget / menu) merge in self_attn/attention/attn automatically; the fixed
# prune-ratio mode does not, so list them yourself.
# quant_cfg is read by the reconstruct method only; low_variance ignores it
cfg = {"methods": {"dense": {"name": "reconstruct", "kwargs": {"prune_ratio": 0.5}}}}
cfg["skip_layers"] = ["self_attn", "lm_head", "shared_expert"]

# quantization-aware saliency
cfg["methods"]["dense"]["kwargs"]["quant_cfg"] = {
    "weights_cfg": {"strategy": "channel", "symmetric": True, "dtype": "int8"}}

# size budget: keep 70% of the parameters
amct.prune(model, cfg, data=calib, size_budget=0.7)
```

The search modes copy the whole model (see the tolerance section above). When the device cannot hold
two copies, keep the pristine weights in host memory and loop over fixed prune ratios yourself — the
fixed ratio does not copy, so the device only ever holds one working model:

```python
import copy
import torch
import amct_pytorch as amct

pristine = model.cpu()                        # baseline stays in host memory
best = None
for prune_ratio in (0.3, 0.4, 0.5):
    trial = copy.deepcopy(pristine).to("npu:0")    # the only model on the device
    cfg = {"methods": {"dense": {"name": "low_variance",
                                 "kwargs": {"prune_ratio": prune_ratio}}}}
    amct.prune(trial, cfg, data=calib)             # fixed ratio: no copy
    if my_evaluator(trial) >= threshold:           # keep the largest ratio that passes
        best = prune_ratio
    del trial
    torch.npu.empty_cache()
```

## Combined use with quantization

Prune first, then quantize, then convert (pruning changes tensor shapes, so quantization calibration is
only correct on the pruned weights):

```python
import amct_pytorch as amct

amct.prune(model, data=calib, tolerance=0.02)  # 1) structured pruning
amct.quantize(model, quant_cfg)                                # 2) quantize on the pruned model
amct.convert(model)                                            # 3) convert the deployment model
```

### Recovery and quantization callbacks: `finetune_fn` / `quant_fn`

Both are optional callbacks applied to each pruned copy **before** it is evaluated during the
tolerance search, so the chosen ratio reflects the post-recovery / post-quantization quality:

- `finetune_fn(model)` — a light recovery pass on the pruned copy (e.g. a few optimizer steps).
- `quant_fn(model)` — applies quantization to the pruned copy so the search accounts for the
  combined prune+quant loss (it never makes the search prune *more*).

```python
from amct_pytorch.pruning import prune_finetune

# recovery only: finetune each candidate before measuring
amct.prune(model, cfg, data=calib, tolerance=0.05, evaluator=ev,
           finetune_fn=lambda m: prune_finetune(m, train_data, steps=300))

# prune + quant under one tolerance: the search sees the quantized quality
amct.prune(model, cfg, data=calib, tolerance=0.03, evaluator=ev,
           quant_fn=lambda m: amct.quantize(m, quant_cfg))
```

The pruning `evaluator=` requires only a protocol: an object exposing `evaluate(model)` that returns an
accuracy metric (higher is better). No base class is involved, so the same evaluator can also be handed to
quantization's `accuracy_based_auto_calibration`.

```python
class MyEvaluator:
    def evaluate(self, model):
        return my_top1_accuracy(model)

ev = MyEvaluator()
amct.prune(model, data=calib, tolerance=0.02, evaluator=ev)
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ...)
```

> The built-in `amct.ModelEvaluator` is a data feeder for quantization calibration; its
> `evaluate(model, iterations)` returns None and cannot be used for accuracy search. For accuracy search use
> a metric-returning evaluator. The pruning side accepts both `evaluate(model)`
> and `evaluate(model, iterations)`.

## Supported prunable structures

Only dimensions whose producer<->consumer interface can be verified are pruned; the rest are skipped.

- **Dense FFN** -- prunes the intermediate dim only (`gate/up.out_features` + `down.in_features`); hidden/residual
  width is preserved. Attention projections (q/k/v/o) are auto-excluded.
- **CNN channels** -- producer `Conv2d` (`groups=1`) -> optional `BatchNorm2d` -> consumer `Conv2d`/`Linear` with
  matching channels, co-resized. Residual `add`-feeding conv auto-excluded; Concat consumers and
  grouped/depthwise convs not pruned.
- **MoE experts** -- whole routed experts removed; hidden in/out unchanged. Shared (always-active) experts excluded.

| Domain | Supported | Notes |
|----|------|------|
| dense | yes | Three-Linear `gate/up/down_proj`, fused `gate_up_proj` (Phi-3/GLM-4), adjacent two Linear/Conv1D (incl. Bloom-style). Llama/Qwen2/Mistral/Qwen3 prune without manual `skip_layers`. |
| cnn | yes | `variance_channel` (heuristic slicing); `reconstruct` (im2col least squares). ResNet-style independent blocks not recognized by same-level detection. |
| moe | yes | Legacy `nn.ModuleList` + `nn.Linear` gate, fused batched experts (`MixtralExperts`/`Qwen3MoeExperts` + `*TopKRouter`), grouped routers (`n_group`/`topk_group`), shared experts + sigmoid routing (noaux_tc), sibling two-tensor fused experts (GraniteMoE), nested router-bias (Ernie4.5). |

Mechanical prunability is covered on 2-layer micro models across common architecture families
(Llama/Qwen/Mixtral/GLM/Phi/GPT/GraniteMoE/Ernie4.5, etc.); see
`tests/amct_pytorch/test_pruning_real_hf_models.py` and
`tests/amct_pytorch/test_pruning_auto_prune.py`.

### Config sync after pruning

Post-prune sizes are written back to `model.config` so that a `save_pretrained` config
matches the pruned weights:

- dense FFN intermediate dim -> `intermediate_size` / `ffn_hidden_size` / `n_inner`
- expert count -> `num_local_experts` / `num_experts` / `n_routed_experts` / `n_experts`
- experts per token -> `num_experts_per_tok` / `moe_top_k` / `top_k` /
  `num_selected_experts`, **lowered** to at most the surviving expert count (otherwise the
  router selects experts that no longer exist and the first forward fails)

**When expert counts differ across layers**: a single scalar cannot describe non-uniform
pruning, so the count fields are **left unwritten** and only a warning is logged — writing
any one layer's value would contradict the others. `top_k` is still lowered to the count of
the **thinnest** layer so every layer can run. Such a model cannot be restored by plain
`from_pretrained` after `save_pretrained`; save the per-layer sizes yourself and rebuild the
structure at load time. To avoid this, prune every layer at the same `prune_ratio` rather
than using sensitivity allocation (`SENSITIVITY_ALLOC_PRUNE_CFG`), which produces non-uniform results.
The same applies to non-uniform dense FFN widths.

# Structured Pruning (amct_pytorch.pruning)

Structured pruning of an already-instantiated `torch.nn.Module` (dense FFN intermediate dimension / CNN channels / MoE experts).
The façade is aligned with `amct.quantize`: the model is modified in-place, with no process-global state — statistics are retrieved via the optional `report=PruneReport()` sink, and search results from the value returned by `accuracy_based_auto_prune` / `size_budget_prune`.
(Unlike `amct.quantize`, structured pruning rewrites the model's `config` dims (`intermediate_size` / `num_experts`) in place, so the pruned model saves/reloads correctly.)
The library never downloads models; the caller is responsible for instantiation (e.g. `from_pretrained(local_path)`).

# Pruning features

## Tolerance-driven auto pruning (recommended, for non-experts)

Usable without knowledge of prune ratios or layer structure: **specify only "how much accuracy loss is acceptable"**, and the largest prune ratio that satisfies the tolerance is searched for and applied automatically. Analogous to the quantization-side [`accuracy_based_auto_calibration`](../../docs/zh/api/accuracy_based_auto_calibration.md)
(accuracy-first + BinarySearch); this is its pruning counterpart.

```python
import amct_pytorch as amct

# lose at most 2%, everything else fully automatic
res = amct.accuracy_based_auto_prune(model, data=calib, tolerance=0.02)
print(res.summary())
# [auto-prune] applied prune_ratio=0.20: weights 607,808 → 568,640 (cut 6.4%), quality loss 0.018 ≤ tolerance 0.020.
```

### Search + post-prune finetune recovery (important)

One-shot structured pruning collapses accuracy to **near random**; if the search evaluates "immediately after pruning", it can select almost no ratio (overly conservative).
When `finetune_fn` is provided, the search **applies (lightweight) finetune recovery to each candidate ratio before evaluating**, so the selected ratio is the more aggressive one that is actually usable "after prune + recovery":

```python
from amct_pytorch.pruning import prune_finetune
amct.prune(model, cfg, data=calib, tolerance=0.05, evaluator=test_acc,
           finetune_fn=lambda m: prune_finetune(m, train_data, steps=300))
```

> Measured (MNIST, **held-out val acc** gate, tolerance 5%): **without `finetune_fn` no ratio is selected (0% pruning);
> with `finetune_fn` 50% channel pruning is selected** (post-recovery val acc still within tolerance; baseline 0.857). Cost: each candidate requires one finetune pass.

- The default **fidelity metric needs no labels**: the original model's top-1 predictions on the calibration data serve as reference, quality = prediction agreement rate after pruning;
  an `evaluator` can also be passed (same scale as `tolerance`, e.g. a custom accuracy).
- `safe_skip_attention=True` (default): automatically skips `self_attn/attention/attn`, preventing the dense domain from misidentifying parallel
  q/k/v projections as a two-layer FFN and breaking the LLM.
- Binary search over `ratio_grid`; any prune/forward failure while probing a ratio is treated as unacceptable, no exception is raised.
  Under non-monotonicity the search is **conservative** (prunes less), which is the safe direction for the user.

### Parameters

| Parameter | Required | Default | Description |
|------|------|--------|------|
| `model` | yes | - | `torch.nn.Module` to prune (in-place) |
| `data` | depends on method | `None` | Calibration data; required by variance-based methods, also the default evaluation set |
| `tolerance` | yes | `0.02` | Upper bound on acceptable accuracy loss (same scale as `evaluator`) |
| `evaluator` | no | top-1 fidelity | accuracy source: a `callable(model)->float` or an `AutoCalibrationEvaluatorBase` (shared with quantization) |
| `eval_data` | no | falls back to `data` | Evaluation batch for the default fidelity metric |
| `ratio_grid` | no | `0.1..0.8` | Candidate prune ratios (ascending) |
| `safe_skip_attention` | no | `True` | Automatically protect attention projections |

> Explicit API: `accuracy_based_auto_prune(model, config=None, *, data, tolerance, ...) -> AutoTuneResult`.
> `amct.prune(..., tolerance=...)` is its equivalent wrapper on the main entry point.

## Fixed prune ratio

```python
from amct_pytorch.pruning import PruneReport
cfg = {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}}
rep = PruneReport()
amct.prune(model, cfg, data=calib, report=rep)   # dense domain automatically avoids attention, no manual skip needed
print(rep.as_dict())
```

**Partial configs prune the rest at defaults.** `methods` only *overrides* per-domain defaults; any domain
you omit is still pruned at its default ratio (cnn 0.30 / dense 0.50 / moe 0.50). To prune **only** a chosen
domain, pin the others to `prune_ratio: 0.0`. This matters for multi-domain models — e.g. an MoE whose
experts contain their own dense FFN: a moe-only config would otherwise also prune that FFN.

## CNN channel pruning (output-preserving reconstruction)

A simple "same-parent sibling conv" heuristic (`variance_channel`) slices channels naively by activation variance, suitable for flat/sequential CNNs.
Recommended: `reconstruct` — within the same cnn domain (identifying the producer conv → bn → consumer sequential coupling), after removing low-saliency
channels it **reconstructs the consumer conv weights via im2col least squares**, absorbing the removed channels' contribution into the kept channels so the output is preserved as far as possible
(ThiNet/CP-style compensation, no finetuning required; in-house implementation, no third-party dependencies):

```python
from amct_pytorch.pruning import CNN_RECONSTRUCT_PRUNE_CFG
amct.prune(model, CNN_RECONSTRUCT_PRUNE_CFG, data=calib)   # calib provides sample inputs for activation statistics
# or customize: {"methods": {"cnn": {"name": "reconstruct",
#                                "kwargs": {"prune_ratio": 0.3, "ridge": 1e-2}}}}
```

- The output head (last Linear) is ignored automatically, keeping the class count unchanged; layers matched by `skip_layers` are also ignored.
- `reconstruct` is markedly less destructive than naive `variance_channel` slicing on trained models (same parameter count).
- Calibration must be representative and diverse (row count ≥ keep×k×k to solve the least squares); when underdetermined or the consumer is not Conv2d, an **explicit warning** is emitted
  and the naive slicing fallback is used (no silent pretense of "reconstructed").
- In a **residual (ResNet)** block the *interior* conv→conv coupling IS pruned; only the conv feeding the residual `add` is auto-excluded (so the residual width is preserved, no damage) — not a wholesale skip. Concat (Inception) consumers and grouped/depthwise convs remain **0 targets** (deferred, see "Supported prunable structures").

## Guarded criterion selection (MoE expert pruning)

> **Advanced (not in the minimal public `__all__`)** — import via `from amct_pytorch.pruning import ...`.

Variance-style routing scores come from a single calibration pass (no gradients / least-squares /
expert forward) and beat activation-frequency (mass) pruning on *some* MoEs — but there is no
universal "better variance" criterion (the best one depends on routing structure).
The **MoE variance criterion menu** does not bet on one criterion: pass the `MOE_VARIANCE_MENU_CFG`
preset to `accuracy_based_auto_prune` to run **one calibration → prune a copy per criterion in the
MENU → pick the best on a small validation set → apply the winner**:

```python
from amct_pytorch.pruning import MOE_VARIANCE_MENU_CFG

res = amct.accuracy_based_auto_prune(
    model, MOE_VARIANCE_MENU_CFG, data=calib, eval_data=val, evaluator=ev)  # keep val != calib
print(res.summary())     # standard AutoTuneResult; chosen variant recorded in the returned AutoTuneResult.report (or a report= sink)
```

- **MENU** (default): `mass` (= `activation_count`, the fallback) / `cond_var` (Var(w|sel)) /
  `peak` (max w — keeps consistently-strong specialists that cond misses) / `cvxpeak` (Var(w|sel)×max w).
- **Guard guarantee**: switch away from `mass` only when a variance criterion is *strictly better* on
  the **validation set**, otherwise fall back to `mass` — so the HYBRID is never worse than fallback
  on validation. (Mirrors `accuracy_based_auto_prune`'s ratio search, but searches over the *criterion*;
  both reuse the same deepcopy-trial + evaluator guard + in-place apply machinery.)
- The gain is largest under **class/expert-usage imbalance + a tail-sensitive metric** (e.g. many-expert
  group-routed MoEs).
- **Honest scope**: on **real pretrained MoE** the criterion is **depth-dependent** — a *global*
  variance criterion (every layer) loses to mass, but variance on the **deep layers only**
  (`boundary=K`) beats mass on multi-expert routers — see **Depth-boundary expert selection on real
  MoE** below. On LLaDA2.1-mini **GSM8K** aggressive *static* expert pruning collapses (dynamic
  masking needed). Net: the real-MoE gain needs the **depth boundary**, not a global criterion;
  "never worse" remains a VAL guarantee (granite's few-expert router correctly collapses to mass).
- The underlying per-criterion score is the new `mass_variance` kwarg `variance_score` ∈
  {`cond` (default), `peak`, `cvxpeak`}, with a depth split `boundary`: **mass on layers ≤ boundary,
  variance on layers > boundary** (default `10`, validated on LLaDA2.1-mini; `-1` = every layer variance;
  `"auto"` = per-layer pick; `list[int]` = variance on exactly those layers). The default `cond` score is
  byte-for-byte the previous behavior (backward compatible). `MOE_VARIANCE_MENU_CFG`'s
  default menu uses the global `boundary=-1`; pass a custom menu with `boundary=K` to search depth splits.

### Depth-boundary expert selection on real MoE

The expert-keep criterion is **depth-dependent**: in the **deep** (last) MoE layers per-expert routing-weight
**variance** selects better survivors than activation **mass**; **shallow** layers favor mass; a global
all-variance criterion underperforms the depth split. `mass_variance` splits by a depth `boundary` K
(mass on layers ≤ K, variance on layers > K). When the per-expert variance signal is unreliable
(few-expert routers), the guarded menu search collapses to all-mass, so the result is never worse than mass.
On real pretrained MoE the achievable cut at a tight quality budget is finetune-gated.

## Guarded recovery selection (dense FFN / CNN channel pruning)

> **Advanced (not in the minimal public `__all__`)** — import via `from amct_pytorch.pruning import ...`.

For dense FFN / CNN channel pruning the accuracy lever is the post-prune **RECOVERY**, not the neuron-
selection criterion. The **dense/CNN recovery menu** mirrors the variance menu but searches the
recovery MENU: pass the `DENSE_RECOVERY_MENU_CFG` preset to `accuracy_based_auto_prune`:

```python
from amct_pytorch.pruning import DENSE_RECOVERY_MENU_CFG

res = amct.accuracy_based_auto_prune(
    model, DENSE_RECOVERY_MENU_CFG, data=calib, eval_data=val, evaluator=ev)  # domain in the preset (dense/cnn)
print(res.summary())     # standard AutoTuneResult; chosen variant recorded in the returned AutoTuneResult.report (or a report= sink)
```

- **MENU** (default): `none` (naive slice, fallback) / `bias` (mean-fold into the consumer bias, no
  least-squares) / `ls` (least-squares reconstruction of the consumer weight). Backed by the new
  `reconstruct` kwarg `recovery` ∈ {`ls` (default), `bias`, `none`} (added to both dense and cnn;
  the default `ls` is the previous behavior, backward compatible).
- **Guard guarantee (be honest about its scope)**: the guard switches off the fallback only on a STRICT
  win **on the validation set** → never worse than the no-recovery baseline on validation. This is a
  VAL guarantee, NOT a test guarantee: if `eval_data` is not representative of the deployment distribution
  the val-pick may not transfer (see the small-calib note). Use `eval_data` representative of deployment.
- **Calibration size is the key knob**: ample calib → `ls` wins; calib < intermediate width → `ls` is
  underdetermined and overfits, `bias` (LS-free) is more robust. The guard rides the bias↔ls transition
  automatically with calibration size.
- **Small-calib safety knob (dense)**: `bias`-fold overfits the calibration region's activation
  mean and can hurt on a distribution-shifted test set (structural val↔test mismatch, not sampling
  noise). For a test-side never-worse guarantee, drop `bias` from the menu — keep only the
  `none`/`ls` entries in `cfg["methods"]["dense"]["menu"]` — trading some small-calib mean gain.
- **CNN has no such floor**: in CNN channel pruning the im2col rows = N×spatial are abundant
  even at few images, so `ls` is not underdetermined and `bias` is rarely the val-winner; the full
  guard is already test-side safe on CNN.
- **Measured scope**: all dense/CNN numbers above are **no-FT** (deploy-as-is, no retraining) relative
  comparisons; recovery is the lever in that no-fine-tune regime. With post-prune fine-tuning the naive
  slice may catch up — a separate question.

## Advanced configuration

```python
# cross-layer sensitivity allocation (default uniform; guard automatically falls back to uniform and warns on degradation)
amct.prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib)
# config["allocation"] = {"strategy": "sensitivity", "ref_ratio": 0.5,
#                          "min_ratio": 0.05, "max_ratio": 0.9, "guard": "calib_nll"}

# MoE output-space expert merge (LS residual pairing + drop fallback; auto-picks the better of calib NLL and activation_count via A/B)
amct.prune(model, MOE_OUTPUT_MERGE_PRUNE_CFG, data=calib)
# moe kwargs: keep_ratio=0.5, selector='calib_nll'|'none'

# quantization-aware saliency: directly reuses the same AMCT calibrate-style scale/offset
cfg["methods"]["dense"]["kwargs"]["quant_cfg"] = {
    "weights_cfg": {"strategy": "channel", "symmetric": True, "dtype": "int8"}}

# budget scope: budget_scope='total' (default) |'prunable'; warns when unreachable + report.budget_unreachable
amct.prune(model, data=calib, size_budget=0.7)   # prune to a size budget (keep 70%)
# full control (target_params / budget_scope): use size_budget_prune (advanced, import from amct_pytorch.pruning)
```

# Combined use with quantization (prune → quantize → convert)

Pruning (structured, removing channels/neurons/experts) and quantization (lowering bits) are **orthogonal** compression axes: size gains
**multiply**, accuracy losses are **approximately additive**. Recommended order: **prune first, then quantize, finally convert the deployment model** —
pruning changes tensor shapes, so quantization calibration is only correct on the pruned weights.

```python
import amct_pytorch as amct
from amct_pytorch.pruning import accuracy_based_auto_prune

accuracy_based_auto_prune(model, data=calib, tolerance=0.02)  # 1) structured pruning
amct.quantize(model, quant_cfg)                                # 2) quantize on the pruned model
amct.convert(model)                                            # 3) convert the deployment model
```

**Joint accuracy budget**: split one total budget across the two steps and measure sequentially yourself —
pruning consumes a measured portion first, then the quantization search runs on the *pruned model* with the remaining budget:

```python
# Split one accuracy budget across prune + quantize yourself (no extra API needed):
total_tolerance = 0.03                                                       # total budget 3%
res = amct.accuracy_based_auto_prune(model, data=calib,
                               tolerance=total_tolerance * 0.5, evaluator=ev)  # give pruning half the upper bound first
prune_drop = res.quality_drop or 0.0                                          # pruning's measured accuracy loss
remaining = max(0.0, total_tolerance - prune_drop)                            # leftover budget for quantization
# use the remaining budget as the acceptance threshold for the quantization accuracy search (shares the same evaluator)
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ...)               # quant budget = remaining
```

`remaining = max(0, total_tolerance - measured prune loss)`, connecting naturally to the quantization-side
[`accuracy_based_auto_calibration`](../../docs/zh/api/accuracy_based_auto_calibration.md).

**Shared accuracy abstraction**: the pruning `evaluator=` and the quantization `accuracy_based_auto_calibration` share
**the same accuracy evaluator** — a subclass of `AutoCalibrationEvaluatorBase` (whose `evaluate(model)`
**returns an accuracy metric**, higher is better). One object drives the accuracy search on both sides; no need to define two metrics:

```python
class MyEvaluator(AutoCalibrationEvaluatorBase):
    def evaluate(self, model):              # return metrics such as top-1 / mAP
        return my_top1_accuracy(model)

ev = MyEvaluator()
accuracy_based_auto_prune(model, data=calib, tolerance=0.02, evaluator=ev)   # pruning
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ...)              # quantization
```

The pruning side accepts both `evaluate(model)` and `evaluate(model, iterations)` signatures.

> Do not confuse: the graph-side built-in **`amct.ModelEvaluator`** is a **data feeder** for quantization calibration;
> its `evaluate(model, iterations)` **returns None** and cannot be used for accuracy search; if passed by mistake, the pruning side raises a clear error.
> For accuracy search use a metric-returning `AutoCalibrationEvaluatorBase` subclass.
>
> accuracy: `evaluator` (callable or `AutoCalibrationEvaluatorBase`) or default top-1 fidelity.

# Appendix: relation to graph subsystem / support boundary

## Relation to the existing graph pruning subsystem (zero breakage)

This module is **complementary to, not a replacement for**, the existing `classic/graph_based` subsystem
(CNN conv channels + retrain + Ascend deployment): it targets LLM dense FFN / MoE experts / modern CNN on
eager `torch.nn.Module`, purely additive (0 changes to existing files). It already aligns vocabulary for
future absorption: the shared accuracy abstraction `AutoCalibrationEvaluatorBase` and a sequential budget hand-off
(prune consumes a measured share, then quantization `accuracy_based_auto_calibration` runs on the remainder). For now it coexists
risk-free as a self-contained, purely additive module (`amct_pytorch.pruning`).

### Supported prunable structures
Structured pruning targets only dimensions whose producer↔consumer interface can be verified, and conservatively skips the rest (never mis-pruned):
- **Dense FFN** — prunes the *intermediate* dimension only (`gate/up.out_features` + `down.in_features`); the hidden/residual width (`down.out_features`) is preserved. Attention projections (q/k/v/o) are auto-excluded.
- **CNN channels** — a producer `Conv2d` (`groups=1`) → optional `BatchNorm2d` → consumer `Conv2d`/`Linear` with `consumer.in_channels == producer.out_channels`; producer-out, BN, and consumer-in are co-resized together. In a **residual (ResNet)** block the *interior* conv→conv coupling is pruned (the conv feeding the residual `add` is auto-excluded → residual width preserved). **Concat (Inception)** consumers (`in = Σ branches`) and **grouped/depthwise** convs are **not pruned** (0 targets) — verifying their safety needs cross-branch dependency tracking, deferred.
- **MoE experts** — whole *routed* experts are removed; the layer input/output (hidden) is unchanged. **Shared (always-active) experts are excluded** from routed-expert pruning.

## Current support boundary (measured with current transformers)

| Domain | Supported | Notes |
|----|------|------|
| dense | √ | LLM FFN: the three-Linear `gate/up/down_proj`, **fused `gate_up_proj`** (Phi-3/GLM-4 style, first half gate, second half up), adjacent two Linear/Conv1D layers (including Bloom-style custom parameter-free activation between them). The dense domain structurally excludes parallel attention projections (q/k/v/o share input and do not satisfy the in→inter→in residual-dim restoration), so Llama/Qwen2/Mistral/Qwen3 prune safely **without manual `skip_layers`**. |
| cnn | √ | `variance_channel` (heuristic) naive slicing; `reconstruct` (im2col least squares) output-preserving, lower damage. ResNet-style independent-block structures are not recognized by same-level detection (known boundary). |
| moe | √ | Legacy `nn.ModuleList` experts + `nn.Linear` gate, **and modern fused (batched) experts** (`MixtralExperts`/`Qwen3MoeExperts` + `*TopKRouter`, batched tensors sliced along the expert dim). Also handles: **grouped routers** (`n_routed_experts`/`n_group`/`topk_group` — syncs the expert count and collapses groups to 1; DeepseekV3-style), **sibling two-tensor fused experts** (experts split across parallel sub-modules, e.g. GraniteMoE `input_linear`/`output_linear`), and **nested router-bias** slicing (e.g. Ernie4.5 `moe_statics.e_score_correction_bias`). Proven by `test_pruning_real_hf_models.py` (Mixtral, Qwen3Moe, GraniteMoe, DeepseekV3, Ernie4_5_Moe). |

**Architecture coverage matrix** (configs shrunk to 2-layer micro models, random weights — verifies mechanical prunability only): covers 29 architecture families —
llama/llama3/qwen2/qwen3/qwen2_moe/qwen3_moe/mixtral/mistral/gemma/gemma2/gemma3/phi/phi3/glm4/
gpt2/gpt_neox/gpt_neo/gpt_bigcode/starcoder2/opt/xglm/olmo/olmo2/stablelm/smollm3/bloom/
granitemoe/deepseekv3/ernie4_5.

Regression baseline: `tests/amct_pytorch/test_pruning_real_hf_models.py`; tolerance feature:
`tests/amct_pytorch/test_pruning_auto_prune.py` (requires transformers).

# Structured Pruning (experimental.pruning)

Structured pruning of an already-instantiated `torch.nn.Module` (dense FFN intermediate dimension / CNN channels / MoE experts).
The façade is aligned with `amct.quantize`: the model is modified in-place, statistics are retrieved via `get_last_report()`.
The library never downloads models; the caller is responsible for instantiation (e.g. `from_pretrained(local_path)`).

## Tolerance-driven auto pruning (recommended, for non-experts)

Usable without knowledge of prune ratios or layer structure: **specify only "how much accuracy loss is acceptable"**, and the largest prune ratio that satisfies the tolerance is searched for and applied automatically. Analogous to the quantization-side [`accuracy_based_auto_calibration`](../../../docs/zh/api/accuracy_based_auto_calibration.md)
(accuracy-first + BinarySearch); this is its pruning counterpart.

```python
import amct_pytorch.experimental.pruning as amct

# 最多损失 2%，其余全自动
amct.prune(model, data=calib, tolerance=0.02)
print(amct.get_last_auto_prune().summary())
# [auto-prune] 已应用剪枝率=0.20：权重 607,808 → 568,640 (削减 6.4%)，质量损失 0.018 ≤ 容差 0.020。
```

### Search + post-prune finetune recovery (important)

One-shot structured pruning collapses accuracy to **near random**; if the search evaluates "immediately after pruning", it can select almost no ratio (overly conservative).
When `finetune_fn` is provided, the search **applies (lightweight) finetune recovery to each candidate ratio before evaluating**, so the selected ratio is the more aggressive one that is actually usable "after prune + recovery":

```python
from amct_pytorch.experimental.pruning import finetune
amct.prune(model, cfg, data=calib, tolerance=0.05, quality_fn=test_acc,
           finetune_fn=lambda m: finetune(m, train_data, steps=300))
```

> Measured (MNIST, real test acc, tolerance 5%): **without `finetune_fn` no ratio is selected (0% pruning);
> with `finetune_fn` 40% channel pruning is selected** (post-recovery acc still within tolerance). Cost: each candidate requires one finetune pass.

- The default **fidelity metric needs no labels**: the original model's top-1 predictions on the calibration data serve as reference, quality = prediction agreement rate after pruning;
  a `quality_fn` can also be passed (same scale as `tolerance`, e.g. a custom accuracy).
- `safe_skip_attention=True` (default): automatically skips `self_attn/attention/attn`, preventing the dense domain from misidentifying parallel
  q/k/v projections as a two-layer FFN and breaking the LLM.
- Binary search over `ratio_grid`; any prune/forward failure while probing a ratio is treated as unacceptable, no exception is raised.
  Under non-monotonicity the search is **conservative** (prunes less), which is the safe direction for the user.

### Parameters

| Parameter | Required | Default | Description |
|------|------|--------|------|
| `model` | yes | - | `torch.nn.Module` to prune (in-place) |
| `data` | depends on method | `None` | Calibration data; required by variance-based methods, also the default evaluation set |
| `tolerance` | yes | `0.02` | Upper bound on acceptable accuracy loss (same scale as `quality_fn`) |
| `quality_fn` | no | top-1 fidelity | Accuracy evaluation function, higher is better |
| `eval_data` | no | falls back to `data` | Evaluation batch for the default fidelity metric |
| `ratio_grid` | no | `0.1..0.8` | Candidate prune ratios (ascending) |
| `safe_skip_attention` | no | `True` | Automatically protect attention projections |

> Explicit API: `accuracy_based_auto_prune(model, config=None, *, data, tolerance, ...) -> AutoTuneResult`.
> `amct.prune(..., tolerance=...)` is its equivalent wrapper on the main entry point.

## Fixed prune ratio

```python
cfg = {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}}
amct.prune(model, cfg, data=calib)         # dense 域自动避开注意力，无需手动 skip
print(amct.get_last_report().as_dict())
```

## CNN channel pruning (output-preserving reconstruction)

A simple "same-parent sibling conv" heuristic (`variance_channel`) slices channels naively by activation variance, suitable for flat/sequential CNNs.
Recommended: `reconstruct` — within the same cnn domain (identifying the producer conv → bn → consumer sequential coupling), after removing low-saliency
channels it **reconstructs the consumer conv weights via im2col least squares**, absorbing the removed channels' contribution into the kept channels so the output is preserved as far as possible
(ThiNet/CP-style compensation, no finetuning required; **in-house implementation, no external/bundled third-party dependencies**):

```python
from amct_pytorch.experimental.pruning import CNN_RECONSTRUCT_PRUNE_CFG
amct.prune(model, CNN_RECONSTRUCT_PRUNE_CFG, data=calib)   # calib 提供样例输入用于激活统计
# 或自定义：{"methods": {"cnn": {"name": "reconstruct",
#                                "kwargs": {"prune_ratio": 0.3, "ridge": 1e-2}}}}
```

- The output head (last Linear) is ignored automatically, keeping the class count unchanged; layers matched by `skip_layers` are also ignored.
- Measured (trained MNIST CNN, prune 0.3): naive `variance_channel` acc 0.343 vs `reconstruct` acc 0.574
  (same parameter count, +23pp) — reconstruction is markedly less destructive on trained models.
- Calibration must be representative and diverse (row count ≥ keep×k×k to solve the least squares); when underdetermined or the consumer is not Conv2d, an **explicit warning** is emitted
  and the naive slicing fallback is used (no silent pretense of "reconstructed").
- Architectures like ResNet that wrap convolutions in independent blocks are not recognized by the same-level detection (0 targets, naturally skipped, no damage) — a known boundary of the same-level detection
  (an earlier implementation based on the Torch-Pruning dependency graph was removed due to severe acc damage, replaced by same-domain output-preserving reconstruction).

## Performance measurements (real trained models + measured int8 on NPU)

16 small models **trained from scratch** (CNN@MNIST / Dense-LM·MoE-LM@WikiText) + 20 real pretrained HF models (10 architecture families),
4 compression budgets × real AMCT int8 (Ascend NPU). Budget-driving metric = **top-1 fidelity** (label-free agreement rate with the original model's predictions);
task metrics are reported (CNN = test acc↑, LM/MoE/HF = perplexity↓).

### Conclusion 1: int8 quantization is nearly free, and "quant-aware pruning" yields no gain under int8

int8 after pruning (`amct.quantize→convert`) loses almost nothing; pruning with `quant_aware=True` (scoring with quantized weights) ≈ plain pruning.

| Model | Budget | After prune | Plain prune + int8 | quant_aware + int8 |
|------|------|:------:|:------------:|:----------------:|
| gpt2 (dense Conv1D, ppl) | param −13.7% (r0.3) | ppl 91.07 | 91.66 | 91.57 |
| moe_E4_L4 (moe experts, ppl) | param −9.9% (r0.1) | ppl 993.1 | 993.1 | (= plain) |
| cnn_w16_d4 (cnn channels) | param−25% (−27% params) | acc 0.954 | 0.954 (int8) | (= plain) |

> Note: the after-prune ppl for dense/moe above uses the original small calibration set (Conclusion 1 only concerns the int8 **delta**, which is calib-independent); dense LM with a larger calib (tokens > FFN intermediate) prunes better — see Conclusion 3.

→ **Quant-aware pruning can be skipped under int8** (differences are in the last decimal places).

### Conclusion 2: sensitivity allocation depends strongly on depth — guard is mandatory (at equal FFN reduction)

Water-fill allocation from per-layer measured sensitivity (fidelity drop when pruning only that layer): **shallow models (<10 layers) win 6/6;
deep models (≥10 layers) only 4/19** — pruning robust layers harder in deep nets damages reachability. Overall 10/25.
(Note: this sensitivity sweep is the **19-HF subset**, before opt-350m was added; the prunability stats in Conclusion 3 are updated to 20 HF.)
Allocation is therefore not the default, and `guard='calib_nll'` forces an A/B comparison: when worse than uniform it automatically falls back with a warning.

| Model (layers) | uniform ppl | sensitivity ppl | winner |
|------|:---:|:---:|:---:|
| lm_L6_d160 (6) | 919.8 | **914.0** | sens |
| SmolLM-135M (30) | 120.2 | **117.4** | sens |
| Cerebras-GPT-111M (10) | 134.1 | **128.9** | sens |
| gpt2 (12) | **116.6** | 121.9 | uniform |
| pythia-410m (24) | **132.5** | 157.5 | uniform |

> The signal must be **measured fidelity drop** (raw saliency is not comparable across layers and performs worse). An earlier bench, lacking tail points after the skip prefix
> (layers.1 mistakenly isolated from layers.10+), overestimated gains at 23/25; corrected results are in the table above.

### Conclusion 3: How much dense LM can be pruned at a 1% quality budget — ~0 without finetune, median 27% (max 61%) with 1-epoch finetune

**1% budget max-prune-ratio search**: for each candidate ratio, run `prune → [finetune] → held-ppl` and binary-search the **largest** ratio whose reported-ppl loss ≤ 1% (= max total-param reduction). Gated on ppl itself (not a fidelity proxy). 20 real pretrained HF dense-LMs (10 architecture families), WikiText-2 held set; finetune on a *distinct* slice of WikiText-2 train, 1 epoch (lr 1e-5 + warmup), disjoint from the eval set:

| | **no-FT** max cut @1% budget | **+1-epoch FT** max cut @1% budget |
|------|:---:|:---:|
| all 20 HF (range / median) | **0–14%** (most 0%) | **14–61% / median 27%** |
| santacoder (GPTBigCode) | 14.3% | **60.9%** |
| phi-1_5 (Phi) | 0% | **48.3%** |
| opt-350m (OPT) | 3.0% | **36.5%** |
| SmolLM-135M (Llama) | 0% | **23.7%** |
| Qwen2.5-1.5B (Qwen2) | 0% | **15.0%** |

**Honest conclusion**: on well-trained pretrained LMs, *training-free* dense-FFN pruning has almost no headroom at 1% (most models max-cut 0% — structured pruning's immediate damage exceeds 1%; fidelity is insensitive to LM damage but ppl is sensitive). A **1-epoch finetune** (lr 1e-5 + warmup, single pass over distinct data) raises the max cut to **14–61% (median 27%)** — i.e. "spend 1% quality to remove 14–61% of params" (not 1% for 1%), and after finetune most models even beat baseline ppl (WikiText domain adaptation). This is not method failure but the structured-pruning norm: CNN reconstruct (output reconstruction absorbs the damage) and MoE expert pruning/merge are finetune-free; dense LM needs finetune. Finetune hyperparameters must be LM-scale: lr 1e-5 + warmup (1e-4 diverges), single pass over distinct data (many epochs over few batches memorizes the train set and held-ppl explodes).

> **Why opt-350m was absent from the earlier count of 19**: its `project_in/project_out` (word-embedding↔hidden projections, whose registration order is the reverse of execution order) were mis-matched as a two-layer FFN, corrupting the model and crashing the forward pass, so it was silently excluded. That mis-match is now fixed by a name-based exclusion in the dense domain (`_looks_like_embed_proj` in `domains/dense.py`); opt-350m is back in → **20 HF**, cutting 36.5% at the 1% budget.
> Cross-domain comparison (same 1% budget, +finetune max cut): CNN@MNIST (test acc) **31–74%**; toy MoE@WikiText **8–29%**; toy Dense-LM only **2–7%** (embedding-dominated, small FFN fraction — a structural not quality limit).
> Reproduce: `python examples/pruning/e2e_dense_finetune_recovery.py` (default pythia-160m; set `AMCT_PRUNE_LM` for another model).

### Boundaries
- **toy Dense-LM** (from-scratch, embedding dominates the param count) cannot exceed ~7% total-param cut even with finetune — a structural limit (FFN fraction too small), not a quality limit; real pretrained HF (normal FFN fraction) reach **14–61%** with finetune (see Conclusion 3).
- Under a tight quality budget (**−1%**), training-free pruning leaves almost no feasible ratio for pretrained LMs (a small cut already pushes ppl past threshold) — finetune is required.

## Advanced configuration (new in v0.4)

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

## Combined use with quantization (prune → quantize → convert)

Pruning (structured, removing channels/neurons/experts) and quantization (lowering bits) are **orthogonal** compression axes: size gains
**multiply**, accuracy losses are **approximately additive**. Recommended order: **prune first, then quantize, finally convert the deployment model** —
pruning changes tensor shapes, so quantization calibration is only correct on the pruned weights (`PrunePass` is designed to
precede `InsertQuantizeModulePass` in the `ModelOptimizer`).

```python
import amct_pytorch as amct
from amct_pytorch.experimental.pruning import accuracy_based_auto_prune

accuracy_based_auto_prune(model, data=calib, tolerance=0.02)  # 1) 结构化剪枝
amct.quantize(model, quant_cfg)                                # 2) 在剪枝后的模型上量化
amct.convert(model)                                            # 3) 转换部署模型
```

**Joint accuracy budget**: `accuracy_based_auto_prune_with_budget` **splits the total budget across the two steps and measures sequentially** —
pruning consumes a measured portion first, then the quantization search runs on the *pruned model* with the remaining budget:

```python
from amct_pytorch.experimental.pruning import accuracy_based_auto_prune_with_budget

jb = accuracy_based_auto_prune_with_budget(
    model, data=calib, total_tolerance=0.03, prune_fraction=0.5)  # 总预算 3%，先给剪枝一半上界
print(jb.summary())                       # 剪枝实测用掉 jb.prune_drop
# 把剩余预算作为量化精度搜索的接受阈值（与量化共用同一 evaluator）
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ..., )   # 预算 = jb.remaining_tolerance
```

`jb.remaining_tolerance = max(0, total_tolerance - measured prune loss)`, connecting naturally to the quantization-side
[`accuracy_based_auto_calibration`](../../../docs/zh/api/accuracy_based_auto_calibration.md).

**Shared accuracy abstraction**: the pruning `evaluator=` and the quantization `accuracy_based_auto_calibration` share
**the same accuracy evaluator** — a subclass of `AutoCalibrationEvaluatorBase` (whose `evaluate(model)`
**returns an accuracy metric**, higher is better). One object drives the accuracy search on both sides; no need to define two metrics:

```python
class MyEvaluator(AutoCalibrationEvaluatorBase):
    def evaluate(self, model):              # 返回 top-1 / mAP 等指标
        return my_top1_accuracy(model)

ev = MyEvaluator()
accuracy_based_auto_prune(model, data=calib, tolerance=0.02, evaluator=ev)   # 剪枝
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ...)              # 量化
```

The pruning side accepts both `evaluate(model)` and `evaluate(model, iterations)` signatures.

> ⚠️ Do not confuse: the graph-side built-in **`amct.ModelEvaluator`** is a **data feeder** for quantization calibration;
> its `evaluate(model, iterations)` **returns None** and cannot be used for accuracy search; if passed by mistake, the pruning side raises a clear error.
> For accuracy search use a metric-returning `AutoCalibrationEvaluatorBase` subclass.
>
> Priority: `quality_fn` > `evaluator` > default top-1 fidelity.

## Relation to the existing graph pruning subsystem (zero breakage)

This module is **complementary to, not a replacement for**, the existing `classic/graph_based` subsystem
(CNN conv channels + retrain + Ascend deployment): it targets LLM dense FFN / MoE experts / modern CNN on
eager `torch.nn.Module`, purely additive (0 changes to existing files). It already aligns vocabulary for
future absorption: the shared accuracy abstraction `AutoCalibrationEvaluatorBase`, sequential budget hand-off
(`accuracy_based_auto_prune_with_budget` → quantization `accuracy_based_auto_calibration`), and a
`PrunePass.run(model)` consistent with the `ModelOptimizer` protocol. If absorbed, it can become a
`backend="eager"` option of the existing entry point (a public-API change, to be agreed via an Issue per
AGENTS.md first); until then it coexists risk-free in experimental form.

## Current support boundary (measured with current transformers)

| Domain | Supported | Notes |
|----|------|------|
| dense | ✅ | LLM FFN: the three-Linear `gate/up/down_proj`, **fused `gate_up_proj`** (Phi-3/GLM-4 style, first half gate, second half up), adjacent two Linear/Conv1D layers (including Bloom-style custom parameter-free activation between them). The dense domain structurally excludes parallel attention projections (q/k/v/o share input and do not satisfy the in→inter→in residual-dim restoration), so Llama/Qwen2/Mistral/Qwen3 prune safely **without manual `skip_layers`**. |
| cnn | ✅ | `variance_channel` (heuristic) naive slicing; **`reconstruct` (im2col least squares) output-preserving, low damage** (measured on trained MNIST CNN @prune0.3: variance 0.343 → reconstruct 0.574). ResNet-style independent-block structures are not recognized by same-level detection (known boundary). |
| moe | ✅ | Supports both the legacy `nn.ModuleList` experts + `nn.Linear` gate structure **and modern transformers fused (batched) experts** (`MixtralExperts`/`Qwen3MoeExperts` + `*TopKRouter`, slicing batched tensors along the expert dim). |

**Architecture coverage matrix** (configs shrunk to 2-layer micro models, random weights — verifies mechanical prunability only): 26/29 families OK —
llama/llama3/qwen2/qwen3/qwen2_moe/mixtral/mistral/gemma/gemma2/gemma3/phi/phi3/glm4/falcon-naming(bloom)/
gpt2/gpt_neox/gpt_neo/gpt_bigcode/starcoder2/opt/xglm/olmo/olmo2/stablelm/smollm3/bloom;
unsupported: mamba (SSM, no FFN), granite MoE variants, falcon (fused QKV shape).

Regression baseline: `tests/amct_pytorch/test_pruning_real_hf_models.py`; tolerance feature:
`tests/amct_pytorch/test_pruning_auto_prune.py` (requires transformers, run with `.venv`).

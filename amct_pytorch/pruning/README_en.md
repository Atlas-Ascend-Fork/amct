# Structured Pruning (amct_pytorch.pruning)

Structured pruning of an already-instantiated `torch.nn.Module` (dense FFN intermediate dimension / CNN channels / MoE experts).
The façade is aligned with `amct.quantize`: the model is modified in-place, statistics are retrieved via `get_last_report()`.
The library never downloads models; the caller is responsible for instantiation (e.g. `from_pretrained(local_path)`).

## Tolerance-driven auto pruning (recommended, for non-experts)

Usable without knowledge of prune ratios or layer structure: **specify only "how much accuracy loss is acceptable"**, and the largest prune ratio that satisfies the tolerance is searched for and applied automatically. Analogous to the quantization-side [`accuracy_based_auto_calibration`](../../docs/zh/api/accuracy_based_auto_calibration.md)
(accuracy-first + BinarySearch); this is its pruning counterpart.

```python
import amct_pytorch as amct

# 最多损失 2%，其余全自动
amct.prune(model, data=calib, tolerance=0.02)
print(amct.get_last_auto_prune().summary())
# [auto-prune] 已应用剪枝率=0.20：权重 607,808 → 568,640 (削减 6.4%)，质量损失 0.018 ≤ 容差 0.020。
```

### Search + post-prune finetune recovery (important)

One-shot structured pruning collapses accuracy to **near random**; if the search evaluates "immediately after pruning", it can select almost no ratio (overly conservative).
When `finetune_fn` is provided, the search **applies (lightweight) finetune recovery to each candidate ratio before evaluating**, so the selected ratio is the more aggressive one that is actually usable "after prune + recovery":

```python
from amct_pytorch.pruning import prune_finetune
amct.prune(model, cfg, data=calib, tolerance=0.05, quality_fn=test_acc,
           finetune_fn=lambda m: prune_finetune(m, train_data, steps=300))
```

> Measured (MNIST, **held-out val acc** gate, tolerance 5%): **without `finetune_fn` no ratio is selected (0% pruning);
> with `finetune_fn` 50% channel pruning is selected** (post-recovery val acc still within tolerance; baseline 0.857). Cost: each candidate requires one finetune pass.

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

> **Partial configs prune the rest at defaults.** `methods` only *overrides* per-domain defaults; any domain
> you omit is still pruned at its default ratio (cnn 0.30 / dense 0.50 / moe 0.50). To prune **only** a chosen
> domain, pin the others to `prune_ratio: 0.0`. This matters for multi-domain models — e.g. an MoE whose
> experts contain their own dense FFN: a moe-only config would otherwise also prune that FFN.

## CNN channel pruning (output-preserving reconstruction)

A simple "same-parent sibling conv" heuristic (`variance_channel`) slices channels naively by activation variance, suitable for flat/sequential CNNs.
Recommended: `reconstruct` — within the same cnn domain (identifying the producer conv → bn → consumer sequential coupling), after removing low-saliency
channels it **reconstructs the consumer conv weights via im2col least squares**, absorbing the removed channels' contribution into the kept channels so the output is preserved as far as possible
(ThiNet/CP-style compensation, no finetuning required; **in-house implementation, no external/bundled third-party dependencies**):

```python
from amct_pytorch.pruning import CNN_RECONSTRUCT_PRUNE_CFG
amct.prune(model, CNN_RECONSTRUCT_PRUNE_CFG, data=calib)   # calib 提供样例输入用于激活统计
# 或自定义：{"methods": {"cnn": {"name": "reconstruct",
#                                "kwargs": {"prune_ratio": 0.3, "ridge": 1e-2}}}}
```

- The output head (last Linear) is ignored automatically, keeping the class count unchanged; layers matched by `skip_layers` are also ignored.
- Measured (trained MNIST CNN, prune 0.3): naive `variance_channel` acc 0.343 vs `reconstruct` acc 0.574
  (same parameter count, +23pp) — reconstruction is markedly less destructive on trained models.
- Calibration must be representative and diverse (row count ≥ keep×k×k to solve the least squares); when underdetermined or the consumer is not Conv2d, an **explicit warning** is emitted
  and the naive slicing fallback is used (no silent pretense of "reconstructed").
- In a **residual (ResNet)** block the *interior* conv→conv coupling IS pruned; only the conv feeding the residual `add` is auto-excluded (so the residual width is preserved, no damage) — not a wholesale skip. Concat (Inception) consumers and grouped/depthwise convs remain **0 targets** (deferred, see "Supported prunable structures").

## Performance measurements (real trained models + measured int8 on NPU)

16 small models **trained from scratch** (CNN@MNIST / Dense-LM·MoE-LM@WikiText) + 20 real pretrained HF models (10 architecture families),
4 compression budgets × real AMCT int8 (Ascend NPU). Budget-driving metric = **top-1 fidelity** (label-free agreement rate with the original model's predictions);
task metrics are reported (CNN = test acc↑, LM/MoE/HF = perplexity↓).

> **Measurement provenance.** All numbers in these conclusions are **reference measurements**. The package ships only small usage samples under [`examples/algorithms/pruning/`](../../examples/algorithms/pruning/README_en.md).

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
(Note: this sensitivity sweep covers 19 HF models, a subset of the 20 used for the prunability stats in Conclusion 3 — the counts are not directly comparable.)
Allocation is therefore not the default, and `guard='calib_nll'` forces an A/B comparison: when worse than uniform it automatically falls back with a warning.

| Model (layers) | uniform ppl | sensitivity ppl | winner |
|------|:---:|:---:|:---:|
| lm_L6_d160 (6) | 919.8 | **914.0** | sens |
| SmolLM-135M (30) | 120.2 | **117.4** | sens |
| Cerebras-GPT-111M (10) | 134.1 | **128.9** | sens |
| gpt2 (12) | **116.6** | 121.9 | uniform |
| pythia-410m (24) | **132.5** | 157.5 | uniform |

> The signal must be **measured fidelity drop** (raw saliency is not comparable across layers and performs worse).

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

> The 20-model table above is the **full-set** measurement (NPU). A separate **4-model subset** sweep under the *identical* recipe (train[:680] single-pass FT, held[700:714], ppl gate == ppl report, 1% tolerance) cross-validates it — SmolLM-135M **23.66%** ≈ full-set **23.7%** — and extends it to the **recovery menu** {low_variance | reconstruct none/bias/ls | guarded HYBRID}:

| model (+1-epoch FT) | low_variance | none | bias | ls | **HYBRID** |
|------|:---:|:---:|:---:|:---:|:---:|
| pythia-160m | 7.0% | 14.0% | 14.0% | **17.5%** | 17.5% |
| gpt-neo-125M | 9.1% | 18.1% | 22.6% | **27.2%** | 27.2% |
| SmolLM-135M | 11.8% | 17.8% | 17.8% | **23.7%** | 23.7% |
| distilgpt2 (Conv1D) | **0%** | 20.8% | 20.8% | **20.8%** | 20.8% |

> no-FT = **0%** for every model × every method (matches the headline). Post-FT, recovery `ls` ≈ **2–3× the `low_variance` cut**; on distilgpt2's Conv1D FFN `low_variance` finds no `nn.Linear` target (**0%**) while `reconstruct` still cuts **20.8%** — the menu's coverage win. **HYBRID** = the guard's val-pick from {none,bias,ls}, never worse than `none` (here it selects `ls`).

**Honest conclusion**: on well-trained pretrained LMs, *training-free* dense-FFN pruning has almost no headroom at 1% (most models max-cut 0% — structured pruning's immediate damage exceeds 1%; fidelity is insensitive to LM damage but ppl is sensitive). A **1-epoch finetune** (lr 1e-5 + warmup, single pass over distinct data) raises the max cut to **14–61% (median 27%)** — i.e. "spend 1% quality to remove 14–61% of params" (not 1% for 1%), and after finetune most models even beat baseline ppl (WikiText domain adaptation). This is not method failure but the structured-pruning norm: CNN reconstruct (output reconstruction absorbs the damage) is finetune-free, and MoE expert pruning/merge is finetune-free on the toy MoE-LMs — but a real pretrained MoE needs finetune too (granite-1b-a400m: ~0% within budget without finetune, **17%** with one epoch), like dense LM. Finetune hyperparameters must be LM-scale: lr 1e-5 + warmup (1e-4 diverges), single pass over distinct data (many epochs over few batches memorizes the train set and held-ppl explodes).

> **OPT-style embedding projections are auto-excluded.** A model's `project_in`/`project_out` (word-embedding<->hidden projections) must not be treated as a two-layer FFN; the dense domain excludes them by name, so OPT prunes safely (opt-350m cuts 36.5% at the 1% budget).
> Cross-domain comparison (same 1% budget, +finetune max cut): CNN@MNIST (test acc) **31–74%**; toy MoE@WikiText **8–29%**; **real pretrained MoE granite-1b-a400m 17%** (mass + 1-epoch FT, held ppl 18.17→15.79); toy Dense-LM only **2–7%** (embedding-dominated, small FFN fraction — a structural not quality limit).
> The per-model 20-HF numbers and the 4-model menu table are reference measurements. For pruning **API usage**, see the samples under [`examples/algorithms/pruning/`](../../examples/algorithms/pruning/README_en.md).

### Boundaries
- **toy Dense-LM** (from-scratch, embedding dominates the param count) cannot exceed ~7% total-param cut even with finetune — a structural limit (FFN fraction too small), not a quality limit; real pretrained HF (normal FFN fraction) reach **14–61%** with finetune (see Conclusion 3).
- Under a tight quality budget (**−1%**), training-free pruning leaves almost no feasible ratio for pretrained LMs (a small cut already pushes ppl past threshold) — finetune is required.

### Conclusion 4: both guarded HYBRIDs are measured on a small deterministic demo (table + conclusion inline)

- **MoE variance-HYBRID** (synthetic specialized-routing MoE classifier, balanced-acc, ratio 0.5): the
  variance criterion beats mass under imbalance (and slightly at balance). The 2-config demo
  (imbalance 1 & 20) measures — **mass baseline is MoE-only** (cnn/dense pinned to
  0, matching the hybrid → fair same-domain comparison; an unpinned baseline also prunes the experts' dense
  FFN at the default 0.5 and inflates the gain):

  | imbalance | base | mass (MoE-only) | hybrid | pick | gain | criterion / selection |
  |---|---|---|---|---|---|---|
  | 1 | 0.991 | 0.903 | 0.919 | peak | +0.017 | +0.017 / +0.000 |
  | 20 | 0.967 | 0.672 | **0.791** | cond_var | **+0.119** | +0.120 / −0.001 |

  > Single-criterion TEST (no menu) equals the hybrid's pick → the gain is the **criterion itself**, not
  > menu-selection (selection-gain ≈ 0).

- **Dense recovery-HYBRID** — the **real-LM** evidence is **Conclusion 3's menu table** (pretrained LMs,
  1% budget, +FT: the guarded HYBRID selects `ls` and reaches ≈ 2–3× the `low_variance` cut). The table
  below is a **synthetic isolated-ladder illustration** (trained toy classifier, aggressively FFN-pruned,
  **no finetune** — *not* comparable to the 1%-budget LM numbers) that exhibits the ladder
  none(collapses)≪bias<ls and the guard's never-worse val-pick in a clean, deterministic setting:

  | ratio | base | none | bias | ls | hybrid_pick | weight↓% |
  |---|---|---|---|---|---|---|
  | 0.50 | 0.908 | 0.830 | 0.880 | **0.907** | ls | 43.9% |
  | 0.75 | 0.908 | 0.461 | 0.846 | **0.906** | ls | 65.9% |

  Both confirm the accuracy lever for dense-FFN pruning is the post-prune **recovery** (least-squares
  reconstruction), not the neuron-selection criterion.
- **True end-to-end (NPU, quantization actually fires)**: on real SmolLM-135M(fp16), recovery-HYBRID prunes
  the FFN → **real int8 quantization fires** (inserts `NpuWeightQuantizedLinear`) → convert. Measured ppl
  base 1.649 → pruned(−17.8%) **1.699** → +int8 **1.704** (int8 near-free, Δ+0.006, consistent with Conclusion 1);
  the guard **ppl-picks `bias`** (calib < FFN width → LS-free bias is the robust val-winner, +0.016 over none) —
  near-lossless at −17.8% params. So a HYBRID-pruned model goes through the full deployment pipeline. The CPU
  composition (prune→quantize→convert runs; quant is a no-op on CPU) is covered by
  `tests/amct_pytorch/test_pruning_quantize_compose.py`.
- The above are CPU **representative** measurements (synthetic, deterministic, seconds). Larger-scale
  validation (multi-arch MoE × imbalance × calib size) is consistent: variance gain rises monotonically
  with imbalance; recovery is the dense/CNN lever and the guard rides the bias↔ls transition with
  calibration size.

## Advanced configuration

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

## Guarded variance-HYBRID (MoE expert pruning)

Variance-style routing scores come from a single calibration pass (no gradients / least-squares /
expert forward) and beat activation-frequency (mass) pruning on *some* MoEs — but there is no
universal "better variance" criterion (the best one depends on routing structure).
`guarded_variance_hybrid_prune` does not bet on one criterion: it runs **one calibration → prunes a
copy per criterion in the MENU → picks the best on a small validation set → applies the winner**:

```python
from amct_pytorch.pruning import guarded_variance_hybrid_prune

res = guarded_variance_hybrid_prune(
    model, data=calib, prune_ratio=0.5, top_k=8, eval_data=val)  # keep val != calib
print(res.summary())     # chosen criterion + validation gain over fallback(mass) + reduction
# res.chosen_criterion / res.criterion_quality / res.gain_over_fallback (guard ensures >= 0)
```

- **MENU** (default): `mass` (= `activation_count`, the fallback) / `cond_var` (Var(w|sel)) /
  `peak` (max w — keeps consistently-strong specialists that cond misses) / `cvxpeak` (Var(w|sel)×max w).
- **Guard guarantee**: switch away from `mass` only when a variance criterion is *strictly better* on
  the **validation set**, otherwise fall back to `mass` — so the HYBRID is **never worse than fallback**
  on validation. (Mirrors `accuracy_based_auto_prune`'s ratio search, but searches over the *criterion*;
  both reuse the same deepcopy-trial + quality_fn guard + in-place apply machinery.)
- The gain is largest under **class/expert-usage imbalance + a tail-sensitive metric** (e.g. many-expert
  group-routed MoEs).
- **Honest scope**: the primary in-repo evidence is the 2-config demo in Conclusion 4
  (+0.017 balanced, **+0.119** at imbalance 20, MoE-only baseline). On **real pretrained MoE** the
  criterion is **depth-dependent** — a *global* variance criterion (every layer) loses to mass, but
  variance on the **deep layers only** (`boundary=K`) beats mass on multi-expert routers (OLMoE −25%
  held ppl, LLaDA lower KL) — see **Depth-boundary expert selection on real MoE** below. The average
  **+6pp** over **60 mini-suite configs** (synthetic-init experts, rising with imbalance) is **[EXTERNAL —
  not reproduced in this repo]**; on LLaDA2.1-mini **GSM8K** aggressive *static* expert pruning collapses
  (dynamic masking needed). Net: the real-MoE gain needs the **depth boundary**, not a global criterion;
  "never worse" remains a VAL guarantee (granite's few-expert router correctly collapses to mass).
- The underlying per-criterion score is the new `mass_variance` kwarg `variance_score` ∈
  {`cond` (default), `peak`, `cvxpeak`}, with a depth split `boundary`: **mass on layers ≤ boundary,
  variance on layers > boundary** (default `10`, validated on LLaDA2.1-mini; `-1` = every layer variance;
  `"auto"` = per-layer pick; `list[int]` = variance on exactly those layers). The default `cond` score is
  byte-for-byte the previous behavior (backward compatible). `guarded_variance_hybrid_prune`'s
  `DEFAULT_MENU` uses the global `boundary=-1`; pass a custom `menu=` with `boundary=K` to search depth splits.

### Depth-boundary expert selection on real MoE

The expert-keep criterion is **depth-dependent**: in the **deep** (last) MoE layers per-expert routing-weight
**variance** selects better survivors than activation **mass**; **shallow** layers favor mass; **a global
all-variance criterion underperforms the depth split on every model**. `mass_variance` splits by a depth
`boundary` K (mass on layers ≤ K, variance on
layers > K). At a fixed prune ratio, deep-layer variance beats mass-only on real pretrained MoE:

| Real MoE (router, MoE layers) | metric ↓ | all-mass | var-deep (best K) | all-var |
|---|---|---|---|---|
| OLMoE-1B-7B (softmax, E64, 16L) | held ppl @0.30 | 46.9 | **35.3** (var last 1) | 38.1 |
| LLaDA2.1-mini (group-limited, E256, 19L) | KL @0.20 (diffusion) | 0.0392 | **0.0391** (var last 10) | 0.0634 |
| granite-1b-a400m (softmax, E32, 24L) | held ppl @0.30 | **94.8** | 94.8 (K=L → mass) | 1487 |

granite's 32-expert softmax router is too small for a reliable per-expert variance signal (all-variance ppl
1487 ≫ mass 94.8) → the boundary sweep **correctly collapses to all-mass** (never worse than mass). The gain
grows with expert count (OLMoE E64). At the −1% quality budget the cut is finetune-gated (real pretrained MoE
cuts ~0% without finetune): OLMoE **0% → 23%** with 1-epoch finetune, and var-deep finetunes to strictly
lower ppl than mass at the same cut (15%: 18.96 vs 19.35; 23%: **20.69 vs 21.85**) — the deep-variance edge
survives finetune. (LLaDA is a diffusion model — KL only, no finetune.)

## Guarded recovery-HYBRID (dense FFN / CNN channel pruning)

For dense FFN / CNN channel pruning the accuracy lever is the post-prune **RECOVERY**, not the neuron-
selection criterion. `guarded_recovery_hybrid_prune` mirrors the variance-HYBRID but searches the
recovery MENU:

```python
from amct_pytorch.pruning import guarded_recovery_hybrid_prune

res = guarded_recovery_hybrid_prune(
    model, data=calib, prune_ratio=0.5, domain="dense", eval_data=val)  # or domain="cnn"
print(res.summary())     # chosen recovery + validation gain over none + reduction
```

- **MENU** (default): `none` (naive slice, fallback) / `bias` (mean-fold into the consumer bias, no
  least-squares) / `ls` (least-squares reconstruction of the consumer weight). Backed by the new
  `reconstruct` kwarg `recovery` ∈ {`ls` (default), `bias`, `none`} (added to both dense and cnn;
  the default `ls` is the previous behavior, backward compatible).
- **Guard guarantee (be honest about its scope)**: the guard switches off the fallback only on a STRICT
  win **on the validation set** → **never worse than the no-recovery baseline ON VALIDATION**. This is a
  VAL guarantee, NOT a test guarantee: if `eval_data` is not representative of the deployment distribution
  the val-pick may not transfer (see the small-calib note). Use `eval_data` representative of deployment.
- **Calibration size is the key knob**: ample calib → `ls` wins; calib < intermediate width → `ls` is
  underdetermined and overfits, `bias` (LS-free) is more robust. The guard rides the bias↔ls transition
  automatically with calibration size.
- **Small-calib safety knob (dense, measured)**: `bias`-fold overfits the calibration region's activation
  mean and can hurt on a distribution-shifted test set. **[EXTERNAL — not reproduced in-repo]** dense 54 configs @ r0.5, no-FT: ~**2%
  of small-calib configs regress on test** (worst −28% ppl), and **neither a larger val set nor a stricter
  margin guard removes it** (structural val↔test mismatch, not sampling noise). For a test-side never-worse
  guarantee pass `recovery_menu=("none", "ls")` to drop bias — trading ~10pp of small-calib mean gain.
- **CNN has no such floor (measured)**: in CNN channel pruning the im2col rows = N×spatial are abundant
  even at 16 images, so `ls` is not underdetermined and `bias` is rarely the val-winner — **[EXTERNAL — not reproduced in-repo]** across
  108 configs with **0% test-side regression**; the full guard is already test-side safe on CNN.
- **Measured scope**: all dense/CNN numbers above are **no-FT** (deploy-as-is, no retraining) relative
  comparisons; recovery is the lever in that no-fine-tune regime. With post-prune fine-tuning the naive
  slice may catch up — a separate question.

## Combined use with quantization (prune → quantize → convert)

Pruning (structured, removing channels/neurons/experts) and quantization (lowering bits) are **orthogonal** compression axes: size gains
**multiply**, accuracy losses are **approximately additive**. Recommended order: **prune first, then quantize, finally convert the deployment model** —
pruning changes tensor shapes, so quantization calibration is only correct on the pruned weights.

```python
import amct_pytorch as amct
from amct_pytorch.pruning import accuracy_based_auto_prune

accuracy_based_auto_prune(model, data=calib, tolerance=0.02)  # 1) 结构化剪枝
amct.quantize(model, quant_cfg)                                # 2) 在剪枝后的模型上量化
amct.convert(model)                                            # 3) 转换部署模型
```

**Joint accuracy budget**: `accuracy_based_auto_prune_with_budget` **splits the total budget across the two steps and measures sequentially** —
pruning consumes a measured portion first, then the quantization search runs on the *pruned model* with the remaining budget:

```python
from amct_pytorch.pruning import accuracy_based_auto_prune_with_budget

jb = accuracy_based_auto_prune_with_budget(
    model, data=calib, total_tolerance=0.03, prune_fraction=0.5)  # 总预算 3%，先给剪枝一半上界
print(jb.summary())                       # 剪枝实测用掉 jb.prune_drop
# 把剩余预算作为量化精度搜索的接受阈值（与量化共用同一 evaluator）
amct.accuracy_based_auto_calibration(model, ev, quant_cfg, ..., )   # 预算 = jb.remaining_tolerance
```

`jb.remaining_tolerance = max(0, total_tolerance - measured prune loss)`, connecting naturally to the quantization-side
[`accuracy_based_auto_calibration`](../../docs/zh/api/accuracy_based_auto_calibration.md).

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
future absorption: the shared accuracy abstraction `AutoCalibrationEvaluatorBase` and sequential budget hand-off
(`accuracy_based_auto_prune_with_budget` → quantization `accuracy_based_auto_calibration`). For now it coexists
risk-free as a self-contained, purely additive module (`amct_pytorch.pruning`).

### Supported prunable structures
Structured pruning targets only dimensions whose producer↔consumer interface can be verified, and conservatively skips the rest (never mis-pruned):
- **Dense FFN** — prunes the *intermediate* dimension only (`gate/up.out_features` + `down.in_features`); the hidden/residual width (`down.out_features`) is preserved. Attention projections (q/k/v/o) are auto-excluded.
- **CNN channels** — a producer `Conv2d` (`groups=1`) → optional `BatchNorm2d` → consumer `Conv2d`/`Linear` with `consumer.in_channels == producer.out_channels`; producer-out, BN, and consumer-in are co-resized together. In a **residual (ResNet)** block the *interior* conv→conv coupling is pruned (the conv feeding the residual `add` is auto-excluded → residual width preserved). **Concat (Inception)** consumers (`in = Σ branches`) and **grouped/depthwise** convs are **not pruned** (0 targets) — verifying their safety needs cross-branch dependency tracking, deferred.
- **MoE experts** — whole *routed* experts are removed; the layer input/output (hidden) is unchanged. **Shared (always-active) experts are excluded** from routed-expert pruning.

## Current support boundary (measured with current transformers)

| Domain | Supported | Notes |
|----|------|------|
| dense | ✅ | LLM FFN: the three-Linear `gate/up/down_proj`, **fused `gate_up_proj`** (Phi-3/GLM-4 style, first half gate, second half up), adjacent two Linear/Conv1D layers (including Bloom-style custom parameter-free activation between them). The dense domain structurally excludes parallel attention projections (q/k/v/o share input and do not satisfy the in→inter→in residual-dim restoration), so Llama/Qwen2/Mistral/Qwen3 prune safely **without manual `skip_layers`**. |
| cnn | ✅ | `variance_channel` (heuristic) naive slicing; **`reconstruct` (im2col least squares) output-preserving, low damage** (measured on trained MNIST CNN @prune0.3: variance 0.343 → reconstruct 0.574). ResNet-style independent-block structures are not recognized by same-level detection (known boundary). |
| moe | ✅ | Legacy `nn.ModuleList` experts + `nn.Linear` gate, **and modern fused (batched) experts** (`MixtralExperts`/`Qwen3MoeExperts` + `*TopKRouter`, batched tensors sliced along the expert dim). Also handles: **grouped routers** (`n_routed_experts`/`n_group`/`topk_group` — syncs the expert count and collapses groups to 1; DeepseekV3-style), **sibling two-tensor fused experts** (experts split across parallel sub-modules, e.g. GraniteMoE `input_linear`/`output_linear`), and **nested router-bias** slicing (e.g. Ernie4.5 `moe_statics.e_score_correction_bias`). Proven by `test_pruning_real_hf_models.py` (Mixtral, Qwen3Moe, GraniteMoe, DeepseekV3, Ernie4_5_Moe). |

**Architecture coverage matrix** (configs shrunk to 2-layer micro models, random weights — verifies mechanical prunability only): covers 29 architecture families —
llama/llama3/qwen2/qwen3/qwen2_moe/qwen3_moe/mixtral/mistral/gemma/gemma2/gemma3/phi/phi3/glm4/
gpt2/gpt_neox/gpt_neo/gpt_bigcode/starcoder2/opt/xglm/olmo/olmo2/stablelm/smollm3/bloom/
granitemoe/deepseekv3/ernie4_5.

Regression baseline: `tests/amct_pytorch/test_pruning_real_hf_models.py`; tolerance feature:
`tests/amct_pytorch/test_pruning_auto_prune.py` (requires transformers).

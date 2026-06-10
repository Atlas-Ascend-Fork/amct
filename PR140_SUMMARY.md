# PR #140 — Structured Pruning for AMCT (`experimental.pruning`)

## Page 1 — Functional changes + how to use prune **with** quantization

### What changed (functional summary)

- **New eager structured-pruning subpackage** `amct_pytorch.experimental.pruning`, complementary to the
  existing graph-based channel pruning. Façade aligned 1:1 with `amct.quantize` (in-place, dict config,
  `get_last_report()`). Domains: **dense FFN neurons / CNN channels / MoE experts**.
- **Tolerance-driven auto ratio search** — `accuracy_based_auto_prune(tolerance=…)`: state an accuracy
  budget, it binary-searches the largest ratio that fits (label-free fidelity / custom `quality_fn` / shared
  `evaluator`).
- **Composes with quantization** — `PrunePass` plugs into `ModelOptimizer`; shares the quantization accuracy
  abstraction (`AutoCalibrationEvaluatorBase`); **joint accuracy budget** split between the two steps
  (`accuracy_based_auto_prune_with_budget` → `JointBudget`).
- **Optional fine-tune recovery** — `finetune(...)` to recover accuracy after pruning; the search can score
  the **recovered** model via `finetune_fn=`.
- **Dependency-graph CNN pruning** (residual/downsample/depthwise). Importance criteria:
  `magnitude / l1 / lamp / bn / random / fpgm / taylor / hessian`. Plus `global_pruning`, `round_to`
  (HW channel alignment), `iterative_steps`.
- **Purely additive** (no existing file modified); vendored Torch-Pruning v1.6.1 (MIT), no external deps.

### Sample: apply pruning and quantization together

```python
import amct_pytorch as amct
import amct_pytorch.experimental.pruning as P
from amct_pytorch.experimental.pruning import finetune

# --- 1) Structured pruning ---------------------------------------------------
# (a) tolerance-driven (recommended): "at most 2% quality loss", auto ratio
P.prune(model, data=calib, tolerance=0.02)
print(P.get_last_auto_prune().summary())

# (b) or a fixed ratio
# P.prune(model, {"methods": {"dense": {"name": "low_variance",
#                                        "kwargs": {"prune_ratio": 0.5}}}}, data=calib)

# --- 2) Optional: fine-tune to recover accuracy ------------------------------
finetune(model, train_data, steps=300)          # one-shot pruning collapses acc; recover it

# --- 3) Quantize the PRUNED model, then convert (INT8, on Ascend NPU) --------
amct.quantize(model)                             # weights must be fp16/bf16 on NPU
amct.convert(model)
```

**Joint accuracy budget** (split one budget across prune → quant):

```python
jb = P.accuracy_based_auto_prune_with_budget(model, data=calib,
                                             total_tolerance=0.03, prune_fraction=0.5)
# prune used jb.prune_drop; feed jb.remaining_tolerance to the quantization search
```

> Notes: recommended order is **prune → (fine-tune) → quantize → convert** — quantization must calibrate on the
> pruned weights. AMCT INT8 weight quant requires **fp16/bf16** weights and **torch_npu** (NPU); on CPU/fp32 it
> is a no-op.

---

## Page 2 — All measured performance (real pretrained weights)

### 1) Prune + INT8 quant composition — losses are additive (SmolLM-135M, fp16, NPU3, perplexity)

| Stage | Perplexity | Δ |
|---|---|---|
| baseline (fp16) | 1.4527 | — |
| prune (dense 0.1) | 1.4679 | **+0.0152** |
| + INT8 weight quant | 1.4712 | **+0.0033** |

### 2) Importance criterion matters — magnitude vs taylor (resnet18, real ImageNet, 30% channel prune, no fine-tune)

| importance | top-1 decision agreement vs original |
|---|---|
| magnitude | **0.000** (destroyed) |
| taylor | **0.938** (preserved) |

### 3) Full real-accuracy E2E — train → prune → fine-tune → quantize (MNIST, labeled test accuracy)

| Stage | Test accuracy | Note |
|---|---|---|
| trained baseline | 0.857 | small CNN |
| prune (taylor 40% + round8) | 0.100 | 74% of params removed |
| fine-tune (300 steps) | **0.810** | recovered +0.709 |
| INT8 quantize (NPU3) | 0.812 | lossless |

### 4) Auto search × fine-tune — recovery changes the chosen ratio (MNIST, real test acc, tolerance 5%)

| Prune ratio | Search WITHOUT finetune | Search WITH finetune (recovered acc) |
|---|---|---|
| 0.30 | acc 0.148 → reject | 0.864 → accept |
| 0.40 | — | 0.825 → accept |
| 0.50 | — | 0.779 → reject |
| **Selected** | **None (0% pruning)** | **0.40 (40% channel pruning)** |

### 5) Fixed-ratio prune → perplexity curve (SmolLM-135M, dense FFN, CPU fp32)

| ratio | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| perplexity (base 4.10) | 5.05 | 5.80 | 6.63 | 7.73 | 12.16 |

### 6) Model support matrix — LLM (dense FFN, prune 0.2, perplexity)

| Model | Family | Status | dense targets | param reduction | ppl before → after |
|---|---|---|---|---|---|
| SmolLM-135M | Llama | ✅ | 30 | 11.8% | 3.50 → 5.52 |
| opt-125m | OPT | ✅ | 12 | 9.0% | 4.17 → 14.06 |
| pythia-160m | GPTNeoX | ✅ | 12 | 7.0% | 3.86 → 12.61 |
| Qwen2.5-0.5B | Qwen2 | ✅ | 24 | 12.7% | 2.81 → 4.00 |
| gpt2 | GPT2 (Conv1D) | ❌ | 0 | 0% | unchanged |
| distilgpt2 | GPT2 (Conv1D) | ❌ | 0 | 0% | unchanged |

→ `nn.Linear` FFN families supported; GPT2 (Conv1D FFN) correctly detected as unsupported (0 targets, model untouched).

### 7) Model support matrix — CNN (dependency-graph, prune 0.3)

| Model | resnet18 | mobilenet_v2 | regnet_y_400mf | convnext_tiny | vgg11_bn | squeezenet1_1 |
|---|---|---|---|---|---|---|
| status | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| param reduction | 50% | 43% | 54% | 72% | 72% | 52% |

→ All six families prune and forward. (Raw agreement collapses under magnitude/no-finetune — see rows 2–4 for the taylor + fine-tune recovery.)

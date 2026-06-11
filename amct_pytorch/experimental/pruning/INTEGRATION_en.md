# Integration with the existing graph pruning subsystem (non-destructive, natural absorption)

This document describes how `experimental.pruning` (eager structured pruning) is **aligned with the existing
`classic/graph_based` pruning subsystem and can eventually be absorbed naturally**, while **never modifying or breaking** existing code.

## 1. Positioning: complementary, not a replacement

| Dimension | Existing graph subsystem | New experimental.pruning |
|------|-------------------|--------------------------|
| Representation | ONNX graph (export→parse) | direct traversal of eager `torch.nn.Module` |
| Configuration | proto `.cfg` file | in-memory `dict`/`PruneConfig` |
| Coverage | CNN conv channels + retrain | **LLM dense FFN / MoE experts** + modern CNN |
| Deployment | Ascend perf-gain aware, cfg artifacts | pure structural trimming, in-place |
| Entry points | `auto_channel_prune_search`, `create_prune_retrain_model` | `prune`, `accuracy_based_auto_prune` |

→ The existing subsystem excels at *CNN + retrain + Ascend deployment*; the new module covers *Transformer/MoE/label-free*, which the existing one does not.
They are two paths of the same "pruning family": **no name collisions, no cross-modification of code**.

## 2. Alignment already in place (the new module speaks the existing vocabulary)

These are the key points that make the new module **absorbable**, all implemented:

1. **Shared accuracy abstraction** (same as quantization's `accuracy_based_auto_calibration`):
   `accuracy_based_auto_prune(evaluator=...)` directly accepts `AutoCalibrationEvaluatorBase`
   subclasses (`evaluate(model)->metric`). Also compatible with `evaluate(model, iterations)`.
   Passing the graph-side `ModelEvaluator` (a data feeder that returns None) by mistake raises a clear error.
2. **Quant-aware saliency**: `reconstruct`/`low_variance` support `quant_aware=True`, scoring
   neurons with **quantized** weights so that "what to prune" matches the deployment quantization scheme — same lineage as the existing quantization-side sensitivity analysis, easing future unification.
3. **Sequential budget hand-off**: `accuracy_based_auto_prune_with_budget(total_tolerance, prune_fraction)`
   produces `remaining_tolerance`, used directly as the acceptance budget of the subsequent quantization `accuracy_based_auto_calibration` —
   a "split and measure sequentially" joint prune→quant budget.
4. **Consistent Pass protocol**: `PrunePass.run(model)` follows the `ModelOptimizer` `run(model)` protocol
   and can live in one optimization pipeline with `InsertQuantizeModulePass` (note: quantization layer_config must be resolved after pruning).

## 3. Naming alignment table (correspondence at absorption time)

| Existing (graph) | New (eager) | Semantic relation |
|----------------|--------------|----------|
| `auto_channel_prune_search` (emits cfg) | `accuracy_based_auto_prune` (in-place) | auto search; the latter + `_with_budget` connects to budgets |
| quantization-side sensitivity analysis | `quant_aware=True` (quantized-state scoring) | saliency aligned with the deployment quantization scheme |
| `ModelEvaluator` (feeder, returns None) | — | **not** usable for accuracy search |
| `AutoCalibrationEvaluatorBase.evaluate`→metric | `evaluator=` | **shared** accuracy abstraction |
| `create_prune_retrain_model` | (no retrain yet) | exclusive to existing subsystem, new module stays out |

## 4. Absorption roadmap (proposal, zero breakage)

- **Status quo**: the new module is isolated in `experimental/`, purely additive (0 changes to existing files).
- **Step 1 (done)**: align evaluator/sensitivity/budget/Pass vocabulary so concepts interoperate.
- **Step 2 (pending, needs consensus)**: add an *eager backend option* to the existing `auto_channel_prune_search`
  (e.g. `backend="eager"`), making this module its implementation for Transformer/MoE, **still one external entry point**;
  the existing graph backend stays default, behavior unchanged.
- **Step 3 (pending)**: merge documentation into `docs/zh/api/`, as a section next to `auto_channel_prune_search.md`.

> Steps 2/3 touch existing public APIs and require prior Issue discussion and consensus (AGENTS.md process).
> Until then, the new module coexists at zero risk in its experimental form.

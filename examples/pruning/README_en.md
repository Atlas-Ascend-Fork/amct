# experimental.pruning practical examples

**Directly runnable** examples for each feature of `amct_pytorch.experimental.pruning`. All models are randomly initialized
micro networks; **no weights are downloaded**, pure CPU is sufficient (CI/offline/no-NPU friendly).

## Running

```bash
# pick any (scripts share _models.py via relative import; run from this directory)
cd examples/pruning
python fixed_ratio_dense.py
python tolerance_auto_prune.py
python cnn_channel_prune.py
python moe_expert_prune.py
python prune_then_quantize.py      # 量化部分缺环境时自动跳过
python evaluator_and_diagnose.py
```

## Examples and covered features

| Script | Demonstrated feature | Corresponding API |
|------|----------|----------|
| `fixed_ratio_dense.py` | fixed prune ratio, dict config, in-place, report | `prune` / `get_last_report` |
| `tolerance_auto_prune.py` | tolerance-driven + **binary search** (tight tolerance prunes nothing / relaxed selects a ratio) | `prune(tolerance=)` / `get_last_auto_prune` |
| `cnn_channel_prune.py` | CNN channel pruning: output-preserving reconstruction vs naive slicing, presets | `prune` + `CNN_RECONSTRUCT_PRUNE_CFG` |
| `moe_expert_prune.py` | MoE expert pruning (activation_count / mass_variance) | `prune` (moe domain) |
| `moe_variance_hybrid.py` | Guarded variance-HYBRID: criterion MENU picked on a val set, never worse than fallback(mass) **on validation** | `guarded_variance_hybrid_prune` |
| `recovery_hybrid.py` | Guarded recovery-HYBRID: dense/CNN recovery MENU {none,bias,ls} picked on a val set | `guarded_recovery_hybrid_prune` |
| `prune_then_quantize.py` | prune→quantize→convert; functional and PrunePass paths | `prune` / `PrunePass` + `amct.quantize/convert` |
| `evaluator_and_diagnose.py` | custom evaluator / quality_fn / feasibility diagnostics | `accuracy_based_auto_prune` / `diagnose` |

## End-to-end reproduction scripts (committed numbers)

These reproduce the inline tables / conclusions in `experimental/pruning/README.md` (CPU-runnable; faster on NPU):

| Script | Reproduces | Corresponding API |
|------|----------|----------|
| `e2e_variance_hybrid.py` | MoE variance-HYBRID 2-config demo (imbalance 1 & 20) | `guarded_variance_hybrid_prune` |
| `e2e_recovery_hybrid.py` | Dense recovery-HYBRID synthetic isolated-ladder (none≪bias<ls) | `guarded_recovery_hybrid_prune` |
| `e2e_dense_budget_menu.py` | Real-LM 1%-budget recovery menu (4-model in-repo subset) | `prune` + recovery menu |
| `e2e_recovery_hybrid_npu_quant.py` | True end-to-end on NPU (recovery-HYBRID prune → real int8 quantize → convert) | `prune` + `amct.quantize/convert` |

## Quick reference: the three pruning domains

| Domain | Triggering structure | Method names |
|----|----------|--------|
| dense | LLM FFN (gate/up/down or two adjacent Linear) | `low_variance` |
| cnn | conv channels (producer→bn→consumer sequential coupling) | `variance_channel` / `reconstruct` |
| moe | gate + ModuleList experts / modern fused experts | `activation_count` / `mass_variance` / `output_merge` (ModuleList experts only) |

## Notes

- **Combined order with quantization**: prune first, then quantize — quantization must resolve `layer_config` on the pruned shapes,
  so `PrunePass` and `InsertQuantizeModulePass` cannot be chained blindly in one `ModelOptimizer`
  (see the path_b comment in `prune_then_quantize.py`).
- **Evaluator reuse**: shares the accuracy abstraction with the quantization-side `accuracy_based_auto_calibration`;
  the object only needs `.evaluate(model, iterations) -> float`. Note the graph-side built-in `ModelEvaluator.evaluate`
  **returns nothing** (it is a calibration data feeder); use a metric-returning implementation (see `evaluator_and_diagnose.py`).
- For real accuracy evaluation, replace the proxy quality functions in the examples with "run the model on a validation set and return accuracy/mAP".

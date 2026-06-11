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
| `prune_then_quantize.py` | prune→quantize→convert; functional and PrunePass paths | `prune` / `PrunePass` + `amct.quantize/convert` |
| `evaluator_and_diagnose.py` | custom evaluator / quality_fn / feasibility diagnostics | `accuracy_based_auto_prune` / `diagnose` |

## Quick reference: the three pruning domains

| Domain | Triggering structure | Method names |
|----|----------|--------|
| dense | LLM FFN (gate/up/down or two adjacent Linear) | `low_variance` |
| cnn | conv channels (producer→bn→consumer sequential coupling) | `variance_channel` / `reconstruct` |
| moe | gate + ModuleList experts / modern fused experts | `activation_count` / `mass_variance` |

## Notes

- **Combined order with quantization**: prune first, then quantize — quantization must resolve `layer_config` on the pruned shapes,
  so `PrunePass` and `InsertQuantizeModulePass` cannot be chained blindly in one `ModelOptimizer`
  (see the path_b comment in `prune_then_quantize.py`).
- **Evaluator reuse**: shares the accuracy abstraction with the quantization-side `accuracy_based_auto_calibration`;
  the object only needs `.evaluate(model, iterations) -> float`. Note the graph-side built-in `ModelEvaluator.evaluate`
  **returns nothing** (it is a calibration data feeder); use a metric-returning implementation (see `evaluator_and_diagnose.py`).
- For real accuracy evaluation, replace the proxy quality functions in the examples with "run the model on a validation set and return accuracy/mAP".

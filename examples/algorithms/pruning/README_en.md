# AMCT Structured Pruning Samples

This sample demonstrates how to call the structured-pruning interface of
`amct_pytorch.pruning` across the three domains:
**dense FFN (prune the intermediate dim) / CNN (prune channels) / MoE (prune experts)**.
The samples use randomly-initialized tiny models — they **download no weights and run on pure CPU**,
so you can get started quickly without network or NPU access.

> For the algorithm details, compression ratios and accuracy conclusions, see the pruning package
> doc [`amct_pytorch/pruning/README_en.md`](../../../amct_pytorch/pruning/README_en.md).

## 1 Pruning Prerequisites

### 1.1 Install Dependencies

The dependencies for this sample are in [requirements.txt](requirements.txt). Both `torch` and
`transformers` are required (`transformers` is pulled in by the amct_pytorch import chain); to run
on NPU you also need a `torch_npu` matching your Python/torch versions and an
installed CANN package. The pruning interface itself ships with the amct_pytorch tool.

### 1.2 Model and Data Preparation

The sample models and data are built by [src/utils.py](src/utils.py) with fixed random seeds
(`MiniMLP`/`MiniCNN`/`MiniMoE`) — **no download, no network**. Replace them with real models and
calibration data for actual use.

### 1.3 Pruning Configuration

Pruning takes a dict config passed directly to `prune()` (same style as `amct.quantize`); pick a
method per domain:

| Domain | Method | Description |
|:--|:--|:--|
| dense | `low_variance` | Prune FFN intermediate dim by activation variance (auto-skips attention projections) |
| dense | `reconstruct` | Least-squares reconstruction to compensate after pruning, recovery ∈ {none, bias, ls} |
| cnn | `variance_channel` | Naive channel slicing by activation variance |
| cnn | `reconstruct` | Output-reconstruction channel pruning (lower damage) |
| moe | `activation_count` | Prune experts by activation frequency, shrink the gate accordingly |
| moe | `mass_variance` | Prune experts by expert mass-variance |

Passing only `tolerance` runs auto-pruning: binary-search the largest prune ratio on `ratio_grid`
that meets the tolerance. The guarded HYBRIDs (`guarded_recovery_hybrid_prune` /
`guarded_variance_hybrid_prune`) pick the best MENU entry measured on a separate small validation
set, **guaranteed never worse than the fallback**.

## 2 Pruning Example

### 2.1 Use Interface Method to Call

Run the samples from the current directory (pure CPU is fine):

```bash
python3 src/run_dense_samples.py   # dense: fixed-ratio / tolerance-auto / recovery-HYBRID / prune+quantize / evaluator·diagnose
python3 src/run_cnn_samples.py     # cnn: variance vs reconstruct channel pruning / recovery-HYBRID
python3 src/run_moe_samples.py     # moe: activation_count vs mass_variance expert pruning / variance-HYBRID
```

Each sample prints params before/after pruning, the reduction ratio, and runs a forward check.

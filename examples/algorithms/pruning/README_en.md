# AMCT Structured Pruning Samples

Calling the `amct_pytorch.pruning` interface across three domains: dense FFN (intermediate dim) /
CNN (channels) / MoE (experts). The directory contains both tiny random-model demos and the
Qwen3.6-35B-A3B community-task sample.

> API details: [`amct_pytorch/pruning/README_en.md`](../../../amct_pytorch/pruning/README_en.md).

## 1 Pruning Prerequisites

### 1.1 Install Dependencies

Dependencies are in [requirements.txt](requirements.txt): `torch` and `transformers` (pulled in by the
amct_pytorch import chain). To run on NPU you also need a `torch_npu` matching your Python/torch versions
and an installed CANN package.

### 1.2 Model and Data Preparation

Sample models and data are built by [src/utils.py](src/utils.py) with fixed seeds
(`MiniMLP`/`MiniCNN`/`MiniMoE`); no download, no network. Replace them with real models and calibration
data for actual use.

### 1.3 Pruning Configuration

Pass a dict config directly to `prune()` (same style as `amct.quantize`); pick a method per domain:

| Domain | Method | Description |
|:--|:--|:--|
| dense | `low_variance` | Prune FFN intermediate dim by activation variance (auto-skips attention) |
| dense | `reconstruct` | Least-squares compensation after pruning, recovery in {none, bias, ls} |
| cnn | `variance_channel` | Naive channel slicing by activation variance |
| cnn | `reconstruct` | Output-reconstruction channel pruning |
| moe | `activation_count` | Prune experts by activation frequency, shrink the gate |
| moe | `mass_variance` | Prune experts by expert mass-variance |

Passing only `tolerance` runs auto-pruning: binary-search the largest prune ratio on `ratio_grid` that
meets the tolerance. With a menu config (`MOE_VARIANCE_MENU_CFG` / `DENSE_RECOVERY_MENU_CFG` / `CNN_RECOVERY_MENU_CFG`),
`prune` switches to MENU selection: it measures every candidate on the separate small validation
set given by `eval_data` and applies the best one.

## 2 Pruning Example

### 2.1 Use Interface Method to Call

Run from the current directory (CPU is fine):

```bash
python3 src/run_dense_samples.py   # dense: fixed-ratio / tolerance-auto / recovery-menu / prune+quantize / evaluator
python3 src/run_cnn_samples.py     # cnn: variance vs reconstruct channel pruning / recovery-menu
python3 src/run_moe_samples.py     # moe: activation_count vs mass_variance expert pruning / variance-menu
```

Each sample prints params before/after pruning, the reduction ratio, and runs a forward check.

## 3 Community Task Sample: Qwen3.6-35B-A3B MoE Expert Pruning

This section covers issue 183 task 1: `activation_count` + `tolerance`.

### 3.1 Run Instructions

- Calibration set: Pileval, default `1` sample, `seq_len=512`
- Evaluation set: WikiText2 test split, fixed `seq_len=4096`, with the first `1` batch used by default
- Search mode: `tolerance`
- Output directory: relative placeholder; the script saves the pruned model, `PruneReport`, and run config
- Pruning runs on CPU and needs enough host RAM for Qwen3.6-35B-A3B
- The datasets are loaded online through `datasets`: Pileval for calibration and the WikiText2 test split

Run commands:

```bash
python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_tolerance.py \
  --model-path ./path/to/Qwen3.6-35B-A3B \
  --save-dir ./outputs/qwen3.6_activation_count_tolerance/t0_1 \
  --tolerance 0.1 \
  --calib-samples 1 \
  --calib-seq-len 512 \
  --eval-batches 1

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_tolerance.py \
  --model-path ./path/to/Qwen3.6-35B-A3B \
  --save-dir ./outputs/qwen3.6_activation_count_tolerance/t0_2 \
  --tolerance 0.2 \
  --calib-samples 1 \
  --calib-seq-len 512 \
  --eval-batches 1

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_tolerance.py \
  --model-path ./path/to/Qwen3.6-35B-A3B \
  --save-dir ./outputs/qwen3.6_activation_count_tolerance/t0_5 \
  --tolerance 0.5 \
  --calib-samples 1 \
  --calib-seq-len 512 \
  --eval-batches 1
```

`--preview-prune-ratio` is used only by `prune_diagnose()` for the dry run, and defaults to `0.1`.
The search grid uses the built-in `0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8` values; pass `--ratio-grid` to override.

If you want to feed the pruned result into the amct eval / quantization flow, see the VL config note in
[Qwen3.6-Moe-Pruning_en.md](../../models/qwen3.6/Qwen3.6-Moe-Pruning_en.md).

### 3.2 Diagnostics and Report

The script prints, in order:

1. Full runtime arguments
2. The `prune_diagnose()` summary
3. The post-pruning `PruneReport`
4. Baseline / pruned PPL and parameter counts

The measured `tolerance=0.1` result is:

```text
[prune-diagnose]
[prune-diagnose] prunable targets: cnn=0, dense=40, moe=40
  fixed-ratio prune: available (cut 9.4%, forward OK)
  acc binary search: available (chosen prune_ratio=0.7)

[PruneReport]
{
  "backend": "ModelBackend(name='huggingface')",
  "params_before": 34660610688,
  "params_after": 34660610688,
  "events": [],
  "budget_unreachable": false,
  "prunable_fraction": null,
  "warnings": [
    "no prune ratio met tolerance 0.100 across 3 candidates; model left unchanged. Relax the tolerance or add a smaller prune ratio to ratio_grid."
  ],
  "per_layer_sparsity": {},
  "allocation_choice": null
}
```

### 3.3 Result Table

| Model | Method | Setting | Baseline PPL | Pruned PPL | Post-process PPL | Params (before -> after) | Reduction | Prune time (min) |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3.6-35B-A3B | `activation_count` | `tolerance=0.1` | 5.405949 | 5.405949 | N/A | 34660610688 -> 34660610688 | 0.00% | 4.94 |
| Qwen3.6-35B-A3B | `activation_count` | `tolerance=0.2` | 5.405949 | 5.405949 | N/A | 34660610688 -> 34660610688 | 0.00% | 2.54 |
| Qwen3.6-35B-A3B | `activation_count` | `tolerance=0.5` | 5.405949 | 5.754602 | N/A | 34660610688 -> 31386923648 | 9.44% | 2.80 |

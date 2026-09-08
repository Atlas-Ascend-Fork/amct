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

> If the amct_pytorch wheel is not installed in your environment, run straight from source with
> `PYTHONPATH=<amct repo root>`, e.g. `PYTHONPATH=../../.. python3 src/run_dense_samples.py`.

## 3 Community Task Sample: Qwen3.6-35B-A3B MoE Expert Pruning

This section covers issue 183 task 1: `activation_count` + `tolerance`.

### 3.1 Run Instructions

- Calibration set: Pileval, default `1` sample, `seq_len=512`
- Evaluation set: WikiText2 test split, fixed `seq_len=4096`, using the full test split by default
- Search mode: `tolerance`
- Output directory: relative placeholder; the script saves the pruned model, `PruneReport`, and run config
- Pruning defaults to `device_map=auto` across the two visible NPU cards
- `--device-map single` falls back to one card; the single-card input device defaults to `npu:0`
- Qwen3.6-35B-A3B needs the combined memory of two cards
- The datasets are loaded online through `datasets`: Pileval for calibration and the WikiText2 test split

Run commands:

```bash
python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_tolerance.py \
  --model-path ./path/to/Qwen3.6-35B-A3B \
  --save-dir ./outputs/qwen3.6_activation_count_tolerance/t0_1 \
  --tolerance 0.1 \
  --calib-samples 1 \
  --calib-seq-len 512

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_tolerance.py \
  --model-path ./path/to/Qwen3.6-35B-A3B \
  --save-dir ./outputs/qwen3.6_activation_count_tolerance/t0_2 \
  --tolerance 0.2 \
  --calib-samples 1 \
  --calib-seq-len 512

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_tolerance.py \
  --model-path ./path/to/Qwen3.6-35B-A3B \
  --save-dir ./outputs/qwen3.6_activation_count_tolerance/t0_5 \
  --tolerance 0.5 \
  --calib-samples 1 \
  --calib-seq-len 512
```

By default the model is sharded across the two visible NPU cards; to fall back to one card, add
`--device-map single --device npu:0`.
`--preview-prune-ratio` is used only by `prune_diagnose()` for the dry run, and defaults to `0.1`.
`--eval-batches` is only for quick truncated evaluation; omit it to use the full WikiText2 test split.
The search grid uses the built-in `0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8` values; pass `--ratio-grid` to override.

If you want to feed the pruned result into the amct eval / quantization flow, see the VL config note in
[Qwen3.6-Moe-Pruning_en.md](../../models/qwen3.6/Qwen3.6-Moe-Pruning_en.md).

### 3.2 Diagnostics and Report

The script prints, in order:

1. Full runtime arguments
2. The `prune_diagnose()` summary
3. The post-pruning `PruneReport`
4. Baseline / pruned PPL and parameter counts

The measured `tolerance=0.1` output on 2 NPUs with the full WikiText2 test split. The ratio search
runs as a binary search over `ratio_grid`, evaluating candidates 0.4 / 0.2 / 0.1 in order; none met
the tolerance, so the model was left unchanged:

```text
[prune-diagnose]
[prune-diagnose] prunable targets: cnn=0, dense=40, moe=40
  fixed-ratio prune: ineffective (0 cut) (cut 0.0%)
  acc binary search: available (chosen prune_ratio=0.7)
  - fixed-ratio prune dry-run error: OutOfMemoryError: NPU out of memory. Tried to allocate
    1.00 GiB (NPU 0; 61.27 GiB total capacity; ...)
    # note: the diagnose dry-run copies the model and OOMs on 2 cards; only this preview
    # item is affected, the formal pruning search is unaffected

PPL evaluation completed: 6.309916   # search-internal baseline
PPL evaluation completed: 9.857123   # candidate prune_ratio=0.4
PPL evaluation completed: 7.833244   # candidate prune_ratio=0.2
PPL evaluation completed: 6.890347   # candidate prune_ratio=0.1
PPL evaluation completed: 6.309916   # model unchanged, re-check

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

Measured on 2 × Ascend 910: evaluation on the full WikiText2 test split, `seq_len=4096`
(72 batches, one sequence per forward); calibration on 1 Pileval sample with `seq_len=512`.
`tolerance` is the allowed absolute PPL increase: `pruned PPL - baseline PPL <= tolerance`.

| Model | Method | Setting | Baseline PPL | Pruned PPL | Post-process PPL | Params (before -> after) | Reduction | Prune time (min) |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3.6-35B-A3B | `activation_count` | `tolerance=0.1` | 6.309916 | 6.309916 | N/A | 34660610688 -> 34660610688 | 0.00% | 18.63 |
| Qwen3.6-35B-A3B | `activation_count` | `tolerance=0.2` | 6.309916 | 6.309916 | N/A | 34660610688 -> 34660610688 | 0.00% | 17.44 |
| Qwen3.6-35B-A3B | `activation_count` | `tolerance=0.5` | 6.309916 | 6.309916 | N/A | 34660610688 -> 34660610688 | 0.00% | 18.28 |

Candidate ratios evaluated during the searches (binary-search order, reproducible across tiers):

| Candidate prune_ratio | Pruned PPL | ΔPPL | Meets tolerance 0.1 / 0.2 / 0.5 |
| ---: | ---: | ---: | --- |
| 0.1 | 6.890347 | 0.580431 | no / no / no |
| 0.2 | 7.833244 | 1.523328 | no / no / no |
| 0.4 | 9.857123 | 3.547207 | no / no / no |

The smallest default candidate 0.1 already costs ΔPPL 0.58, beyond all three tolerances, so the
model stays unchanged in every tier; pass a finer `--ratio-grid` (e.g. `0.02,0.05,0.1`) to make
pruning take effect.

## 4 Community Sample: Qwen3.6-35B-A3B MoE `mass_variance` + `tolerance` (Task 2)

Corresponds to Issue [#183](https://gitcode.com/cann/amct/issues/183) task 2: structured expert
pruning on a real MoE checkpoint.
Script: [`src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py`](src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py).

### 4.1 How to Run

Environment: CANNLab **NPU A3** (910C, host RAM preferably ≥150GB). Pruning runs on CPU; WikiText2
PPL is measured with AMCT blockwise eval on `npu:0` at `seq_len=4096`.

```bash
# repo root
source /home/developer/Ascend/cann/set_env.sh
export MODEL_DIR=/data/models/Qwen3.6-35B-A3B   # placeholder; point to local weights
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# baseline once
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model "$MODEL_DIR" \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml

# run each tolerance independently from the original model (default ratio_grid)
# Qwen3.6 local weights need an explicit --trust-remote-code (trusted local dir only)
# Replace BASELINE_PPL with your measured baseline; 6.308547 matches this README table
BASELINE_PPL=6.308547
for T in 0.1 0.2 0.5; do
  python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py \
    --model "$MODEL_DIR" \
    --trust-remote-code \
    --tolerance "$T" \
    --baseline-ppl "$BASELINE_PPL" \
    --skip-baseline-eval \
    --search-evaluator fidelity \
    --output-dir ./output/qwen3_6_35b_a3b_mass_variance_tolerance
done
```

Note: acceptance / result-table metrics are always WikiText2 blockwise PPL
(`amct_pytorch.eval`, `seq_len=4096`). `--search-evaluator fidelity` is only for the
tolerance **search** stage (full-seq PPL search is too expensive; short-seq `proxy_ppl`
often rejects every ratio at 0.1/0.2/0.5). Keep `fidelity` to reproduce this table.

Key arguments:

| Arg | Meaning |
|:--|:--|
| `--tolerance` | Search tolerance (under default `fidelity`: max top-1 fidelity / quality drop, **not** a WikiText2 PPL delta); task requires `0.1` / `0.2` / `0.5` separately |
| `--trust-remote-code` | Allow custom Python from the model dir (off by default). `--model` must be a local directory; enable only for trusted local weights |
| `--search-evaluator fidelity` | AMCT built-in fidelity for search (not comparable with `proxy_ppl`); **final eval / result table is always WikiText2 blockwise PPL** |
| `--calib-nsamples` / `--calib-seq-len` | Pileval calibration count/length (default 8 / 512) |
| `--eval-seq-len` | Evaluation sequence length, fixed at 4096 |
| `--output-dir` | Artifact directory (relative / placeholder path) |

Calibration: `mit-han-lab/pile-val-backup`. Evaluation: WikiText2 `wikitext-2-raw-v1` test.
After `save_pretrained`, the script restores VL-nested `config.json` and updates
`text_config.num_experts` so the `qwen3_6_moe` eval path can load the checkpoint.

### 4.2 Diagnosis and Report

Excerpt from a real `tolerance=0.1` / `--search-evaluator fidelity` run (matches `result.json`).

`prune_diagnose()` output:

```text
DiagnosisReport(targets={'cnn': 0, 'dense': 40, 'moe': 40}, prune_works=True,
prune_reduction=0.46498353029826456, prune_forward_ok=True, search_works=True,
search_chosen_ratio=0.1, notes=[])
```

`PruneReport` excerpt (`tolerance=0.1`):

```json
{
  "backend": "huggingface",
  "params_before": 34660610688,
  "params_after": 24965460608,
  "events": [
    {"stage": "cnn", "method": "variance_channel", "detail": "No compatible pruning targets were found"},
    {"stage": "moe", "method": "mass_variance", "detail": "mass-based prune 256 -> 179"}
  ],
  "warnings": []
}
```

`tolerance=0.2` events show `256 -> 128`; `tolerance=0.5` shows `256 -> 51`
(`text_config.num_experts=51`). Full JSON is in each `result.json`.

### 4.3 Results

Measured on CANNLab A3, WikiText2 blockwise BF16, `seq_len=4096`. Search evaluator=`fidelity`.

| Model | Method | Setting | Baseline PPL | Pruned PPL | Post PPL | Params (before → after) | Cut rate | Prune time (min) |
|:--|:--|:--|--:|--:|:--|:--|--:|--:|
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.1` | 6.308547 | 7.974751 | N/A | 34660610688 → 24965460608 | 0.2797 | 7.29 |
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.2` | 6.308547 | 10.896645 | N/A | 34660610688 → 18543997568 | 0.4650 | 7.15 |
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.5` | 6.308547 | 22.681002 | N/A | 34660610688 → 8848922975 | 0.7447 | 7.35 |

Routed experts per layer start at 256; after pruning they are **179 / 128 / 51**.
For `tolerance=0.5`, params/cut-rate are cross-checked from the pruned checkpoint
(`num_experts=51`) against the 0.1/0.2 `params_after` vs expert-count relation; prune time is
diagnose+search+apply on the same script path (excluding post-eval).

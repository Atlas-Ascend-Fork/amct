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

Environment: CANNLab / cloud **NPU A3** (preferably **2 visible NPUs**, host RAM ≥150GB).
Default `--device-map auto` shards the model onto NPUs; **both tolerance search and
acceptance run WikiText2 PPL on NPU** (same as task 1). Use `--device-map cpu` only if
NPU memory is insufficient.

`tolerance` means allowed **absolute PPL increase**: `pruned_ppl - baseline_ppl <= tolerance`.

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

# run each tolerance independently (default WikiText2 PPL search + default ratio_grid)
# Qwen3.6 local weights need an explicit --trust-remote-code (trusted local dir only)
# Replace BASELINE_PPL with your measured baseline
BASELINE_PPL=6.308547
for T in 0.1 0.2 0.5; do
  python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py \
    --model "$MODEL_DIR" \
    --trust-remote-code \
    --tolerance "$T" \
    --baseline-ppl "$BASELINE_PPL" \
    --skip-baseline-eval \
    --search-evaluator wikitext2_ppl \
    --device-map auto \
    --output-dir ./output/qwen3_6_35b_a3b_mass_variance_tolerance
done
```

Note: the smallest default `ratio_grid` candidate is 0.1. If that ΔPPL already exceeds
0.1/0.2/0.5, all three tiers may leave the model unchanged (same class of outcome as task 1;
valid). Pass a finer grid such as `--ratio-grid 0.02,0.05,0.1` if pruning must take effect.

Key arguments:

| Arg | Meaning |
|:--|:--|
| `--tolerance` | Allowed absolute WikiText2 PPL increase; task requires `0.1` / `0.2` / `0.5` separately |
| `--trust-remote-code` | Allow custom Python from the model dir (off by default). `--model` must be a local directory; enable only for trusted local weights |
| `--device-map auto` | Default; shard across visible NPUs (same as task 1). `single` / `cpu` are fallbacks |
| `--search-evaluator wikitext2_ppl` | Default; search metric aligned with acceptance. `fidelity` / `proxy_ppl` are for comparison only — do not mix into the result table |
| `--eval-batches` | Optional cap on WikiText2 batches for search; omit for the full test split |
| `--ratio-grid` | Optional comma-separated prune ratios; default is library `DEFAULT_RATIO_GRID` |
| `--calib-nsamples` / `--calib-seq-len` | Pileval calibration count/length (default 8 / 512) |
| `--eval-seq-len` | Evaluation sequence length, fixed at 4096 |
| `--output-dir` | Artifact directory (relative / placeholder path) |

Calibration: `mit-han-lab/pile-val-backup`. Evaluation: WikiText2 `wikitext-2-raw-v1` test.
After `save_pretrained`, the script restores VL-nested `config.json` and updates
`text_config.num_experts` so the `qwen3_6_moe` eval path can load the checkpoint.

### 4.2 Diagnosis and Report

Measured with `--device-map auto`, `--search-evaluator wikitext2_ppl`, `search_device=npu:0`.

For `tolerance=0.1` / `0.2`, no default-grid candidate meets the tolerance; the model is left
unchanged and `PruneReport` contains a warning such as:

```text
no prune ratio met tolerance 0.100 across 3 candidates; model left unchanged.
```

For `tolerance=0.5`, search selects a feasible ratio and pruning takes effect; see the matching
`result.json` for `params_after` and per-layer `mass_variance` events.

### 4.3 Results

Measured on cloud A3 with `--device-map auto`, WikiText2 PPL, `seq_len=4096`,
`--search-evaluator wikitext2_ppl`.

| Model | Method | Setting | Baseline PPL | Pruned PPL | Post PPL | Params (before → after) | Cut rate | Prune time (min) |
|:--|:--|:--|--:|--:|:--|:--|--:|--:|
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.1` | 6.308547 | 6.308547 | N/A | 34660610688 → 34660610688 | 0.0000 | 17.95 |
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.2` | 6.308547 | 6.308547 | N/A | 34660610688 → 34660610688 | 0.0000 | 17.12 |
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.5` | 6.308547 | 6.586955 | N/A | 34660610688 → 31386923648 | 0.0945 | 17.99 |

Notes: `0.1` / `0.2` find no acceptable candidate on the default grid (same class of valid
outcome as task 1). At `0.5`, ΔPPL≈0.278 ≤ 0.5 so pruning applies (~9.45% cut). Values come
from each tier's `result.json`.

## 5 Community Task 17: Qwen3-0.6B Dense `reconstruct` + `finetune` (ratio)

Corresponds to Issue [#183](https://gitcode.com/cann/amct/issues/183) task 17: start from the
unpruned Qwen3-0.6B checkpoint, apply Dense FFN intermediate-dim `reconstruct` pruning at a
fixed `prune_ratio=0.5`, then recover with `prune_finetune()`.
Script: [`src/run_qwen3_0_6b_pruning_reconstruct_ratio_finetune.py`](src/run_qwen3_0_6b_pruning_reconstruct_ratio_finetune.py).

### 5.1 Run Instructions

- Calibration: Pileval (`mit-han-lab/pile-val-backup`), default `8` samples, `seq_len=512`
- Evaluation: WikiText2 `wikitext-2-raw-v1` test split, fixed `seq_len=4096`, full split by default
- Mode: `ratio`. Only `methods.dense.name=reconstruct` and `kwargs.prune_ratio=0.5`; `recovery` uses the library default `ls`
- Do **not** pass `tolerance` / `size_budget` / `evaluator` to `amct.prune()` (ratio mode errors if they are passed)
- Recovery train: `prune_finetune(model, data=pileval_batches, steps=200, lr=1e-4)` on the same Pileval calib batches
- Single card: `npu:0`. Outputs go to the relative placeholder `./outputs/...`; do not commit weights
- The script prints the full runtime arguments and writes diagnose / `PruneReport` / metrics under `save_dir`

```bash
# from examples/algorithms/pruning/
source /home/developer/Ascend/cann/set_env.sh
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

python3 src/run_qwen3_0_6b_pruning_reconstruct_ratio_finetune.py \
  --model-path /path/to/Qwen3-0.6B \
  --save-dir ./outputs/qwen3_0_6b_reconstruct_ratio_finetune \
  --device npu:0 \
  --prune-ratio 0.5 \
  --calib-samples 8 \
  --calib-seq-len 512 \
  --seq-len 4096 \
  --finetune-steps 200 \
  --finetune-lr 1e-4
```

Key arguments:

| Arg | Meaning |
|:--|:--|
| `--model-path` | Local Qwen3-0.6B weight directory (placeholder) |
| `--prune-ratio` | Dense-domain removal ratio; this task has only the `0.5` tier |
| `--calib-samples` / `--calib-seq-len` | Pileval calibration count/length (default 8 / 512) |
| `--seq-len` | WikiText2 evaluation length, fixed at 4096 |
| `--finetune-steps` / `--finetune-lr` | `prune_finetune` steps and learning rate (default 200 / 1e-4) |
| `--device` | Single-card device, default `npu:0` |
| `--save-dir` | Relative output directory for reports and optional weights |
| `--eval-batches` | Optional eval-batch cap; omit for the full test split |

Prune time: from the start of calibration collection through `amct.prune()` (excludes finetune). Finetune time is recorded separately.

### 5.2 Diagnostics and Report

The script prints the full runtime arguments, the `prune_diagnose()` summary, `PruneReport.as_dict()` after pruning, and a short finetune-loss summary. The text below is from the local run; originals are in `./outputs/qwen3_0_6b_reconstruct_ratio_finetune/diagnose_summary.txt` and `prune_report.json`.

`prune_diagnose()` dry-runs a model copy: fixed-ratio prune is available (about 22.2% cut), but the copy lands on CPU while calib batches stay on `npu:0`, so the forward check and acc search fail with a device mismatch. Formal `amct.prune()` is unaffected (the original model stays on `npu:0`).

```text
[prune-diagnose] prunable targets: cnn=0, dense=28, moe=0
  fixed-ratio prune: available (cut 22.2%, forward failed)
  acc binary search: unavailable
  - acc search dry-run error: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, npu:0 and cpu! (when checking argument for argument indices in method wrapper_NPU__embedding)
```

`PruneReport.as_dict()` summary: `backend=huggingface`, `params_before=596049920`, `params_after=463929344`, all 28 `model.layers.*.mlp.down_proj` layers use `reconstruct` + `ls` (`3072 -> 1536`), `per_layer_sparsity=0.5`, `warnings=[]`. Full dict: local `prune_report.json`.

```json
{
  "backend": "huggingface",
  "params_before": 596049920,
  "params_after": 463929344,
  "budget_unreachable": false,
  "prunable_fraction": null,
  "warnings": [],
  "allocation_choice": null,
  "events": "28x dense/reconstruct ls 3072->1536 on model.layers.{0..27}.mlp.down_proj; cnn/moe: no targets",
  "per_layer_sparsity": "model.layers.{0..27}.mlp.down_proj = 0.5"
}
```

### 5.3 Result Table

Measured on a single Ascend 910 (`npu:0`), WikiText2 test, `seq_len=4096` (73 batches). Calibration: 8 Pileval samples, `seq_len=512`. Finetune: `steps=200`, `lr=1e-4` on those 8 Pileval batches.

| Model | Method | Setting | Baseline PPL | Pruned PPL | Post-process PPL | Params (before -> after) | Reduction | Prune time (min) |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3-0.6B | `reconstruct` + `finetune` | `prune_ratio=0.5` | 19.153452 | 78.375824 | 1863.838989 | 596049920 -> 463929344 | 22.17% | 0.51 |

Finetune time: 0.46 min (not included in prune time). Train loss fell from 3.190 to 0.020 while eval PPL rose, i.e. 200 steps on 8 short sequences overfit the calib set. Numbers are measured; hyperparameters match the task-book defaults.

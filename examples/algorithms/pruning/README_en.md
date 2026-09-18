# AMCT Structured Pruning Samples

Calling the `amct_pytorch.pruning` interface across three domains: dense FFN (intermediate dim) /
CNN (channels) / MoE (experts). The directory contains both tiny random-model demos and the
Qwen3.6-35B-A3B community-task sample.

> API details: [`amct_pytorch/pruning/README_en.md`](../../../amct_pytorch/pruning/README_en.md).

## 1 Pruning Prerequisites

### 1.1 Install Dependencies

Dependencies are in [requirements.txt](requirements.txt): `torch` and `transformers` (pulled in by the
amct_pytorch import chain). To run on NPU you also need a `TorchNPU` matching your Python/torch versions
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

## 5 Community Task Sample: `mass_variance` + Quantization (Task 18)

Task 18 runs Qwen3.6's public `amct_pytorch.eval` blockwise workflow for W8A8
`attn-linear + moe` quantization after MoE expert pruning with `tolerance=0.1`. The script is
[`src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance_quant.py`](src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance_quant.py).
Weights and datasets are not committed; paths below are relative placeholders.

### 5.1 Run Instructions

```bash
python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance_quant.py \
  --model ./path/to/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --device-map auto \
  --tolerance 0.1 \
  --ratio-grid 0.05 \
  --bit-config amct_pytorch/configs/w8a8.yaml \
  --quant-target attn-linear moe --quant-dtype int \
  --output-dir ./output/qwen3_6_35b_a3b_mass_variance_quant
```

The script collects Pileval calibration data and runs the `mass_variance` tolerance search from the
original model, then evaluates quantization on the pruned model using the WikiText2 `test` split at
`seq_len=4096`. If no default-grid candidate meets the tolerance, pruning legitimately leaves the model
unchanged and quantization is still evaluated on the original checkpoint. Commands, the pruning log tail,
quantized PPL, and the return code are written to `result.json`. `--skip-quant-eval` is for debugging only.

### 5.2 Diagnosis and Report

The full arguments, pruning command, and quantization command are printed at startup. The pruning
`prune_diagnose()` output and `PruneReport` are retained in the nested task output `result.json`; the
quantization stage emits `PPL evaluation completed: ...`, captured as `quant_ppl`. Qwen3.6's fused MoE
requires this dedicated workflow, so the script intentionally uses `amct_pytorch.eval` instead of the
classic `amct.quantize` API.

### 5.3 Results

| Model | Method | Setting | Baseline PPL | Pruned PPL | Post PPL | Params (before → after) | Cut rate | Prune time (min) |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3.6-35B-A3B | `mass_variance` + `quant` | `tolerance=0.1`, `ratio_grid=0.05`, W8A8 `attn-linear + moe` | 6.308547 | 6.354577 | 6.449746 | 34660610688 → 33023767168 | 0.0472 | 10.39 |

Results are from an A3 dual-NPU run. The pruning report and quantization log are retained in the
output directory's `result.json`.

## 6 Community Task Example: Qwen3.6-35B-A3B MoE `activation_count` + `budget` (Task 3)

This section corresponds to Task 3 in issue 183: structured MoE expert pruning based on `activation_count`, using three target parameter budgets: `size_budget=0.9`, `0.8`, and `0.5`.

### 6.1 Task Objective and Experimental Setup

Qwen3.6-35B-A3B contains 40 MoE layers, with 256 routed experts per layer and 8 experts activated for each token. The original BF16 weights occupy approximately 64.56 GiB. Because the complete model may exceed the available memory of a single Ascend 910 card, the pruning stage loads the complete model on the CPU. The pruned model can then be evaluated on WikiText2, quantized, or deployed.

```text
Model:               Qwen3.6-35B-A3B
Method:              activation_count
Target:              MoE experts
Mode:                size_budget
size_budget:         0.9, 0.8, 0.5
Calibration set:     Pileval (mit-han-lab/pile-val-backup, validation)
Evaluation set:      WikiText2 (wikitext-2-raw-v1, test split)
Sequence length:     4096
Calibration samples: 4
Post-processing:     300-step recovery fine-tuning for size_budget=0.5
```

`activation_count` counts expert activations using the calibration data, preferentially removes experts with fewer activations, and shrinks the router and expert weights accordingly.

### 6.2 Model Preparation

Download the original [Qwen/Qwen3.6-35B-A3B weights](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) to a local model directory, for example:

```text
./path/to/Qwen3.6-35B-A3B
```

The weights are in `bfloat16` format and do not require conversion. The directory should contain:

```text
config.json
model.safetensors.index.json
model-00001-of-00026.safetensors
...
model-00026-of-00026.safetensors
tokenizer.json
tokenizer_config.json
```

The calibration and evaluation datasets are downloaded and cached automatically on the first run.

### 6.3 Run Pruning

Run the following commands from `amct/examples/algorithms/pruning`. Each budget must be evaluated independently from the same original model directory:

```bash
python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path ./path/to/Qwen3.6-35B-A3B \
  --size_budget 0.9 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --report_dir ./outputs/qwen3_6_pruning_results \
  --save_model_dir ./outputs/Qwen3.6-35B-A3B-pruned-budget09 \
  --trust_remote_code

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path ./path/to/Qwen3.6-35B-A3B \
  --size_budget 0.8 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.2,0.3,0.4 \
  --report_dir ./outputs/qwen3_6_pruning_results \
  --save_model_dir ./outputs/Qwen3.6-35B-A3B-pruned-budget08 \
  --trust_remote_code

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path ./path/to/Qwen3.6-35B-A3B \
  --size_budget 0.5 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.5,0.6,0.7 \
  --report_dir ./outputs/qwen3_6_pruning_results \
  --save_model_dir ./outputs/Qwen3.6-35B-A3B-pruned-budget05 \
  --trust_remote_code
```

The script loads the complete model on the CPU and performs structured pruning, then runs one forward-pass check on the pruned model. The pruned config, tokenizer, and safetensors weights are written to the output directory only when `--save_model_dir` is provided. If the argument is omitted, only the JSON report remains after the process exits.
Parameter Description:
| Argument | Default | Required | Description |
| --- | --- | --- | --- |
| `--model_path` | none | Yes | Local model directory |
| `--size_budget` | none | Yes | Target parameter budget; one of 0.9 / 0.8 / 0.5 |
| `--seq_len` | 4096 | No | Calibration sequence length |
| `--calibration_samples` | 4 | Number of calibration samples |
| `--pileval_source` | `huggingface` | No | Pileval data source: `huggingface` or `modelscope` |
| `--ratio_grid` | `0.1,0.2,0.3` | No | Candidate expert pruning ratios tried by AMCT, comma-separated, each between 0 and 1 |
| `--report_dir` | none | Yes | Directory for the JSON pruning report |
| `--save_model_dir` | `""` | No | Directory to save the pruned model; if empty, weights are not saved |
| `--trust_remote_code` | `False` | No | Allows execution of custom code within the model directory; disabled by default. --model_path must be a local directory; enable only for trusted local weights. |


### 6.4 Diagnosis and Reports

The pruning script calls `amct_pytorch.pruning.prune_diagnose`.For example, when size_budget=0.9, the diagnostic results are as follows:

```text
{
  "targets": {
    "cnn": 0,
    "dense": 40,
    "moe": 40
  },
  "prune_works": true,
  "prune_reduction": 0.09444977959183498,
  "prune_forward_ok": true,
  "search_works": true,
  "search_chosen_ratio": 0.5,
  "notes": []
}
```

The key fields in the `PruneReport` produced by the formal pruning runs are:

```text
params_before: 34660610688
params_after: 28239147648
budget_unreachable: False
prunable_fraction: 0.9335997444269842
```

The full content is saved in the JSON report output by the script.


### 6.5 WikiText2 Evaluation and Results

PPL evaluation uses the `wikitext-2-raw-v1` test split with a fixed `seq_len=4096`. Pileval is used only for calibration and must not be used for PPL evaluation. When the original or pruned model cannot fit completely on a single card, use the AMCT blockwise evaluation entry point:

```bash
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model ./path/to/Qwen3.6-35B-A3B \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml
```

To evaluate a pruned model, replace only the `--model` path, for example:

```bash
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model ./outputs/Qwen3.6-35B-A3B-pruned-budget09 \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml
```

| Model           | Method             | Setting           |      Baseline PPL |         Pruned PPL |  Post-process PPL | Parameters (before -> after)     | Reduction | Pruning Time (min) |
| --------------- | ------------------ | ----------------- | ----------------: | -----------------: | ----------------: | -------------------------------- | --------: | -----------------: |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.9` | 6.308547019958496 |  6.664599895477295 |               N/A | 34,660,610,688 -> 28,239,147,648 |  18.5267% |              20.11 |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.8` | 6.308547019958496 | 7.1991167068481445 |               N/A | 34,660,610,688 -> 24,965,460,608 |  27.9717% |              15.44 |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.5` | 6.308547019958496 |  9.780414581298828 | 8.294227600097656 | 34,660,610,688 -> 15,270,310,528 |  55.9433% |              15.07 |

### 6.6 Recovery Fine-tuning for `size_budget=0.5`

Recovery fine-tuning was performed only for the smallest `size_budget=0.5` model, which supports full-parameter training on a single 64 GiB Ascend 910. Recovery starts from the pruned model, does not run pruning again, and does not overwrite the pruned model.

```text
Input model:       ./outputs/Qwen3.6-35B-A3B-pruned-budget05
Output model:      ./outputs/Qwen3.6-35B-A3B-pruned-budget05-finetuned
Training set:      WikiText2 wikitext-2-raw-v1 train split
Training batches:  300 distinct contiguous token blocks
Batch size:        1
seq_len:           512
Training tokens:   153,600
Training steps:    300
Optimizer:         SGD
Learning rate:     1e-2
Momentum:          0.0
Warmup:            20 steps
Gradient clipping: 1.0
Checkpointing:     enabled, use_reentrant=False
Device:            npu:0
Training time:     27.371 min
Peak allocated:    58.219 GiB
Peak reserved:     59.793 GiB
```

Momentum-free SGD was selected to avoid the first- and second-moment optimizer states maintained by AdamW. Because SGD does not adaptively scale updates using gradient statistics, this run used `lr=1e-2` instead of directly reusing a typical AdamW learning rate such as `1e-5` or `2e-5`.

The default causal language modeling loss in `prune_finetune()` accepts a dictionary containing `input_ids` and also uses `input_ids` as the labels. This run explicitly passes momentum-free SGD:

```python
import os
import torch
from amct_pytorch.pruning import prune_finetune

model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.to("npu:0")

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-2, momentum=0.0)
result = prune_finetune(
    model,
    batches,
    steps=300,
    lr=1e-2,
    warmup=20,
    optimizer=optimizer,
    device="npu:0",
    grad_clip=1.0,
    log_every=25,
)

model.to("cpu")
model.save_pretrained(
    os.environ["RECOVERY_OUTPUT_DIR"],
    safe_serialization=True,
)
```

At runtime, the incompatible optional `torchaudio` package must also be masked before importing Transformers, as done by the pruning script. Save the tokenizer and preserve the outer Qwen3.6 wrapper configuration from the pruned model's `config.json`. After recovery training, evaluate the output directory using the same WikiText2 test, `seq_len=4096`, and blockwise BF16 command described in this section.

The results of this recovery run are:

```text
Original-model PPL: 6.308547019958496
Pruned PPL:         9.780414581298828
Recovered PPL:      8.294227600097656
```

Compared with the unrecovered pruned model, PPL decreased by `1.486186981201172`, or approximately `15.20%`. If the PPL increase caused by pruning is treated as the recovery target, this run recovered approximately `42.81%`. The recovered PPL remains `1.985680580139160` above the original-model baseline, so the result demonstrates clear but incomplete recovery.

### 6.7 Parameters for Further Recovery

To improve recovery further, change only a small number of parameters in each experiment and compare PPL using the same WikiText2 test command:

1. `steps` and training-data volume: increase the number of distinct training batches and training steps. If `steps` exceeds the batch count, `prune_finetune()` cycles through the batches; increasing the number of distinct training tokens should be preferred.
2. `lr`: `1e-2` is the starting point for momentum-free SGD in this run. Values such as `3e-3` and `1e-2` can be compared; an excessive learning rate may cause the loss to diverge.
3. `warmup`: increase warmup when increasing the number of training steps, while maintaining a reasonable proportion. This run used `20/300`.
4. `seq_len`: longer contexts more closely match the `seq_len=4096` evaluation but increase activation memory. Peak reserved memory already reached `59.793 GiB`; run new one-step and ten-step memory tests before increasing it.
5. Training data: increase its volume and diversity, but never use the WikiText2 test split for training.
6. `grad_clip`: this run used `1.0`. Reduce it if gradients or loss become unstable, while continuing to use PPL on a fixed evaluation set as the final criterion.

Use separate reports and output directories for different recovery configurations to avoid overwriting the pruned model. The BF16 `size_budget=0.5` weights occupy approximately 28.4 GiB. When disk space is limited, record the PPL and report before removing recovery models that are no longer needed.

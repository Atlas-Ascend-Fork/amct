# Qwen3.6-35B-A3B MoE Structured Pruning

Qwen3.6-35B-A3B contains 40 MoE layers, 256 routed experts per layer, and activates 8 experts per token. The original BF16 weights occupy approximately 64.56 GiB. Because the complete model may exceed the available memory of a single Ascend 910 card, the pruning stage loads the complete model on the CPU. The pruned model can then be evaluated on WikiText2, quantized, or deployed.

This pruning experiment uses:

```text
method=activation_count
target=moe
mode=size_budget
size_budget=0.9, 0.8, 0.5
calibration=Pileval (mit-han-lab/pile-val-backup, validation)
evaluation=WikiText2 (wikitext-2-raw-v1, test)
seq_len=4096
```

See:

- [AMCT structured pruning API](../../../amct_pytorch/pruning/README.md)
- [Chinese API documentation](../../../docs/zh/api/README.md)

## 1. Requirements

### 1.1 Hardware and Software

- Ascend 910/A3 NPU;
- Python 3.11;
- CPU-only PyTorch;
- A `torch_npu` version compatible with PyTorch;
- CANN Toolkit and Ops;
- At least approximately 70 GiB of disk space for model weights;
- At least 150 GiB of host memory is recommended.

See [requirements.txt](requirements.txt) for the dependencies.

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install `torch`, `torch_npu`, and CANN separately according to the NPU environment.

### 1.2 Verified Environment

This example was verified with:

```text
Python:       3.11.4
torch:        2.7.1+cpu
torch_npu:    2.7.1.post4
transformers: 5.12.1
datasets:     4.8.4
```

### 1.3 torchaudio Compatibility

If the pre-installed `torchaudio` package is incompatible with the installed `torch` version, importing a model through Transformers may fail with a dynamic-library error. The formal Qwen script masks this unrelated optional dependency inside the process; audio functionality is not required.

## 2. Directory Layout

```text
examples/algorithms/pruning/
|-- README.md
|-- README_en.md
|-- requirements.txt
`-- src/
    |-- run_qwen3_6_35b_a3b_pruning_activation_count_budget.py
    `-- utils.py
```

## 3. Qwen3.6-35B-A3B Example

### 3.1 Task

This example performs structured MoE expert pruning for Qwen3.6-35B-A3B:

```text
Model:               Qwen3.6-35B-A3B
Method:              activation_count
Target:              MoE experts
Mode:                size_budget
Calibration set:     Pileval (official dataset: `mit-han-lab/pile-val-backup`)
Evaluation set:      WikiText2 (`wikitext-2-raw-v1` test split)
Sequence length:     4096
Calibration samples: 4
Post-processing:     300-step recovery fine-tuning for `size_budget=0.5`
```

`activation_count` uses calibration data to count expert activations, preferentially removes experts with fewer activations, and shrinks the router and expert weights accordingly.

### 3.2 Model Preparation

Download the original [Qwen/Qwen3.6-35B-A3B weights](https://link.gitcode.com/?target=https%3A%2F%2Fhuggingface.co%2FQwen%2FQwen3.6-35B-A3B&from=https%3A%2F%2Fgitcode.com%2Fcann%2Famct%2Fblob%2Fmaster%2Fexamples%2Fmodels%2Fqwen3.6%2FQwen3.6-Moe-Pruning.md&lang=zh&theme=white) to a fixed path, for example `/path/to/Qwen3.6-35B-A3B`. The weights are in `bfloat16` format and do not need conversion.

The directory should contain:

```text
config.json
model.safetensors.index.json
model-00001-of-00026.safetensors
...
model-00026-of-00026.safetensors
tokenizer.json
tokenizer_config.json
```

The first run downloads and caches the calibration and evaluation datasets.

## Run Instructions

Formal script path:

```text
src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py
```

Before running the commands, set the following paths:

```bash
export AMCT_ROOT=/path/to/amct
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B
export RESULT_DIR=/path/to/qwen3_6_pruning_results
export BUDGET09_DIR=/path/to/Qwen3.6-35B-A3B-pruned-budget09
export BUDGET08_DIR=/path/to/Qwen3.6-35B-A3B-pruned-budget08
export BUDGET05_DIR=/path/to/Qwen3.6-35B-A3B-pruned-budget05
export RECOVERY_OUTPUT_DIR=/path/to/Qwen3.6-35B-A3B-pruned-budget05-finetuned
export RECOVERY_REPORT=/path/to/recovery_train_budget05_lr1e-2.json
```


Run `size_budget=0.9`:

```bash
cd $AMCT_ROOT

PYTHONPATH=$AMCT_ROOT \
python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path "$MODEL_DIR" \
  --size_budget 0.9 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --report_dir "$RESULT_DIR" \
  --save_model_dir $BUDGET09_DIR
```

Run `size_budget=0.8`:

```bash
PYTHONPATH=$AMCT_ROOT \
python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path "$MODEL_DIR" \
  --size_budget 0.8 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.2,0.3,0.4 \
  --report_dir "$RESULT_DIR" \
  --save_model_dir $BUDGET08_DIR
```

Run `size_budget=0.5`:

```bash
PYTHONPATH=$AMCT_ROOT \
python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path "$MODEL_DIR" \
  --size_budget 0.5 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.5,0.6,0.7 \
  --report_dir "$RESULT_DIR" \
  --save_model_dir $BUDGET05_DIR
```

Arguments:

| Argument                | Meaning                                                      |
| ----------------------- | ------------------------------------------------------------ |
| `--model_path`          | Local model directory                                        |
| `--size_budget`         | Target parameter budget                                      |
| `--seq_len`             | Calibration sequence length                                  |
| `--calibration_samples` | Number of calibration samples                                |
| `--pileval_source`      | Official Hugging Face dataset or ModelScope mirror           |
| `--ratio_grid`          | Candidate expert pruning ratios tried by AMCT                |
| `--report_dir`          | JSON report output directory                                 |
| `--save_model_dir`      | Optional directory for the pruned model; weights are not saved if omitted |

The script loads the model and performs structured pruning on the CPU, then runs one forward check on the pruned model. The pruned model remains in the current Python process until `--save_model_dir` is provided. With that argument, the script writes the pruned config, tokenizer, and safetensors weights to the specified directory. Without it, only the JSON report remains after the process exits.

### 3.3 Output Reports

Default report directory:

```text
$RESULT_DIR
```

Output files:

```text
qwen3_6_35b_a3b_activation_count_budget_0.9.json
qwen3_6_35b_a3b_activation_count_budget_0.8.json
qwen3_6_35b_a3b_activation_count_budget_0.5.json
```

Each report contains:

- Parameter counts before and after pruning;
- Actual parameter reduction;
- Expert sparsity for each MoE layer;
- Whether the budget was unreachable;
- The fraction of parameters that can be pruned;
- Pruning events and warnings.

## Diagnosis and Report

The formal script calls:

```python
from amct_pytorch.pruning import prune_diagnose
```

The diagnosis confirms:

```text
cnn targets:   0
dense targets: 40
moe targets:   40
fixed-ratio pruning: available
accuracy search:     available
```

Only the `moe` domain is enabled in this example, so only MoE expert structures are modified.

After pruning, the script checks:

```text
logits shape: (1, 4096, 248320)
logits finite: True
```

This confirms that the pruned model completes a normal forward pass.

Core fields from the formal `PruneReport` are:

```text
size_budget=0.9
params_before=34660610688
params_after=28239147648
experts=256 -> 205
budget_unreachable=False

size_budget=0.8
params_before=34660610688
params_after=24965460608
experts=256 -> 179
budget_unreachable=False

size_budget=0.5
params_before=34660610688
params_after=15270310528
experts=256 -> 102
budget_unreachable=False
```

The complete `events`, `per_layer_sparsity`, and `warnings` are stored in the JSON reports. The diagnosis above uses a trial `prune_ratio=0.1`; formal pruning uses the corresponding `size_budget` and `ratio_grid` for each run.

## Results Table

| Model           | Method             | Setting           |      Baseline PPL |         Pruned PPL |  Post-process PPL |     Parameters (before -> after) | Reduction | Duration (min) |
| --------------- | ------------------ | ----------------- | ----------------: | -----------------: | ----------------: | -------------------------------: | --------: | -------------: |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.9` | 6.308547019958496 |  6.664599895477295 |               N/A | 34,660,610,688 -> 28,239,147,648 |  18.5267% |          28.89 |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.8` | 6.308547019958496 | 7.1991167068481445 |               N/A | 34,660,610,688 -> 24,965,460,608 |  27.9717% |          16.84 |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.5` | 6.308547019958496 |  9.780414581298828 | 8.294227600097656 | 34,660,610,688 -> 15,270,310,528 |  55.9433% |          20.46 |

All formal experiments used:

```text
seq_len=4096
calibration_samples=4
budget_unreachable=False
logits finite=True
```

## WikiText2 Evaluation

The example uses the `wikitext-2-raw-v1` test split with a fixed `seq_len=4096`. Pileval is used only for calibration and must not be used for PPL evaluation.

When the original model does not fit on one card, use AMCT's blockwise evaluation entry point. It addresses NPU memory by loading model blocks one at a time onto the NPU; the model directory remains on local disk. The `--model` argument specifies the directory containing the weight files; it does not require the complete original model to remain in NPU memory.

The baseline model is already available at `$MODEL_DIR`, so it does not need to be saved again by the pruning script. Use the same command for the baseline and every pruned model:

```bash
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model $MODEL_DIR \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml
```

To evaluate a pruned model, pass `--save_model_dir` during pruning because pruning modifies the model in place; the result is lost when the process exits if no output directory is provided. During evaluation, replace only the `--model` path, for example:

```bash
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model $BUDGET09_DIR \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml
```

For `budget08` and `budget05`, replace only the model directory. The same torchaudio compatibility issue must also be handled when running evaluation.

## Recovery Fine-tuning

Recovery fine-tuning was performed only for the smallest `size_budget=0.5` model, which supports full-parameter training on one 64 GiB Ascend 910. Recovery starts from the pruned model and does not run pruning again or overwrite the pruned model:

```text
Input model:       $BUDGET05_DIR
Output model:      $RECOVERY_OUTPUT_DIR
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

Momentum-free SGD was selected to avoid the first- and second-moment optimizer states maintained by AdamW. Because SGD does not adaptively scale updates from gradient statistics, this recovery run used `lr=1e-2` rather than directly reusing an AdamW-scale learning rate such as `1e-5` or `2e-5`.

### Run Recovery Fine-tuning

The default causal-LM loss in `prune_finetune()` accepts a dictionary containing `input_ids` and uses `input_ids` as the labels. This run explicitly passes momentum-free SGD so that the interface does not create its default AdamW optimizer.

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

At runtime, also apply the same incompatible-`torchaudio` stub used by the pruning script before importing Transformers, save the tokenizer, and preserve the outer Qwen3.6 wrapper configuration from the pruned model's `config.json`. Training parameters and results are saved to `$RECOVERY_REPORT`. After recovery, evaluate the output directory with the same WikiText2 test, `seq_len=4096`, blockwise BF16 command described above.

### Current Recovery Level

The WikiText2 test results for this recovery run are:

```text
Original-model PPL: 6.308547019958496
Pruned PPL:         9.780414581298828
Recovered PPL:      8.294227600097656
```

Lower PPL is better. Relative to the unrecovered pruned model, PPL decreased by `1.486186981201172`, or approximately `15.20%`. If the PPL increase caused by pruning is treated as the recovery target, this run recovered approximately `42.81%`:

```text
(9.780414581298828 - 8.294227600097656)
------------------------------------------------ = 42.81%
(9.780414581298828 - 6.308547019958496)
```

The recovered PPL remains `1.985680580139160` above the original-model baseline. This is a clear but incomplete recovery. The first and final training losses come from different batches and cannot independently measure generalization; the WikiText2 test PPL is the final metric.

### Parameters for Additional Recovery

To improve recovery further, change only a small number of parameters per experiment and compare PPL with the same WikiText2 test command:

1. `steps` and training-data volume: increase the number of independent training batches and the number of steps. If `steps` exceeds the batch count, `prune_finetune()` cycles through the batches; excessive repetition may overfit, so increasing the number of distinct training tokens should be preferred.
2. `lr`: `1e-2` is the starting point for momentum-free SGD in this run. Compare a small range around it, such as `3e-3` and `1e-2`; an excessive learning rate may make the loss diverge, while a very small value may barely update the weights.
3. `warmup`: increase warmup when increasing the training steps, while keeping a reasonable fraction of total steps. This run used `20/300`.
4. `seq_len`: longer contexts may better match the `seq_len=4096` evaluation, but substantially increase activation memory. Peak reserved memory already reached `59.793 GiB`, so every increase must be preceded by new one-step and ten-step memory tests on a single 64 GiB card.
5. Training data: increase the volume and diversity, but never train on the WikiText2 test split. The test split is only for final evaluation.
6. `grad_clip`: the current value is `1.0`. Reduce it if the gradients or loss become unstable; choose configurations using a fixed validation metric or final test PPL, not training loss alone.

For multiple recovery configurations, keep a separate training report for each run and do not overwrite the pruned model. The BF16 `budget05` weights occupy approximately 28.4 GiB; when disk space is limited, record the PPL and report before removing an obsolete recovered-model directory.
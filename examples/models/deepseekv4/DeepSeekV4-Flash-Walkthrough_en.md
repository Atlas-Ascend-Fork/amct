# DeepSeek-V4-Flash End-to-End Post-Training Quantization Workflow

This document introduces how to use AMCT to complete end-to-end accuracy evaluation, post-training quantization, and deployment weight export for DeepSeek-V4-Flash.

## Overall Overview

This document completes the following steps:

- Environment preparation and verification.
- Model weight confirmation.
- BF16 model accuracy evaluation as baseline.
- Evaluate accuracy loss under direct conversion quantization.
- Extract calibration dataset based on quantization targets.
- Perform PTQ based on calibration dataset, and save quantization parameters.
- Evaluate quantization model accuracy loss, export quantization parameters.

This sample is constructed and demonstrated through Python code snippets. Each step prints commands through `show_cmd`. If you decide to directly run the current step, you can switch to `run_cmd(..., dry_run=False)`.

## 1. Environment Preparation

This chapter is the environment preparation stage, including environment installation methods and whether dependencies have been successfully installed.

```python
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path("/path/to/AMCT")
REQUIREMENTS_TXT = REPO_ROOT / "requirements.txt"

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")
print(f"Requirements file: {REQUIREMENTS_TXT}")

# Uncomment the next line when the environment still needs dependency installation.
# subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_TXT)], check=True)

import torch
import transformers

print(f"torch version: {torch.__version__}")
print(f"transformers version: {transformers.__version__}")
print("Environment sanity check passed.")
```

## 2. New Model Import and Model Weight Preparation

If you encounter a model that has not yet been adapted in the code repository, you also need to adapt the new model according to a fixed pattern. This chapter is based on DeepSeek-v4-Flash that has been adapted in the code repository for weight preparation and parameter setting.

Since all processes in the current code repository are based on `bfloat16`, if official weights are in FP8 or FP4 format, you need to first convert weights to `bfloat16`. Specifically use the deploy interface and set parameter `granularity = tensor`. For converted bf16 weights, if you want to deploy later, you can also deploy with one click. If you want to continue quantizing to different data formats with different bit widths, you can perform PTQ based on converted bf16 weights.

DeepSeek-v4-Flash official open source uses mixed FP8+FP4 weights, so this sample needs to first convert to `bfloat16` weights.

```python
from pathlib import Path
import shlex
import subprocess

REPO_ROOT = Path("/path/to/AMCT")
MODEL_PATH = Path("/path/to/deepseek-v4-flash")
CALIB_SOURCE_DIR = Path("/path/to/calibration_or_pileval_source")
OUTPUT_ROOT = REPO_ROOT / "outputs" / "deepseek_v4_flash_walkthrough"
EXPORT_DIR = OUTPUT_ROOT / "export"

MODEL_NAME = "deepseek_v4"
DEVICE = "npu:0"
GRANULARITY = "tensor"
SEQ_LEN = 4096
NSAMPLES = 128

ALL_QUANT_TARGETS = ["mlp", "moe", "attn-linear", "attn-cache"]
PTQ_TARGETS = ["moe"]
QUANT_DTYPE = "int"
PTQ_ALGOS = ["lwc"]

BF16_BIT_CONFIG = REPO_ROOT / "amct_pytorch/configs/bf16.yaml"
W8A8_BIT_CONFIG = REPO_ROOT / "amct_pytorch/configs/w4a8.yaml"

MLP_MOE_PARAM_DIR = OUTPUT_ROOT / "ptq_params" / MODEL_NAME / "mlp_moe"

for path in [OUTPUT_ROOT, EXPORT_DIR, MLP_MOE_PARAM_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def pretty_cmd(parts):
    return " \
  ".join(shlex.quote(str(part)) for part in parts)


def show_cmd(parts):
    print(pretty_cmd(parts))


def run_cmd(parts, cwd=REPO_ROOT, dry_run=True):
    print(pretty_cmd(parts))
    if not dry_run:
        subprocess.run([str(part) for part in parts], cwd=str(cwd), check=True)


required_files = [
    MODEL_PATH / "config.json",
    MODEL_PATH / "model.safetensors.index.json",
]

optional_files = [
    MODEL_PATH / "tokenizer.json",
    MODEL_PATH / "tokenizer_config.json",
]

print(f"REPO_ROOT exists: {REPO_ROOT.exists()} -> {REPO_ROOT}")
print(f"MODEL_PATH exists: {MODEL_PATH.exists()} -> {MODEL_PATH}")
print()

for path in required_files:
    print(f"[required] {path}: {path.exists()}")

for path in optional_files:
    print(f"[optional] {path}: {path.exists()}")

NEW_MODEL_PATH = Path("/path/to/deepseek-v4-flash-bf16")
NEW_MODEL_PATH.mkdir(parents=True, exist_ok=True)
baseline_cmd = [
    "python", "amct_pytorch/cli/llm/deploy.py",
    "--model", MODEL_PATH,
    "--model_name", MODEL_NAME,
    "--device", DEVICE,
    "--granularity", GRANULARITY,
    "--eval_mode", "bf16",
    "--output_dir", NEW_MODEL_PATH,
]

show_cmd(baseline_cmd)
```

## 3. Model Accuracy Baseline Test

This chapter is the BF16 accuracy evaluation before quantization, serving as the accuracy baseline for subsequent quantization experiments, used to evaluate quantization accuracy loss.

```python
baseline_dir = OUTPUT_ROOT / "01_baseline_eval"
baseline_dir.mkdir(parents=True, exist_ok=True)

baseline_cmd = [
    "python", "-m", "amct_pytorch.eval",
    "--model", MODEL_PATH,
    "--model_name", MODEL_NAME,
    "--device", DEVICE,
    "--granularity", GRANULARITY,
    "--eval_mode", "bf16",
    "--bit_config", BF16_BIT_CONFIG,
    "--seq_len", str(SEQ_LEN),
    "--output_dir", baseline_dir,
    "--wikitext_final_out", baseline_dir / "wikitext",
]

show_cmd(baseline_cmd)
# run_cmd(baseline_cmd, dry_run=False)
```

## 4. Direct Conversion Quantization Accuracy Loss Evaluation

This section evaluates the accuracy performance of the model under direct conversion quantization without introducing quantization algorithms, providing data reference for subsequent application of quantization algorithms.

```python
direct_quant_dir = OUTPUT_ROOT / "02_direct_quant_eval"
direct_quant_dir.mkdir(parents=True, exist_ok=True)

direct_quant_cmd = [
    "python", "-m", "amct_pytorch.eval",
    "--model", MODEL_PATH,
    "--model_name", MODEL_NAME,
    "--device", DEVICE,
    "--granularity", GRANULARITY,
    "--eval_mode", "quant",
    "--quant_dtype", QUANT_DTYPE,
    "--bit_config", W8A8_BIT_CONFIG,
    "--seq_len", str(SEQ_LEN),
    "--output_dir", direct_quant_dir,
    "--wikitext_final_out", direct_quant_dir / "wikitext",
    "--quant_target", *ALL_QUANT_TARGETS,
]

show_cmd(direct_quant_cmd)
# run_cmd(direct_quant_cmd, dry_run=False)
```

## 5. Typical Calibration Data Preparation

This chapter prepares calibration data for PTQ in AMCT.

`extract_ptq_data.py` only supports a single `quant_target`, so this section will construct instructions for each `quant_target` separately and save extracted inputs in corresponding paths.

```python
calib_output_root = OUTPUT_ROOT / "03_calibration_data"
calib_output_root.mkdir(parents=True, exist_ok=True)

extract_cmds = {}
for target in PTQ_TARGETS:
    target_output_dir = calib_output_root / target.replace("-", "_")
    target_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "amct_pytorch.extract_ptq_data",
        "--model", MODEL_PATH,
        "--model_name", MODEL_NAME,
        "--device", DEVICE,
        "--granularity", GRANULARITY,
        "--quant_target", target,
        "--data_dir", CALIB_SOURCE_DIR,
        "--output_dir", target_output_dir,
        "--nsamples", str(NSAMPLES),
        "--seq_len", str(SEQ_LEN),
    ]
    extract_cmds[target] = cmd

for target, cmd in extract_cmds.items():
    print(f"\n[{target}]")
    show_cmd(cmd)

# for cmd in extract_cmds.values():
#     run_cmd(cmd, dry_run=False)
```

## 6. Post-Training Quantization

This section performs PTQ for each quantization target `quant_target` and saves learned parameters for subsequent accuracy evaluation.

`ptq.py` only supports a single `quant_target`, so this section will construct instructions for each `quant_target` separately and save learned parameters in corresponding paths.

```python
ptq_output_root = OUTPUT_ROOT / "04_ptq_runs"
ptq_output_root.mkdir(parents=True, exist_ok=True)

param_dir_by_target = {
    "mlp": MLP_MOE_PARAM_DIR,
    "moe": MLP_MOE_PARAM_DIR,
    "attn-linear": ATTN_LINEAR_PARAM_DIR,
    "attn-cache": ATTN_CACHE_PARAM_DIR,
}

ptq_cmds = {}
for target in PTQ_TARGETS:
    target_output_dir = ptq_output_root / target.replace("-", "_")
    target_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "amct_pytorch.ptq",
        "--model", MODEL_PATH,
        "--model_name", MODEL_NAME,
        "--device", DEVICE,
        "--granularity", GRANULARITY,
        "--quant_target", target,
        "--quant_dtype", QUANT_DTYPE,
        "--bit_config", W8A8_BIT_CONFIG,
        "--data_dir", calib_output_root / target.replace("-", "_"),
        "--output_dir", target_output_dir,
        "--seq_len", str(SEQ_LEN),
    ]

    if PTQ_ALGOS:
        cmd.extend(["--algos", *PTQ_ALGOS])

    if target in {"mlp", "moe"}:
        cmd.extend(["--moe_mlp_param_dir", MLP_MOE_PARAM_DIR])
    elif target == "attn-linear":
        cmd.extend(["--attn_linear_param_dir", ATTN_LINEAR_PARAM_DIR])
    elif target == "attn-cache":
        cmd.extend(["--attn_cache_param_dir", ATTN_CACHE_PARAM_DIR])

    ptq_cmds[target] = cmd

for target, cmd in ptq_cmds.items():
    print(f"\n[{target}]")
    show_cmd(cmd)

# for cmd in ptq_cmds.values():
#     run_cmd(cmd, dry_run=False)
```

## 7. Direct Conversion Quantization Accuracy Evaluation Based on Post-Training Quantization

After completing PTQ, add quantization algorithms in direct conversion quantization accuracy evaluation, compare with baseline test and direct conversion quantization accuracy without quantization algorithms, and verify quantization algorithm effectiveness.

```python
calibrated_eval_dir = OUTPUT_ROOT / "05_calibrated_quant_eval"
calibrated_eval_dir.mkdir(parents=True, exist_ok=True)

calibrated_eval_cmd = [
    "python", "-m", "amct_pytorch.eval",
    "--model", MODEL_PATH,
    "--model_name", MODEL_NAME,
    "--device", DEVICE,
    "--granularity", GRANULARITY,
    "--eval_mode", "quant",
    "--quant_dtype", QUANT_DTYPE,
    "--bit_config", W8A8_BIT_CONFIG,
    "--seq_len", str(SEQ_LEN),
    "--output_dir", calibrated_eval_dir,
    "--wikitext_final_out", calibrated_eval_dir / "wikitext",
    "--quant_target", *ALL_QUANT_TARGETS,
    "--moe_mlp_param_dir", MLP_MOE_PARAM_DIR,
    "--attn_linear_param_dir", ATTN_LINEAR_PARAM_DIR,
    "--attn_cache_param_dir", ATTN_CACHE_PARAM_DIR,
]

show_cmd(calibrated_eval_cmd)
# run_cmd(calibrated_eval_cmd, dry_run=False)
```

## 8. Quantization Model Export

This section exports deployable quantization weights, updated configuration files, and auxiliary files to an independent output directory for direct use by downstream inference repositories.

```python
export_cmd = [
    "python", "-m", "amct_pytorch.deploy",
    "--model", MODEL_PATH,
    "--model_name", MODEL_NAME,
    "--device", DEVICE,
    "--granularity", GRANULARITY,
    "--quant_target", *ALL_QUANT_TARGETS,
    "--quant_dtype", QUANT_DTYPE,
    "--bit_config", W8A8_BIT_CONFIG,
    "--output_dir", EXPORT_DIR,
    "--moe_mlp_param_dir", MLP_MOE_PARAM_DIR,
    "--attn_linear_param_dir", ATTN_LINEAR_PARAM_DIR,
    "--attn_cache_param_dir", ATTN_CACHE_PARAM_DIR,
]

show_cmd(export_cmd)
# run_cmd(export_cmd, dry_run=False)
```
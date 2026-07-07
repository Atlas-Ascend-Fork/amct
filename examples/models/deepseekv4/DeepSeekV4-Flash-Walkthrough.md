# DeepSeek-V4-Flash 端到端后训练量化流程

本文档介绍如何使用 AMCT 对 DeepSeek-V4-Flash 完成端到端精度评测、后训练量化和部署权重导出。

## 整体概览

本文档完成以下步骤：

- 环境准备与验证。
- 模型权重确认。
- 模型 bf16 精度评测作为基准。
- 评估量化直转下的精度损失。
- 根据量化目标提取校准数据集。
- 基于校准数据集进行 PTQ，同时保存量化系数。
- 评估量化模型精度损失，导出量化系数。

本样例通过 Python 代码片段构建展示，每个步骤通过 `show_cmd` 打印命令。如果确定要直接运行当前步骤，可以切换到 `run_cmd(..., dry_run=False)`。

## 1. 环境准备

本章节为环境准备阶段，包括环境安装方式以及依赖项是否已经成功安装。

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

## 2. 新模型导入以及模型权重准备

如果遇到代码仓中还未适配的模型，还需要对新模型按照固定模式进行适配。本章节基于代码仓已经适配的 DeepSeek-v4-Flash 进行权重准备和参数设置。

由于当前代码仓所有流程都基于 `bfloat16` 进行，如果官方权重为 FP8 或 FP4 格式，需要先将权重转化为 `bfloat16`。具体使用 deploy 接口，并设置传参 `granularity = tensor`。对转换后的 bf16 权重，如果后续想部署，也可以一键部署。如果要继续量化到不同数据格式下的不同 bit 位，可以基于转换后的 bf16 权重进行 PTQ。

DeepSeek-v4-Flash 官方开源使用混合 FP8+FP4 的权重，因此该样例需要先转换为 `bfloat16` 权重。

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

## 3. 模型精度基准测试

该章节为量化前的 bf16 精度评测，作为后续量化实验的精度基准，用于评估量化精度损失。

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

## 4. 直转量化精度损失评估

本节评估直转量化且不引入量化算法时模型的精度表现，为后续应用量化算法提供直转量化的数据参考。

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

## 5. 典型校准数据准备

该章节为 AMCT 中的 PTQ 准备校准数据。

`extract_ptq_data.py` 只支持单个 `quant_target`，因此本节会为每个 `quant_target` 分别构建指令，并将提取到的输入保存在对应路径下。

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

本节为每个量化目标 `quant_target` 进行 PTQ，并保存学习到的参数，以便后续精度评估。

`ptq.py` 只支持单个 `quant_target`，因此本节会为每个 `quant_target` 分别构建指令，并将学习到的参数保存在对应路径下。

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

## 7. 基于 Post-Training Quantization 的直转量化精度评估

完成 PTQ 后，在直转量化精度评估中加入量化算法，与基准测试、无量化算法的直转量化精度进行比对，验证量化算法效果。

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

## 8. 量化模型导出

本节将可部署的量化权重、更新后的配置文件以及辅助文件导出到独立输出目录中，以供下游推理仓库直接使用。

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

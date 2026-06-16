# Introduction

This project provides calling samples for different scenarios. After setting up the environment, you can try to run them according to actual scenarios:

| Sample | Algorithm | Description |
| ------ |------ |------ |
| Quantize model using MIN-MAX algorithm | [Min-Max](algorithms/minmax/README_CN.md) | Simple quantization based on extrema, best for beginners |
| Quantize model using AWQ algorithm | [AWQ](algorithms/awq/README_CN.md) | Activation-aware weight quantization, suitable for large model PTQ |
| Quantize model using GPTQ algorithm | [GPTQ](algorithms/gptq/README_CN.md) | Weight quantization based on second-order information, layer-by-layer optimization |
| Quantize model using SmoothQuant algorithm | [SmoothQuant](algorithms/smoothquant/README_CN.md) | W8A8 quantization that smooths activation distribution |
| Quantize model using Cast direct conversion algorithm | [Cast](algorithms/cast/README_CN.md) | HiFloat8 data direct conversion |
| Quantize model using Quantile algorithm | [Quantile](algorithms/quantile/README_CN.md) | HiFloat8 quantile quantization |
| Quantize model using OFMR algorithm | [OFMR](algorithms/ofmr/README_CN.md) | Output feature Min-Max quantization |
| Quantize model using MXQuant algorithm | [MXQuant](algorithms/mxquant/README_CN.md) | Micro-scaling floating-point quantization (MXFP8/MXFP4) |
| Quantize model using FlatQuant algorithm | [FlatQuant](algorithms/flatquant/README_CN.md) | Quantization that flattens distribution through affine transformation |

## Quick Verification

`run_example.sh` provides one-command end-to-end verification covering the full quantization pipeline (quantize → calibrate → convert → forward) for four algorithms: cast, minmax, smoothquant, and awq. It uses random calibration data — no dataset download required.

**Usage**

```bash
# Default: download Qwen2-0.5B and run all cases
bash examples/run_example.sh

# Use a local model (skip download)
bash examples/run_example.sh --model_name_or_path /data/Qwen2-0.5B

# Use a mirror to speed up download
bash examples/run_example.sh --hf_endpoint https://hf-mirror.com

# Run specific cases only
bash examples/run_example.sh --cases minmax,smoothquant --device npu:0
```

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `--model_name_or_path` | `Qwen/Qwen2-0.5B` | HuggingFace model name or local path; skips download when a local path is given |
| `--model_dir` | `/tmp/amct_verify_model` | Target directory for model download |
| `--device` | `npu:0` | Device to run on |
| `--cases` | `cast,minmax,smoothquant,awq` | Comma-separated list of cases to run |
| `--hf_endpoint` | (none) | HuggingFace mirror endpoint |

**Notes**

- The cast case requires native hifloat8 support or amct_ops. It is automatically marked `SKIP` when the environment does not meet the requirement, without affecting the overall result.
- Prints `[AMCT_EXAMPLE_ALL_PASS]` when all cases pass, or `[AMCT_EXAMPLE_FAILED]` if any case fails.

# Introduction

This project provides calling samples for different scenarios. After setting up the environment, you can try to run them according to actual scenarios:

| Sample | Algorithm | Description |
| ------ |------ |------ |
| Quantize model using MIN-MAX algorithm | [Min-Max](algorithms/minmax/README_en.md) | Simple quantization based on extrema, best for beginners |
| Quantize model using AWQ algorithm | [AWQ](algorithms/awq/README_en.md) | Activation-aware weight quantization, suitable for large model PTQ |
| Quantize model using GPTQ algorithm | [GPTQ](algorithms/gptq/README_en.md) | Weight quantization based on second-order information, layer-by-layer optimization |
| Quantize model using SmoothQuant algorithm | [SmoothQuant](algorithms/smoothquant/README_en.md) | W8A8 quantization that smooths activation distribution |
| Quantize model using Cast direct conversion algorithm | [Cast](algorithms/cast/README_en.md) | HiFloat8 data direct conversion |
| Quantize model using Quantile algorithm | [Quantile](algorithms/quantile/README_en.md) | HiFloat8 quantile quantization |
| Quantize model using OFMR algorithm | [OFMR](algorithms/ofmr/README_en.md) | Output feature Min-Max quantization |
| Quantize model using MXQuant algorithm | [MXQuant](algorithms/mxquant/README_en.md) | Micro-scaling floating-point quantization (MXFP8/MXFP4) |
| Quantize model using FlatQuant algorithm (Experimental) | [FlatQuant](algorithms/flatquant/README_en.md) | Quantization that flattens distribution through affine transformation |

> **Note**: Samples marked as "Experimental" depend on content under `amct_pytorch/experimental/`.
> Build the package with `bash build.sh --torch --experimental` (or `--pkg --experimental`) before use.
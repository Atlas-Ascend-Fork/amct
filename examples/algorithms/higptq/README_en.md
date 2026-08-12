# AMCT Large Model HiGPTQ Quantization

## 1 Algorithm Introduction

HiGPTQ is a GPTQ variant targeting Huawei's self-developed **HiFloat4** (`hifx4`) format. It reuses standard GPTQ's Hessian-inverse-based OBS (Optimal Brain Surgeon) error-compensation framework, but replaces the per-column quantization loop with a **64-element group** loop, matching HiFloat4's 3-level hierarchical block scale (L1 E6M2 over 64 elements, L2/L3 micro-exponents over 8/4 elements). Columns within a group must be quantized jointly and cannot be solved independently the way int/mxfp columns are. HiGPTQ currently only supports **weight-only** quantization; activations always stay in BF16, and any activation bit-width in `bit_config` is ignored.

Unlike [GPTQ](../gptq/README_en.md) and the other samples in this directory, HiGPTQ is built on AMCT's newer **CLI workflow framework** (`amct_pytorch/cli/llm`), driven through command-line entry points such as `python -m amct_pytorch.{extract_ptq_data,ptq,eval}`, with configuration passed as a YAML file via `--bit_config` rather than a `quant_cfg` dict constructed through a Python interface. See [AMCT PyTorch LLM Quantization Tool Guide](../../../docs/zh/AMCT_Pytorch_LLM.md) for the full CLI parameter reference.

## 2 Quantization Prerequisites

### 2.1 Install Dependencies

Follow [Install & Verify](../../../README.md#安装验证) in the repository root to build and install `amct_pytorch` from source.

### 2.2 Model and Dataset Preparation

This sample uses **Qwen3.5-9B** (a dense model, `--model_name qwen3_5`) as an example. Download the model weights yourself and store them locally (e.g. `/path/to/Qwen3.5-9B`).

- Calibration data: `extract_ptq_data` loads **Pileval** online by default; override the local cache path with the `PILEVAL_LOCAL_PATH` environment variable.
- Evaluation dataset: `eval` loads **WikiText2** online by default; override the local cache path with the `WIKITEXT2_LOCAL_PATH` environment variable.

### 2.3 Quantization Configuration

HiGPTQ uses the built-in [`hifloat4.yaml`](../../../amct_pytorch/configs/hifloat4.yaml) as `bit_config`: 4-bit HiFloat4 weights, 16-bit activations, and 4-bit weights for MoE routed/shared experts as well. The corresponding CLI flags are:

```bash
--quant_dtype hifp --algos higptq --bit_config amct_pytorch/configs/hifloat4.yaml
```

`--algos higptq` is special-cased by `LlmPtqWorkflow` to run under the `calibrate` solver (a single closed-form Hessian pass, no gradients or epochs). This is mutually exclusive with gradient-trained algorithms such as `lwc`/`lac`/`autoround`; they cannot be combined in the same run.

## 3 Quantization Example

Qwen3.5-9B is a dense model, so this sample runs the full flow separately for the **attn-linear** (attention linear layers) and **mlp** (FFN) `quant_target`s. `ptq`/`extract_ptq_data` currently only accept a single `quant_target` per run, so the steps below that go through those two entry points each need to run once per target.

### 3.1 Extract PTQ Calibration Data

```bash
python -m amct_pytorch.extract_ptq_data \
    --model /path/to/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/attn-linear \
    --quant_target attn-linear
```

Repeat the same command for `mlp`: swap `--quant_target` to `mlp` and `--data_dir` to `ptq_data/qwen3_5/mlp`.

### 3.2 Run HiGPTQ PTQ (Hessian Calibration)

```bash
python -m amct_pytorch.ptq \
    --model /path/to/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/attn-linear \
    --quant_dtype hifp \
    --algos higptq \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --quant_target attn-linear \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --output_dir ptq_result/
```

Repeat the same command for `mlp`: swap `--quant_target`/`--data_dir` as in 3.1, and use `--moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp` in place of `--attn_linear_param_dir`.

### 3.3 Quantization Accuracy Evaluation

Unlike `extract_ptq_data`/`ptq`, `eval` accepts multiple `--quant_target` values in one run, loading both targets' PTQ params and applying them together:

```bash
python -m amct_pytorch.eval \
    --model /path/to/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --eval_mode quant \
    --quant_target mlp attn-linear \
    --quant_dtype hifp \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp
```

After evaluation, a log file containing the WikiText2 PPL is written under `output_dir/logs/` (default `./outputs/logs/`).

### 3.4 Export a BF16 Deployment Checkpoint (Optional)

To deploy on inference engines without a HiFloat4 kernel (e.g. vLLM), first pack the HiGPTQ-calibrated weights into a real 4-bit HiFloat4 checkpoint with `amct_pytorch.deploy`, then dequantize it back to BF16. The dequantization routine `hif4_unpack` is the exact inverse of the packing routine `hif4_pack`, so the result is numerically identical to a direct fake-quant simulation.

```bash
python -m amct_pytorch.deploy \
    --model /path/to/Qwen3.5-9B \
    --model_name qwen3_5 \
    --granularity block \
    --device npu:0 \
    --quant_target attn-linear mlp \
    --quant_dtype hifp \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp \
    --output_dir ptq_result/deploy/qwen3_5

python examples/algorithms/higptq/src/dequantize_higptq_deploy.py \
    --input_dir ptq_result/deploy/qwen3_5 \
    --output_dir ptq_result/bf16_export/qwen3_5
```

The full 3.1-3.4 commands are also collected in [`src/run_higptq_example.sh`](src/run_higptq_example.sh) (edit `MODEL_PATH` in the script before running it).

## 4 Common Notes

- `ptq` and `extract_ptq_data` each process only one `quant_target` per run; `eval` and `deploy` can load multiple `quant_target`s together.
- HiGPTQ only supports weight quantization; any `<16-bit` activation entry in `bit_config` is ignored (a warning is printed at run time).
- Use `--granularity block`; `--algos higptq` cannot be combined with gradient-trained algorithms such as `lwc`/`lac`/`autoround`.
- For the full parameter reference (`--seq_len`, `--data_dir`, `--attn_linear_param_dir`, etc.), see [AMCT PyTorch LLM Quantization Tool Guide · Parameters](../../../docs/zh/AMCT_Pytorch_LLM.md#3-参数说明).

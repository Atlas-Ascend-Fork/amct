# Qwen3-0.6B W4A4 AutoRound Quantization Example

This example corresponds to Task 8 in Issue [#182](https://gitcode.com/cann/amct/issues/182): quantize Qwen3-0.6B using `w4a4` + `autoround`, and run both **int4/int4 and mxfp4/mxfp4** configurations.

This directory is placed alongside `qwen3.6` and `deepseekv4`. The Qwen3.6 documentation is not modified.

## Instructions

Environment: single NPU device `npu:0`. First, run `source /home/developer/Ascend/cann/set_env.sh`, and install `amct_pytorch` or set `PYTHONPATH` to the repository root directory.

- Model: local Qwen3-0.6B weights, with `--model_name qwen3` and `--trust_remote_code`
- Calibration dataset: Pileval (`mit-han-lab/pile-val-backup`), with the CLI default `nsamples=128`
- Evaluation dataset: WikiText2 `wikitext-2-raw-v1` test split, with `--seq_len 4096` and `--granularity block`
- BF16 and quantized evaluations use the same model weight directory and evaluation configuration
- Quantization covers Attention (`attn-linear`) and MLP. Each format must complete the full workflow: extract → PTQ (Attention) → PTQ (MLP) → quantized evaluation with PTQ parameters
- The scripts print all runtime parameters at startup. Output directories are relative placeholder paths, and no datasets or model weights are committed

```bash
# Repository root directory
source /home/developer/Ascend/cann/set_env.sh

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# Download the Qwen3-0.6B model
hf download Qwen/Qwen3-0.6B --local-dir /workspace/models/Qwen3-0.6B

export MODEL=/path/to/Qwen3-0.6B  # Placeholder: replace with the local model path

cd examples/models/qwen3-0.6B

# 1) BF16 baseline shared by both formats
bash scripts/eval_bf16.sh

# 2) Extract PTQ offline data for Attention and MLP
bash scripts/extract_ptq_data.sh

# 3) int4/int4
QUANT_DTYPE=int bash scripts/ptq_attn.sh
QUANT_DTYPE=int bash scripts/ptq_mlp.sh
QUANT_DTYPE=int bash scripts/eval_quant.sh

# 4) mxfp4/mxfp4
QUANT_DTYPE=mxfp bash scripts/ptq_attn.sh
QUANT_DTYPE=mxfp bash scripts/ptq_mlp.sh
QUANT_DTYPE=mxfp bash scripts/eval_quant.sh
```

Main parameters:

| Parameter / Environment Variable | Description |
|:--|:--|
| `MODEL` | Local model weight directory using a placeholder path |
| `--model_name qwen3` | Dense Qwen3 workflow |
| `--seq_len 4096` | Sequence length used for calibration and evaluation |
| `--granularity block` | Block-level granularity |
| `--quant_dtype int \| mxfp` | Selects either int4/int4 or mxfp4/mxfp4 |
| `--algos autoround` | Quantization algorithm |
| `--bit_config` | `configs/w4a4_int_autoround.yaml` or `configs/w4a4_mxfp_autoround.yaml`, derived from `w4a4.yaml` |
| `--quant_target` | One PTQ target per run: `attn-linear` or `mlp`; quantized evaluation loads both |
| `NSAMPLES` | Number of Pileval calibration samples; default: 128 |
| `END_BLOCK_IDX` | Defaults to 28 layers for Qwen3-0.6B |

## Output Artifacts

Relative placeholder directories, located under this example directory by default:

| Artifact | Path | Size |
|:--|:--|:--|
| Attention PTQ offline data | `./outputs/qwen3_0_6b/ptq_data/attn-linear` | 28.675 GiB |
| MLP PTQ offline data | `./outputs/qwen3_0_6b/ptq_data/mlp` | 28.675 GiB |
| Total offline data (Attention + MLP) | Sum of the two directories above | 57.349 GiB |
| INT Attention PTQ parameters | `./outputs/qwen3_0_6b_w4a8_int/ptq_params/qwen3/attn-linear` | 674 MiB |
| INT MLP PTQ parameters | `./outputs/qwen3_0_6b_w4a8_int/ptq_params/qwen3/mlp` | 1010 MiB |
| MXFP Attention PTQ parameters | `./outputs/qwen3_0_6b_w4a8_mxfp/ptq_params/qwen3/attn-linear` | 715 MiB |
| MXFP MLP PTQ parameters | `./outputs/qwen3_0_6b_w4a8_mxfp/ptq_params/qwen3/mlp` | 1010 MiB |

Quantization duration is measured from the start of the first `ptq` run for each format until all Attention and MLP parameters have been written. Model downloading and environment setup are excluded.

The offline data size is the combined size of the `ptq_data` output directories: 57.349 GiB.

## Results

BF16 and quantized evaluations use the same `MODEL` directory and the same `seq_len=4096` / `granularity=block` configuration.

| Model | Data Type | Algorithm | Quantization Format | BF16 PPL | Quantized PPL | PPL Difference | Quantization Time (min) | Offline Data Size (GiB) |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|
| Qwen3-0.6B | `w4a4` | `autoround` | `int4/int4` | **19.157549** | **934986.312500** | **934,967.154951** | **87.68** | **57.349** |
| Qwen3-0.6B | `w4a4` | `autoround` | `mxfp4/mxfp4` | **19.157549** | **31.795736** | **12.638187** | **88.51** | **57.349** |

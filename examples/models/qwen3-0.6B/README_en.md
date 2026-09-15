# Qwen3-0.6B w4a4 AutoRound Quantization Example

This example addresses Task 8 in Issue [#182](https://gitcode.com/cann/amct/issues/182): quantize Qwen3-0.6B using `w4a4` with `autoround`, running **both int4/int4 and mxfp4/mxfp4**.

This directory is a sibling of `qwen3.6` and `deepseekv4`. It does not modify the Qwen3.6 documentation.

## Usage

Environment: a single NPU, `npu:0`. First run `source /home/developer/Ascend/cann/set_env.sh`, then install `amct_pytorch` or set `PYTHONPATH` to the repository root.

- Model: local Qwen3-0.6B weights, with `--model_name qwen3` and `--trust_remote_code`.
- Calibration dataset: Pileval (`mit-han-lab/pile-val-backup`), with the CLI default of `nsamples=128`.
- Evaluation dataset: the WikiText2 `wikitext-2-raw-v1` test split, with `--seq_len 4096` and `--granularity block`.
- BF16 and quantized evaluations use the same model directory and evaluation settings.
- Quantization covers both Attention (`attn-linear`) and MLP. For each format, complete the workflow: data extraction → Attention PTQ → MLP PTQ → quantized evaluation with the generated PTQ parameters. The extracted data can be shared by both formats.
- Scripts print the complete runtime arguments before execution. Output paths are relative placeholders; do not commit datasets or model weights.

```bash
# Run from the repository root.
source /home/developer/Ascend/cann/set_env.sh

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# Download the Qwen3-0.6B model.
hf download Qwen/Qwen3-0.6B --local-dir /workspace/models/Qwen3-0.6B

# Replace the placeholder with the local model directory.
export MODEL=/path/to/Qwen3-0.6B

cd examples/models/qwen3-0.6B

# 1) Evaluate the BF16 baseline shared by both formats.
bash scripts/eval_bf16.sh

# 2) Extract offline PTQ data for Attention and MLP.
bash scripts/extract_ptq_data.sh

# 3) Run int4/int4 quantization and evaluation.
QUANT_DTYPE=int bash scripts/ptq_attn.sh
QUANT_DTYPE=int bash scripts/ptq_mlp.sh
QUANT_DTYPE=int bash scripts/eval_quant.sh

# 4) Run mxfp4/mxfp4 quantization and evaluation.
QUANT_DTYPE=mxfp bash scripts/ptq_attn.sh
QUANT_DTYPE=mxfp bash scripts/ptq_mlp.sh
QUANT_DTYPE=mxfp bash scripts/eval_quant.sh
```

Main parameters:

| Parameter / Environment Variable | Description |
|:--|:--|
| `MODEL` | Local model directory; replace the placeholder path. |
| `--model_name qwen3` | Uses the dense Qwen3 model implementation. |
| `--seq_len 4096` | Sequence length for calibration and evaluation. |
| `--granularity block` | Block-level granularity. |
| `--quant_dtype int` or `--quant_dtype mxfp` | Selects int4/int4 or mxfp4/mxfp4, respectively, together with the corresponding bit configuration. |
| `--algos autoround` | Quantization algorithm. |
| `--bit_config` | `configs/w4a4_int_autoround.yaml` or `configs/w4a4_mxfp_autoround.yaml`, derived from `w4a4.yaml`. |
| `--quant_target` | One target per PTQ run: `attn-linear` or `mlp`. Quantized evaluation loads parameters for both targets. |
| `NSAMPLES` | Number of Pileval calibration samples; defaults to 128. |

## Output Artifacts

Relative output paths, located under this example directory by default:

| Artifact | Path | Size |
|:--|:--|:--|
| Offline PTQ data for Attention | `./outputs/ptq_data/attn-linear` | 28.675 GiB |
| Offline PTQ data for MLP | `./outputs/ptq_data/mlp` | 28.675 GiB |
| Total offline data (Attention + MLP) | Combined size of the two directories above | 57.349 GiB |
| int PTQ parameters for Attention | `./outputs/int/ptq_params/qwen3/attn-linear` | 674 MiB |
| int PTQ parameters for MLP | `./outputs/int/ptq_params/qwen3/mlp` | 1010 MiB |
| mxfp PTQ parameters for Attention | `./outputs/mxfp/ptq_params/qwen3/attn-linear` | 715 MiB |
| mxfp PTQ parameters for MLP | `./outputs/mxfp/ptq_params/qwen3/mlp` | 1010 MiB |

Quantization time is measured from the start of the first `ptq` run for a given format until all Attention and MLP parameters have been written. It excludes model download and environment setup.

The offline data size is the combined size of the `ptq_data` output directories: 57.349 GiB.

## Results

BF16 and quantized evaluations use the same `MODEL` and the same settings: `seq_len=4096` and `granularity=block`.

PPL difference is calculated as quantized PPL minus BF16 PPL.

| Model | Precision | Algorithm | Quantization Format | BF16 PPL | Quantized PPL | PPL Difference | Quantization Time (min) | Offline Data Size (GiB) |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|
| Qwen3-0.6B | `w4a4` | `autoround` | `int4/int4` | **19.157549** | **934986.312500** | **934,967.154951** | **87.68** | **57.349** |
| Qwen3-0.6B | `w4a4` | `autoround` | `mxfp4/mxfp4` | **19.157549** | **31.795736** | **12.638187** | **88.51** | **57.349** |

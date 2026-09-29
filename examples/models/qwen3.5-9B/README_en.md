# Qwen3.5-9B W4A4 `autoround` Quantization Example

This example corresponds to Task 16 in Issue [#182](https://gitcode.com/cann/amct/issues/182): perform `w4a4` + `autoround` quantization on Qwen3.5-9B, running both **int4/int4 and mxfp4/mxfp4**.

This directory is located alongside `qwen3.6` and `deepseekv4`. No other documentation is modified.

## Running Instructions

Environment: Single card `npu:0`. 

First, configure the environment according to your local CANN installation path, and install `amct_pytorch` (or set `PYTHONPATH` to point to the repository root directory).  

One-stop platform example: `source /home/developer/Ascend/cann/set_env.sh`. If you need to download calibration/evaluation data via mirror, you can set `HF_ENDPOINT` accordingly.

* Model: local Qwen3.5-9B weights, using `--model_name qwen3_5` and `--trust_remote_code`
* Calibration dataset: Pileval (`mit-han-lab/pile-val-backup`, CLI default `nsamples=128`)
* Evaluation dataset: WikiText2 `wikitext-2-raw-v1` test split, using `--seq_len 4096` and `--granularity block`
* BF16 and quantized evaluation use the same model weight directory and the same evaluation configuration
* Quantization covers Attention (`attn-linear`) and MLP (`mlp`). For each format, the complete workflow must be executed: extract → PTQ(attn) → PTQ(mlp) → quantized evaluation with PTQ parameters
* The scripts print the complete runtime parameters at startup. Artifact directories are relative placeholder paths, and generated data/model weights are not committed

```bash
export HF_ENDPOINT=https://hf-mirror.com

# Download the Qwen3.5-9B model
hf download Qwen/Qwen3.5-9B --local-dir /path/to/Qwen3.5-9B # Placeholder: replace with the local model weight path

export MODEL=/path/to/Qwen3.5-9B  # Placeholder: replace with the local model weight path

cd examples/models/qwen3.5-9B

# 1) BF16 baseline
bash scripts/eval_bf16.sh

# 2) Extract PTQ offline data (attn + mlp)
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

| Parameter / Environment Variable | Description                                                                                          |
| :------------------------------- | :--------------------------------------------------------------------------------------------------- |
| `MODEL`                          | Local model weight directory                                                                         |
| `--model_name qwen3_5`           | Qwen3_5 model type                                                                                   |
| `--seq_len 4096`                 | Sequence length used for calibration and evaluation                                                  |
| `--granularity block`            | `Block`-level granularity                                                                            |
| `--quant_dtype int \| mxfp`      | `int4/int4` or `mxfp4/mxfp4`                                                                         |
| `--algos autoround`              | Quantization algorithm                                                                               |
| `--bit_config`                   | `amct_pytorch/configs/w4a4.yaml`                                                                     |
| `--quant_target`                 | One PTQ target per run: `attn-linear` or `mlp`; both are loaded together during quantized evaluation |
| `NSAMPLES`                       | Number of Pileval calibration samples; default is 128                                                |
| `EPOCHS`                         | Number of AutoRound optimization epochs; set to `15` in this example                               |
| `BASE_LR`                        | Base learning rate used by AutoRound optimization; default is `1e-5`                               |
| `--start_block_idx`              | Starting Transformer block index for PTQ; set to `0` in this example                               |
| `--end_block_idx`                | Ending Transformer block index for PTQ; set to `32` in this example                                |

## Output Artifacts

Relative placeholder directories, located under this example directory by default:

| Artifact                        | Path                                            | Size      |
| :------------------------------ | :---------------------------------------------- | :-------- |
| PTQ offline data — Attention    | `./outputs/ptq_data/attn-linear`                | 129 GiB   |
| PTQ offline data — MLP          | `./outputs/ptq_data/mlp`                        | 129 GiB   |
| Total offline data (attn + mlp) | Sum of the two directories above                | 258 GiB   |
| INT PTQ parameters — Attention  | `./outputs/int/ptq_params/qwen3_5/attn-linear`  | 7965 MiB  |
| INT PTQ parameters — MLP        | `./outputs/int/ptq_params/qwen3_5/mlp`          | 18440 MiB |
| MXFP PTQ parameters — Attention | `./outputs/mxfp/ptq_params/qwen3_5/attn-linear` | 8458 MiB  |
| MXFP PTQ parameters — MLP       | `./outputs/mxfp/ptq_params/qwen3_5/mlp`         | 19585 MiB |

Quantization duration is measured from the start of the first `ptq` run for the corresponding format until both the Attention and MLP parameters have been fully written. Model downloading and environment setup are not included.

The offline data size is the combined size of the `ptq_data` output directories: 258 GiB.

## Results

BF16 and quantized evaluation use the same `MODEL`, as well as the same `seq_len=4096` / `granularity=block` configuration.

| Model      | Data Type | Algorithm   | Quantization Format | BF16 PPL     | Quantized PPL    | PPL Difference    | Quantization Time (min) | Offline Data Size (GiB) |
| ---------- | --------- | ----------- | ------------------- | ------------ | ---------------- | ----------------- | ----------------------- | ----------------------- |
| Qwen3.5-9B | `w4a4`    | `autoround` | `int4/int4`         | **8.209169** | **15138.553711** | **15130.344542** | **247.68**              | **258**                 |
| Qwen3.5-9B | `w4a4`    | `autoround` | `mxfp4/mxfp4`       | **8.209169** | **9.210809**     | **1.001640**     | **242.53**             | **258**                 |

**Note:** The current INT4 AutoRound quantization results in an extremely high PPL, mainly due to the large quantization error introduced by 4-bit activation quantization. INT4 is highly sensitive to outliers in activations. When a small number of large activation values determine the quantization `scale`, a large portion of the activation values within the normal range may be compressed into only a few integer levels, or even quantized to 0, resulting in significant information loss.

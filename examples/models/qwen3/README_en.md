# Qwen3-0.6B W4A8 `lwc` + `lac` and Qwen3-0.6B W4A4 `autoround` Quantization Example

This example corresponds to Task 1 and Task 8 in Issue [#182](https://gitcode.com/cann/amct/issues/182):

* Perform `w4a8` + `lwc`/`lac` quantization on Qwen3-0.6B, running both **int4/int8 and mxfp4/mxfp8**.
* Perform `w4a4` + `autoround` quantization on Qwen3-0.6B, running both **int4/int4 and mxfp4/mxfp4**.

This directory is located alongside `qwen3.6` and `deepseekv4`. The Qwen3.6 documentation is not modified.

The CLI entry points are consistent with [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md):

```bash
python3 -m amct_pytorch.eval
python3 -m amct_pytorch.extract_ptq_data
python3 -m amct_pytorch.ptq
```

The `w4a8` quantization configuration directly uses the existing repository configuration:

```text
amct_pytorch/configs/w4a8.yaml
```

The `int` and `mxfp` formats are distinguished by `--quant_dtype`; no additional configuration file needs to be copied.

The `w4a4` quantization configuration directly uses the existing repository configuration:

```text
amct_pytorch/configs/w4a4.yaml
```

The `int` and `mxfp` formats are distinguished by `--quant_dtype`; no additional configuration file needs to be copied.

## Running Instructions

Environment: single NPU card using `npu:0`.

First configure the environment according to the local CANN installation path, and install `amct_pytorch` (or set `PYTHONPATH` to point to the repository root).

Example for the one-stop platform:

```bash
source /home/developer/Ascend/cann/set_env.sh
```

If a mirror is required for downloading calibration/evaluation datasets, configure `HF_ENDPOINT` as needed.

Some container images may encounter `torchaudio` crashes when importing `transformers`. This is a local dependency issue. After fixing the environment, use the official `python3 -m` entry points. This example does not provide an additional environment wrapper script.

* Model: local Qwen3-0.6B weights, using `--model_name qwen3` and `--trust_remote_code`
* Calibration dataset: Pileval (`mit-han-lab/pile-val-backup`, CLI default `nsamples=128`)
* Evaluation dataset: WikiText2 `wikitext-2-raw-v1` test split, using `--seq_len 4096` and `--granularity block`
* BF16 and quantized evaluation use the same model weight directory and the same evaluation configuration
* Quantization covers Attention (`attn-linear`) and MLP. For each format, the complete workflow must be executed: extract → PTQ(attn) → PTQ(mlp) → quantized evaluation with PTQ parameters
* Artifact directories are relative placeholder paths. Model weights and generated data are not committed.

Run the following commands from the repository root:

```bash
source /path/to/cann/set_env.sh
export MODEL=/path/to/Qwen3-0.6B   # Placeholder: replace with local weights

# 1) BF16 baseline (shared by both `lwc`/`lac` and `autoround` quantization)
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml

# 2) Extract PTQ offline data (attn + mlp), shared by both lwc/lac and autoround quantization
python3 -m amct_pytorch.extract_ptq_data \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --quant_target attn-linear \
  --nsamples 128 \
  --data_dir examples/models/qwen3/outputs/qwen3_0_6b/ptq_data/attn-linear

python3 -m amct_pytorch.extract_ptq_data \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --quant_target mlp \
  --nsamples 128 \
  --data_dir examples/models/qwen3/outputs/qwen3_0_6b/ptq_data/mlp

# 3) int4/int8 `lwc`/`lac` quantization
python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/qwen3/outputs/qwen3_0_6b/ptq_data/attn-linear \
  --quant_dtype int \
  --algos lwc lac \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_target attn-linear \
  --start_block_idx 0 \
  --end_block_idx 28 \
  --epochs 15 \
  --base_lr 1e-5 \
  --output_dir examples/models/qwen3/outputs/lwc_lac/qwen3_0_6b_w4a8_int

python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/qwen3/outputs/qwen3_0_6b/ptq_data/mlp \
  --quant_dtype int \
  --algos lwc lac \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_target mlp \
  --start_block_idx 0 \
  --end_block_idx 28 \
  --epochs 15 \
  --base_lr 1e-5 \
  --output_dir examples/models/qwen3/outputs/lwc_lac/qwen3_0_6b_w4a8_int

# 4) int4/int4 `autoround` quantization
python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/qwen3/outputs/qwen3_0_6b/ptq_data/attn-linear \
  --quant_dtype int \
  --algos autoround \
  --bit_config amct_pytorch/configs/w4a4.yaml \
  --quant_target attn-linear \
  --start_block_idx 0 \
  --end_block_idx 28 \
  --epochs 15 \
  --base_lr 1e-5 \
  --output_dir examples/models/qwen3/outputs/autoround/qwen3_0_6b_w4a4_int

python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/qwen3/outputs/qwen3_0_6b/ptq_data/mlp \
  --quant_dtype int \
  --algos autoround \
  --bit_config amct_pytorch/configs/w4a4.yaml \
  --quant_target mlp \
  --start_block_idx 0 \
  --end_block_idx 28 \
  --epochs 15 \
  --base_lr 1e-5 \
  --output_dir examples/models/qwen3/outputs/autoround/qwen3_0_6b_w4a4_int

# 5) Evaluate int4/int8 `lwc`/`lac` quantization
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode quant \
  --quant_target attn-linear mlp \
  --quant_dtype int \
  --algos lwc lac \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --attn_linear_param_dir examples/models/qwen3/outputs/lwc_lac/qwen3_0_6b_w4a8_int/ptq_params/qwen3/attn-linear \
  --moe_mlp_param_dir examples/models/qwen3/outputs/lwc_lac/qwen3_0_6b_w4a8_int/ptq_params/qwen3/mlp

# 6) Evaluate int4/int4 `autoround` quantization
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode quant \
  --quant_target attn-linear mlp \
  --quant_dtype int \
  --algos autoround \
  --bit_config amct_pytorch/configs/w4a4.yaml \
  --attn_linear_param_dir examples/models/qwen3/outputs/autoround/qwen3_0_6b_w4a4_int/ptq_params/qwen3/attn-linear \
  --moe_mlp_param_dir examples/models/qwen3/outputs/autoround/qwen3_0_6b_w4a4_int/ptq_params/qwen3/mlp

# 7) mxfp4/mxfp8: Replace the command --quant_dtype int in steps 3 and 5 with mxfp, and change the --output_dir / parameter directory to outputs/lwc_lac/qwen3_0_6b_w4a8_mxfp. Run each of them separately.
# 8) mxfp4/mxfp4: Replace the command --quant_dtype int in steps 4 and 6 with mxfp, and change the --output_dir / parameter directory to outputs/autoround/qwen3_0_6b_w4a4_mxfp. Run each of them separately.
```

## Qwen3-0.6B W4A8 `lwc` + `lac`

Main parameters:

| Parameter / Environment Variable | Description                                                                                         |
| :------------------------------- | :-------------------------------------------------------------------------------------------------- |
| `MODEL`                          | Local weight directory (placeholder path)                                                           |
| `--model_name qwen3`             | Dense Qwen3 execution path                                                                          |
| `--seq_len 4096`                 | Sequence length used for calibration and evaluation                                                 |
| `--granularity block`            | Block-level granularity                                                                             |
| `--quant_dtype int\|mxfp`        | int4/int8 or mxfp4/mxfp8                                                                            |
| `--algos lwc lac`                | Quantization algorithms                                                                             |
| `--bit_config`                   | Repository configuration `amct_pytorch/configs/w4a8.yaml`                                           |
| `--quant_target`                 | One target per extract/PTQ run: `attn-linear` or `mlp`; both are loaded during quantized evaluation |
| `--nsamples`                     | Number of Pileval calibration samples, default 128                                                  |
| `--end_block_idx`                | Qwen3-0.6B contains 28 blocks by default                                                            |

## Qwen3-0.6B W4A4 `autoround`

Main parameters:

| Parameter / Environment Variable | Description                                                                                         |
| :------------------------------- | :-------------------------------------------------------------------------------------------------- |
| `MODEL`                          | Local weight directory (placeholder path)                                                           |
| `--model_name qwen3`             | Dense Qwen3 execution path                                                                          |
| `--seq_len 4096`                 | Sequence length used for calibration and evaluation                                                 |
| `--granularity block`            | Block-level granularity                                                                             |
| `--quant_dtype int\|mxfp`        | int4/int4 or mxfp4/mxfp4                                                                            |
| `--algos autoround`              | Quantization algorithm                                                                              |
| `--bit_config`                   | Repository configuration `amct_pytorch/configs/w4a4.yaml`                                           |
| `--quant_target`                 | One target per extract/PTQ run: `attn-linear` or `mlp`; both are loaded during quantized evaluation |
| `--nsamples`                     | Number of Pileval calibration samples, default 128                                                  |
| `--end_block_idx`                | Qwen3-0.6B contains 28 blocks by default                                                            |

**Note:** `common.sh` defaults to `w4a8 + lwc/lac`; to run `w4a4` with `autoround`, you need to explicitly override `QUANT_TYPE` and `ALGOS`.

Example:

```bash
export MODEL=/path/to/Qwen3-0.6B
cd examples/models/qwen3
QUANT_DTYPE=int QUANT_TYPE=w4a4 ALGOS=autoround bash scripts/ptq_attn.sh
QUANT_DTYPE=int QUANT_TYPE=w4a4 ALGOS=autoround bash scripts/ptq_mlp.sh
QUANT_DTYPE=int QUANT_TYPE=w4a4 ALGOS=autoround bash scripts/eval_quant.sh
# Switch QUANT_DTYPE to mxfp to test quantization results under the mxfp data type.
```

## Output Artifacts

Qwen3-0.6B W4A8 `lwc` + `lac`

Relative placeholder directories (located under this example directory by default):

| Artifact                        | Path                                                                  | Size       |
| :------------------------------ | :-------------------------------------------------------------------- | :--------- |
| PTQ offline data — Attention    | `./outputs/qwen3_0_6b/ptq_data/attn-linear`                           | 28.002 GiB |
| PTQ offline data — MLP          | `./outputs/qwen3_0_6b/ptq_data/mlp`                                   | 28.002 GiB |
| Total offline data (attn + mlp) | Sum of the two directories above                                      | 56.004 GiB |
| INT PTQ parameters — Attention  | `./outputs/lwc_lac/qwen3_0_6b_w4a8_int/ptq_params/qwen3/attn-linear`  | 1.30 MiB   |
| INT PTQ parameters — MLP        | `./outputs/lwc_lac/qwen3_0_6b_w4a8_int/ptq_params/qwen3/mlp`          | 1.67 MiB   |
| MXFP PTQ parameters — Attention | `./outputs/lwc_lac/qwen3_0_6b_w4a8_mxfp/ptq_params/qwen3/attn-linear` | 42.20 MiB  |
| MXFP PTQ parameters — MLP       | `./outputs/lwc_lac/qwen3_0_6b_w4a8_mxfp/ptq_params/qwen3/mlp`         | 63.14 MiB  |

Qwen3-0.6B W4A4 `autoround`

Relative placeholder directories (located under this example directory by default):

| Artifact                        | Path                                                                    | Size       |
| :------------------------------ | :---------------------------------------------------------------------- | :--------- |
| PTQ offline data — Attention    | `./outputs/qwen3_0_6b/ptq_data/attn-linear`                             | 28.002 GiB |
| PTQ offline data — MLP          | `./outputs/qwen3_0_6b/ptq_data/mlp`                                     | 28.002 GiB |
| Total offline data (attn + mlp) | Sum of the two directories above                                        | 56.004 GiB |
| INT PTQ parameters — Attention  | `./outputs/autoround/qwen3_0_6b_w4a4_int/ptq_params/qwen3/attn-linear`  | 674 MiB    |
| INT PTQ parameters — MLP        | `./outputs/autoround/qwen3_0_6b_w4a4_int/ptq_params/qwen3/mlp`          | 1010 MiB   |
| MXFP PTQ parameters — Attention | `./outputs/autoround/qwen3_0_6b_w4a4_mxfp/ptq_params/qwen3/attn-linear` | 715 MiB    |
| MXFP PTQ parameters — MLP       | `./outputs/autoround/qwen3_0_6b_w4a4_mxfp/ptq_params/qwen3/mlp`         | 1072 MiB   |

Quantization duration is measured from the start of the first `ptq` run for the corresponding format until all Attention and MLP parameters have been written. Model downloading and environment setup are not included.

The offline data size is the total size of the extract output directories, measured in GiB.

## Results

BF16 and quantized evaluation use the same `MODEL`, as well as the same `seq_len=4096` / `granularity=block` configuration.

| Model      | Data Type | Algorithm     | Quantization Format |  BF16 PPL | Quantized PPL | PPL Difference | Quantization Time (min) | Offline Data Size (GiB) |
| ---------- | --------- | ------------- | ------------------- | --------: | ------------: | -------------: | ----------------------: | ----------------------: |
| Qwen3-0.6B | `w4a8`    | `lwc` + `lac` | `int4/int8`         | 19.157549 |     47.486488 |      28.328939 |                   37.35 |                  56.004 |
| Qwen3-0.6B | `w4a8`    | `lwc` + `lac` | `mxfp4/mxfp8`       | 19.157549 |     23.898159 |       4.740610 |                   48.88 |                  56.004 |
| Qwen3-0.6B | `w4a4`    | `autoround`   | `int4/int4`         | 19.157549 |     97.062500 |      77.904951 |                   87.68 |                  56.004 |
| Qwen3-0.6B | `w4a4`    | `autoround`   | `mxfp4/mxfp4`       | 19.157549 |     31.795736 |      12.638187 |                   88.51 |                  56.004 |

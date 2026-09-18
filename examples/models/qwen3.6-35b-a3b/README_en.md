# Qwen3.6-35B-A3B w4a8 `autoround` Quantization Sample

Covers **task 20** of Issue [#182](https://gitcode.com/cann/amct/issues/182): quantize Qwen3.6-35B-A3B with `w4a8` + `autoround`, running **both int4/int8 and mxfp4/mxfp8**.

CLI entries match [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md): `python3 -m amct_pytorch.eval` / `extract_ptq_data` / `ptq`.
The bit-width policy reuses the in-repo [`amct_pytorch/configs/w4a8.yaml`](../../../amct_pytorch/configs/w4a8.yaml); `int` / `mxfp` is selected by the CLI `--quant_dtype`.

| Format | Config file | CLI `--quant_dtype` |
| --- | --- | --- |
| int4/int8 | `amct_pytorch/configs/w4a8.yaml` | `int` |
| mxfp4/mxfp8 | `amct_pytorch/configs/w4a8.yaml` | `mxfp` |

The `w4a8.yaml` MoE policy: routed experts w4/a8, shared expert w8/a8.

**Quantization covers both Attention and MoE**: PTQ for the `attn-linear` and `moe` targets is actually executed by default (`scripts/ptq_mlp.sh` defaults to `RUN_MOE_PTQ=1`), and the quantized evaluation loads PTQ params from both.

## How to run

Environment: start from a single `npu:0` card (MoE PTQ takes days there; shorten it with the per-block split shown in [`examples/ptq_multi_npu.sh`](../../../examples/ptq_multi_npu.sh)). Configure the environment with the local CANN installation path first, then install `amct_pytorch` (or point `PYTHONPATH` at the repository root).
One-stop platform example: `source /home/developer/Ascend/cann/set_env.sh`. Set `HF_ENDPOINT` yourself if the calibration/eval datasets must be fetched through a mirror.

- Model: local Qwen3.6-35B-A3B weights; `--model_name qwen3_6_moe`; `--trust_remote_code`
- Calibration: Pileval (`mit-han-lab/pile-val-backup`), CLI default `nsamples=128`
- Evaluation: WikiText2 `wikitext-2-raw-v1` test; `--seq_len 4096`; `--granularity block`
- BF16 and quantized evaluations share the same weight directory and eval configuration
- Per format: extract → PTQ(attn) → PTQ(moe) → quantized eval with PTQ params loaded
- Output directories are relative placeholders; no data/weights are committed

Run from the repository root:

```bash
source /path/to/cann/set_env.sh
export MODEL=/path/to/Qwen3.6-35B-A3B   # placeholder, point to local weights

# 1) BF16 baseline (shared by both formats)
bash examples/models/qwen3.6-35b-a3b/scripts/eval_bf16.sh

# 2) Extract PTQ offline data (attn + moe, shared by both formats; run once)
bash examples/models/qwen3.6-35b-a3b/scripts/extract_ptq_data.sh

# 3) int4/int8: PTQ(attn) → PTQ(moe) → quantized eval
export QUANT_DTYPE=int
bash examples/models/qwen3.6-35b-a3b/scripts/ptq_attn.sh
bash examples/models/qwen3.6-35b-a3b/scripts/ptq_mlp.sh
bash examples/models/qwen3.6-35b-a3b/scripts/eval_quant.sh

# 4) mxfp4/mxfp8: switch QUANT_DTYPE and repeat PTQ(attn) → PTQ(moe) → quantized eval
export QUANT_DTYPE=mxfp
bash examples/models/qwen3.6-35b-a3b/scripts/ptq_attn.sh
bash examples/models/qwen3.6-35b-a3b/scripts/ptq_mlp.sh
bash examples/models/qwen3.6-35b-a3b/scripts/eval_quant.sh
```

Equivalent `python3 -m ...` commands also work (int shown):

```bash
python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/qwen3.6-35b-a3b/outputs/qwen3_6_35b_a3b/ptq_data/attn-linear \
  --quant_dtype int \
  --algos autoround \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_target attn-linear \
  --start_block_idx 0 \
  --end_block_idx 40 \
  --epochs 10 \
  --base_lr 1e-3 \
  --output_dir examples/models/qwen3.6-35b-a3b/outputs/qwen3_6_35b_a3b_w4a8_int
```

For the MoE target switch `--quant_target` to `moe` and `--data_dir` to `.../ptq_data/moe`, keeping the rest unchanged; the quantized eval passes both `--attn_linear_param_dir` and `--moe_mlp_param_dir` (`--quant_target attn-linear moe`).

Main parameters:

| Parameter / env var | Meaning |
|:--|:--|
| `MODEL` | Local weight directory (placeholder path) |
| `--model_name qwen3_6_moe` | Qwen3.6 MoE pathway |
| `--seq_len 4096` | Sequence length for calibration and evaluation |
| `--granularity block` | Block granularity |
| `--quant_dtype int\|mxfp` | int4/int8 or mxfp4/mxfp8 |
| `--algos autoround` | Quantization algorithm |
| `--bit_config` | `amct_pytorch/configs/w4a8.yaml` |
| `--quant_target` | One target per extract/PTQ run: `attn-linear` or `moe`; the quantized eval loads both |
| `--nsamples` | Pileval calibration samples, default 128 |
| `--epochs` / `--base_lr` | autoround defaults `10` / `1e-3` (tunable) |
| `MOE_NSAMPLES` / `MOE_EPOCHS` / `MOE_BASE_LR` | MoE-side overrides for calibration samples / epochs / learning rate (default same as Attention); expert-wise autoround walks ~`40x256` units, so lower them to bound card time under a tight budget and record the actual values in the result table |
| `--end_block_idx` | Default 40 (exclusive; PTQ truncates beyond the layer count) |
| `RUN_MOE_PTQ` | Default `1` (run MoE PTQ); set `0` only for smoke tests, the result table does not apply |

## Artifacts

Relative placeholder directories (under this sample directory by default):

| Artifact | Path | Measured size (GiB) |
|:--|:--|--:|
| PTQ offline data, Attention | `./outputs/qwen3_6_35b_a3b/ptq_data/attn-linear` | ~81 |
| PTQ offline data, MoE | `./outputs/qwen3_6_35b_a3b/ptq_data/moe` | ~21 |
| int PTQ params, Attention | `./outputs/qwen3_6_35b_a3b_w4a8_int/ptq_params/qwen3_6_moe/attn-linear` | ~5 |
| int PTQ params, MoE | `./outputs/qwen3_6_35b_a3b_w4a8_int/ptq_params/qwen3_6_moe/moe` | ~120 |
| mxfp PTQ params, Attention | `./outputs/qwen3_6_35b_a3b_w4a8_mxfp/ptq_params/qwen3_6_moe/attn-linear` | ~5 |
| mxfp PTQ params, MoE | `./outputs/qwen3_6_35b_a3b_w4a8_mxfp/ptq_params/qwen3_6_moe/moe` | ~128 |

Quantization wall time: measured from the first `ptq` of a format until all attn+moe params are written (model download and environment setup excluded); scripts print start/end timestamps. Offline data size is the combined extract output in GiB.

## Results

BF16 and quantized evaluations share the same `MODEL` and the same `seq_len=4096` / `granularity=block` configuration.
The eval covers `attn-linear` + `moe`, loading autoround PTQ params for both.

Measured settings behind the result table: Attention uses the default protocol
(`nsamples=128` / `epochs=10` / `cali_bsz=4`); MoE is overridden with
`MOE_NSAMPLES=32` / `MOE_EPOCHS=1` / `MOE_CALI_BSZ=8` (single Atlas A3 card, 10240 expert units).
Use the same settings to reproduce the table; quantization time is wall-clock including unit I/O.

| Model | Dtype | Algorithm | Format | BF16 PPL | Quant PPL | PPL delta | Quant time (min) | Offline data (GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | `w4a8` | `autoround` | `int4/int8` | 6.3085 | 6.7924 | +0.4839 | 264.4 | 102 |
| Qwen3.6-35B-A3B | `w4a8` | `autoround` | `mxfp4/mxfp8` | 6.3085 | 6.5046 | +0.1961 | 303.2 | 102 |

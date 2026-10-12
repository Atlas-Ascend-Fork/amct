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

Environment: start from a single `npu:0` card. Under the default protocol (`MOE_NSAMPLES=128` / `MOE_EPOCHS=10`) MoE PTQ takes days per format; the result table in this sample uses `MOE_NSAMPLES=32` / `MOE_EPOCHS=1`, measured on one card at 243.5 min for the full int run and 303.2 min for the full mxfp run (see Results). To run the default protocol or go faster, use the per-block split shown in [`examples/ptq_multi_npu.sh`](../../../examples/ptq_multi_npu.sh). Configure the environment with the local CANN installation path first, then install `amct_pytorch` (or point `PYTHONPATH` at the repository root).
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
# Result-table protocol: MoE is overridden with 32/1/8 (Attention keeps the default 128/10/4).
# Without these three variables MoE extracts 128 samples by default (about 4x the data in the
# table) and the PPL values in the table are not reproducible.
export MOE_NSAMPLES=32 MOE_EPOCHS=1 MOE_CALI_BSZ=8
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

For the MoE target, start from the Attention command and change `--quant_target` to `moe`, `--data_dir` to `.../ptq_data/moe`, and also set `--nsamples` / `--cali_bsz` / `--epochs` to the MoE values (this sample: `32` / `8` / `1`; `--base_lr` stays `1e-3` — these are the script's `MOE_NSAMPLES` / `MOE_CALI_BSZ` / `MOE_EPOCHS`); the quantized eval passes both `--attn_linear_param_dir` and `--moe_mlp_param_dir` (`--quant_target attn-linear moe`). Copying the Attention command and only switching `--quant_target` / `--data_dir` leaves `--nsamples` / `--cali_bsz` at the CLI defaults `128 / 4` while `--epochs` remains the `10` in the command, which does not match the result-table protocol.

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
| `--nsamples` / `--cali_bsz` | Pileval calibration samples (CLI default 128) / calibration batch size (CLI default 4) |
| `--epochs` / `--base_lr` | This sample uses `10` / `1e-3` (CLI defaults are `15` / `1e-5`); tunable |
| `MOE_NSAMPLES` / `MOE_EPOCHS` / `MOE_BASE_LR` / `MOE_CALI_BSZ` | MoE-side overrides for calibration samples / epochs / learning rate / calibration batch size (default same as Attention); expert-wise autoround walks ~`40x256` units, so lower them to bound card time under a tight budget and record the actual values in the result table. Note `MOE_NSAMPLES`, `MOE_EPOCHS` and `MOE_CALI_BSZ` must all be >= 1 (the cosine scheduler's `T_max = epochs × ⌈nsamples / cali_bsz⌉` is legal as long as none of them is zero; the script validates this and errors out)|
| `--end_block_idx` | This sample uses 40 (exclusive; Qwen3.6-35B-A3B has 40 layers, so every layer is covered); CLI default is 61 and is truncated when it exceeds the layer count |
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
| BF16 baseline log (shared by both formats) | `./outputs/qwen3_6_35b_a3b_bf16/logs/eval_qwen3_6_moe_bf16.log` | — |

Quantization wall time: measured from the first `ptq` of a format (`ptq_attn.sh`) until all attn+moe params are written (model download and environment setup excluded); both `ptq_attn.sh` and `ptq_mlp.sh` print `start:` / `end:` timestamps. The int row is a full end-to-end measurement; the mxfp row's duration is from its first full run — afterwards only the moe segment was re-run with the same script (86.7 min) and the quantized PPL refilled, while the attn segment (script and schedule unchanged) was not re-run; re-run the attn segment if a full mxfp wall-time figure is needed. Offline data size is the combined extract output in GiB.

Eval logs: `eval_quant.sh` writes `<OUTPUT_DIR>/logs/eval_*.log`, with int and mxfp under their own `OUTPUT_DIR` so the two formats no longer share one file; `eval_bf16.sh` writes `<BF16_OUTPUT_DIR>/logs/` (default `./outputs/qwen3_6_35b_a3b_bf16/logs/`), independent of `QUANT_DTYPE`, so both formats share the same BF16 baseline log. loguru appends.

## Results

BF16 and quantized evaluations share the same `MODEL` and the same `seq_len=4096` / `granularity=block` configuration.
The eval covers `attn-linear` + `moe`, loading autoround PTQ params for both.

Measured settings behind the result table: Attention uses the default protocol
(`nsamples=128` / `epochs=10` / `cali_bsz=4`); MoE is overridden with
`MOE_NSAMPLES=32` / `MOE_EPOCHS=1` / `MOE_CALI_BSZ=8` (single Atlas A3 card, 10240 expert units).
Use the same settings to reproduce the table; quantization time is wall-clock including unit I/O.

| Model | Dtype | Algorithm | Format | BF16 PPL | Quant PPL | PPL delta | Quant time (min) | Offline data (GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | `w4a8` | `autoround` | `int4/int8` | 6.3085 | 6.8263 | +0.5178 | 243.5 | 102 |
| Qwen3.6-35B-A3B | `w4a8` | `autoround` | `mxfp4/mxfp8` | 6.3085 | 6.4729 | +0.1644 | 303.2 | 102 |

> The delta column is computed from the **full-precision** (unrounded) quantized and BF16 PPL values, so it can differ slightly in the last digit from subtracting the 4-decimal columns shown here.

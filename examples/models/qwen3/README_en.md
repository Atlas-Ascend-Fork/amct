# Qwen3-0.6B w4a8 `lwc` + `lac` Quantization Sample

Corresponds to Issue [#182](https://gitcode.com/cann/amct/issues/182) task 1: quantize Qwen3-0.6B with `w4a8` + `lwc`/`lac` in **both int4/int8 and mxfp4/mxfp8**.
This directory sits next to `qwen3.6` and `deepseekv4`. Do not edit the Qwen3.6 docs.

The CLI matches [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md): `python3 -m amct_pytorch.eval` / `extract_ptq_data` / `ptq`.
Use the in-repo [`amct_pytorch/configs/w4a8.yaml`](../../../amct_pytorch/configs/w4a8.yaml) for both formats; switch int / mxfp with `--quant_dtype`. Do not copy the policy into this sample.

## Run Instructions

Environment: single card `npu:0`. Source the CANN env for your install path and install `amct_pytorch` (or set `PYTHONPATH` to the repo root).
One-stop platform example: `source /home/developer/Ascend/cann/set_env.sh`. Set `HF_ENDPOINT` yourself if you need a dataset mirror.
Some images ship a broken `torchaudio` that crashes during `transformers` import. Fix that local dependency, then use the official `python3 -m` entry. This sample does not ship an env wrapper.

- Model: local Qwen3-0.6B weights, `--model_name qwen3`, `--trust_remote_code`
- Calibration: Pileval (`mit-han-lab/pile-val-backup`, CLI default `nsamples=128`)
- Evaluation: WikiText2 `wikitext-2-raw-v1` test, `--seq_len 4096`, `--granularity block`
- BF16 and quantized eval must use the same weight directory and the same eval config
- Quantize Attention (`attn-linear`) and MLP; each format must finish extract → PTQ(attn) → PTQ(mlp) → quant eval with PTQ params
- Product directories are relative placeholders; do not commit data or weights

Run from the repo root:

```bash
source /path/to/cann/set_env.sh
export MODEL=/path/to/Qwen3-0.6B   # placeholder; point to local weights

# 1) BF16 baseline (shared by both formats)
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml

# 2) Extract PTQ offline data (attn + mlp)
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

# 3) int4/int8
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
  --output_dir examples/models/qwen3/outputs/qwen3_0_6b_w4a8_int

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
  --output_dir examples/models/qwen3/outputs/qwen3_0_6b_w4a8_int

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
  --attn_linear_param_dir examples/models/qwen3/outputs/qwen3_0_6b_w4a8_int/ptq_params/qwen3/attn-linear \
  --moe_mlp_param_dir examples/models/qwen3/outputs/qwen3_0_6b_w4a8_int/ptq_params/qwen3/mlp

# 4) mxfp4/mxfp8: rerun step 3 with --quant_dtype mxfp and
#    --output_dir / param dirs under outputs/qwen3_0_6b_w4a8_mxfp.
```

You can also export `MODEL` and call `scripts/*.sh`. The scripts print the full runtime arguments and use the same in-repo `w4a8.yaml`.

Key arguments:

| Arg / env | Meaning |
|:--|:--|
| `MODEL` | Local weight directory (placeholder) |
| `--model_name qwen3` | Dense Qwen3 path |
| `--seq_len 4096` | Calibration and evaluation sequence length |
| `--granularity block` | Block granularity |
| `--quant_dtype int\|mxfp` | int4/int8 or mxfp4/mxfp8 |
| `--algos lwc lac` | Quantization algorithms |
| `--bit_config` | In-repo `amct_pytorch/configs/w4a8.yaml` |
| `--quant_target` | One target per extract/PTQ step: `attn-linear` or `mlp`; quant eval loads both |
| `--nsamples` | Pileval calibration count, default 128 |
| `--end_block_idx` | Qwen3-0.6B default 28 layers |

## Artifacts

Relative placeholder directories (under this sample by default):

| Artifact | Path | Size |
|:--|:--|:--|
| PTQ offline data, Attention | `./outputs/qwen3_0_6b/ptq_data/attn-linear` | 28.002 GiB |
| PTQ offline data, MLP | `./outputs/qwen3_0_6b/ptq_data/mlp` | 28.002 GiB |
| Offline data total (attn+mlp) | sum of the two directories | 56.004 GiB |
| int PTQ params, Attention | `./outputs/qwen3_0_6b_w4a8_int/ptq_params/qwen3/attn-linear` | 1.30 MiB |
| int PTQ params, MLP | `./outputs/qwen3_0_6b_w4a8_int/ptq_params/qwen3/mlp` | 1.67 MiB |
| mxfp PTQ params, Attention | `./outputs/qwen3_0_6b_w4a8_mxfp/ptq_params/qwen3/attn-linear` | 42.20 MiB |
| mxfp PTQ params, MLP | `./outputs/qwen3_0_6b_w4a8_mxfp/ptq_params/qwen3/mlp` | 63.14 MiB |

Quant time: from the first `ptq` of that format until both attn and mlp params are written (excludes model download and env setup). Offline data size is the combined extract output, in GiB.

## Result Table

BF16 and quantized eval use the same `MODEL` and the same `seq_len=4096` / `granularity=block` config.

| Model | Data type | Algorithm | Format | BF16 PPL | Quant PPL | PPL delta | PTQ minutes | Offline data (GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | `w4a8` | `lwc` + `lac` | `int4/int8` | 19.157549 | 47.486488 | 28.328939 | 37.35 | 56.004 |
| Qwen3-0.6B | `w4a8` | `lwc` + `lac` | `mxfp4/mxfp8` | 19.157549 | 23.898159 | 4.740610 | 48.88 | 56.004 |

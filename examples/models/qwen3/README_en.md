# Qwen3-0.6B w4a8 `lwc` + `lac` Quantization Sample

Corresponds to Issue [#182](https://gitcode.com/cann/amct/issues/182) task 1: quantize Qwen3-0.6B with `w4a8` + `lwc`/`lac` in **both int4/int8 and mxfp4/mxfp8**.
This directory sits next to `qwen3.6` and `deepseekv4`. Do not edit the Qwen3.6 docs.

The CLI matches [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md): `python3 -m amct_pytorch.eval` / `extract_ptq_data` / `ptq`.
Some images ship a broken `torchaudio` that crashes during `transformers` import. The sample scripts call the same CLI through `scripts/amct_cli.py` after masking that dependency. If `python3 -m` works in your env, you can skip the wrapper.

## Run Instructions

Environment: single card `npu:0`. Source `/home/developer/Ascend/cann/set_env.sh` and install `amct_pytorch` (or set `PYTHONPATH` to the repo root).

- Model: local Qwen3-0.6B weights, `--model_name qwen3`, `--trust_remote_code`
- Calibration: Pileval (`mit-han-lab/pile-val-backup`, CLI default `nsamples=128`)
- Evaluation: WikiText2 `wikitext-2-raw-v1` test, `--seq_len 4096`, `--granularity block`
- BF16 and quantized eval must use the same weight directory and the same eval config
- Quantize Attention (`attn-linear`) and MLP; each format must finish extract → PTQ(attn) → PTQ(mlp) → quant eval with PTQ params
- Scripts print the full runtime arguments; product directories are relative placeholders; do not commit data or weights

```bash
# repo root
source /home/developer/Ascend/cann/set_env.sh
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export MODEL=/path/to/Qwen3-0.6B   # placeholder; point to local weights

cd examples/models/qwen3

# 1) BF16 baseline (shared by both formats)
bash scripts/eval_bf16.sh

# 2) Extract PTQ offline data (attn + mlp)
bash scripts/extract_ptq_data.sh

# 3) int4/int8
QUANT_DTYPE=int bash scripts/ptq_attn.sh
QUANT_DTYPE=int bash scripts/ptq_mlp.sh
QUANT_DTYPE=int bash scripts/eval_quant.sh

# 4) mxfp4/mxfp8
QUANT_DTYPE=mxfp bash scripts/ptq_attn.sh
QUANT_DTYPE=mxfp bash scripts/ptq_mlp.sh
QUANT_DTYPE=mxfp bash scripts/eval_quant.sh
```

Key arguments:

| Arg / env | Meaning |
|:--|:--|
| `MODEL` | Local weight directory (placeholder) |
| `--model_name qwen3` | Dense Qwen3 path |
| `--seq_len 4096` | Calibration and evaluation sequence length |
| `--granularity block` | Block granularity |
| `--quant_dtype int\|mxfp` | int4/int8 or mxfp4/mxfp8 |
| `--algos lwc lac` | Quantization algorithms |
| `--bit_config` | `configs/w4a8_int_lwc_lac.yaml` or `configs/w4a8_mxfp_lwc_lac.yaml` (derived from `w4a8.yaml`) |
| `--quant_target` | One target per extract/PTQ step: `attn-linear` or `mlp`; quant eval loads both |
| `NSAMPLES` | Pileval calibration count, default 128 |
| `END_BLOCK_IDX` | Qwen3-0.6B default 28 layers |

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

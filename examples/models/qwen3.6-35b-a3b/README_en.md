# Qwen3.6-35B-A3B w4a4 `autoround` Quantization Sample

Corresponds to Issue [#182](https://gitcode.com/cann/amct/issues/182) **task 24**: run `w4a4` + `autoround` on Qwen3.6-35B-A3B for **both int4/int4 and mxfp4/mxfp4**.

CLI entry points match [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md): `python3 -m amct_pytorch.eval` / `extract_ptq_data` / `ptq`.  
Configs live under `configs/`:

| Format | Config | CLI `--quant_dtype` |
| --- | --- | --- |
| int4/int4 | `configs/w4a4_int_autoround.yaml` | `int` |
| mxfp4/mxfp4 | `configs/w4a4_mxfp_autoround.yaml` | `mxfp` |

MoE quantization targets are `attn-linear` + `moe` (the adapter rejects `mlp`).

## How to Run

Environment: single NPU `npu:0`. Source your CANN env and install `amct_pytorch` (or set `PYTHONPATH` to the repo root).  
One-stop platform example: `source /home/developer/Ascend/cann/set_env.sh`. Set `HF_ENDPOINT` if you need a mirror for calibration/eval datasets.

- Model: local Qwen3.6-35B-A3B weights; `--model_name qwen3_6_moe`; `--trust_remote_code`
- Calibration: Pileval (`mit-han-lab/pile-val-backup`), default `nsamples=128`
- Evaluation: WikiText2 `wikitext-2-raw-v1` test; `--seq_len 4096`; `--granularity block`
- BF16 and quantized eval must use the same weight dir and eval settings
- Cover Attention (`attn-linear`) and MoE at eval (`--quant_target attn-linear moe`)
- **autoround PTQ defaults to Attention only** (same as official [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md)). Full MoE is ~`40×256` expert units and can take days on one NPU; `ptq_mlp.sh` SKIPs unless `RUN_MOE_PTQ=1`
- Per format: extract → PTQ(attn) → quantized eval (load attn PTQ params; MoE fake-quant via bit_config)
- Artifact paths are relative placeholders; do not commit data/weights

From the repository root:

```bash
source /path/to/cann/set_env.sh
export MODEL=/path/to/Qwen3.6-35B-A3B   # placeholder

bash examples/models/qwen3.6-35b-a3b/scripts/eval_bf16.sh
bash examples/models/qwen3.6-35b-a3b/scripts/extract_ptq_data.sh

export QUANT_DTYPE=int
bash examples/models/qwen3.6-35b-a3b/scripts/ptq_attn.sh
bash examples/models/qwen3.6-35b-a3b/scripts/ptq_mlp.sh   # SKIP by default
bash examples/models/qwen3.6-35b-a3b/scripts/eval_quant.sh

export QUANT_DTYPE=mxfp
bash examples/models/qwen3.6-35b-a3b/scripts/ptq_attn.sh
bash examples/models/qwen3.6-35b-a3b/scripts/eval_quant.sh
```

Scripts print the full runtime arguments on start.

Main parameters:

| Parameter / env | Meaning |
|:--|:--|
| `MODEL` | Local weight directory (placeholder) |
| `--model_name qwen3_6_moe` | Qwen3.6 MoE path |
| `--seq_len 4096` | Calibration/eval sequence length |
| `--granularity block` | Block granularity |
| `--quant_dtype int\|mxfp` | int4/int4 or mxfp4/mxfp4 |
| `--algos autoround` | Quantization algorithm |
| `--bit_config` | `configs/w4a4_{int\|mxfp}_autoround.yaml` |
| `--quant_target` | One target per extract/PTQ call: `attn-linear` or `moe`; quantized eval loads both |
| `--nsamples` | Pileval samples, default 128 |
| `--epochs` / `--base_lr` | autoround defaults `10` / `1e-3` |
| `--end_block_idx` | Default 40 (exclusive; clamped to `num_layers`) |

## Artifacts

Relative placeholder paths (default under this sample directory):

| Artifact | Path | Size |
|:--|:--|:--|
| PTQ offline data Attention | `./outputs/qwen3_6_35b_a3b/ptq_data/attn-linear` | TBD GiB |
| PTQ offline data MoE | `./outputs/qwen3_6_35b_a3b/ptq_data/moe` | TBD GiB |
| Offline data total (attn+moe) | sum of the two dirs above | TBD GiB |
| int PTQ params Attention | `./outputs/qwen3_6_35b_a3b_w4a4_int/ptq_params/qwen3_6_moe/attn-linear` | TBD |
| int PTQ params MoE | `./outputs/qwen3_6_35b_a3b_w4a4_int/ptq_params/qwen3_6_moe/moe` | TBD |
| mxfp PTQ params Attention | `./outputs/qwen3_6_35b_a3b_w4a4_mxfp/ptq_params/qwen3_6_moe/attn-linear` | TBD |
| mxfp PTQ params MoE | `./outputs/qwen3_6_35b_a3b_w4a4_mxfp/ptq_params/qwen3_6_moe/moe` | TBD |

Quantization time: from the first `ptq` of that format until attn+moe params are fully written (exclude model download and env setup). Offline data size: sum of extract output dirs, in GiB.

## Results

BF16 and quantized eval use the same `MODEL` and the same `seq_len=4096` / `granularity=block` settings.

| Model | Data type | Algorithm | Quant format | BF16 PPL | Quant PPL | PPL delta | Quant time (min) | Offline data (GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | `w4a4` | `autoround` | `int4/int4` | - | - | - | - | - |
| Qwen3.6-35B-A3B | `w4a4` | `autoround` | `mxfp4/mxfp4` | - | - | - | - | - |

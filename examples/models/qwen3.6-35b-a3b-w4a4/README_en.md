# Qwen3.6-35B-A3B w4a4 `autoround` Quantization Sample

Corresponds to Issue [#182](https://gitcode.com/cann/amct/issues/182) **task 24** (directory `qwen3.6-35b-a3b-w4a4/`, separate from other quant configs of the same model to avoid path conflicts): run `w4a4` + `autoround` on Qwen3.6-35B-A3B for **both int4/int4 and mxfp4/mxfp4**.

CLI entry points match [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md): `python3 -m amct_pytorch.eval` / `extract_ptq_data` / `ptq`.  
Bit-width config reuses in-repo [`amct_pytorch/configs/w4a4.yaml`](../../../amct_pytorch/configs/w4a4.yaml); choose `int` / `mxfp` via CLI `--quant_dtype`.

| Format | Config | CLI `--quant_dtype` |
| --- | --- | --- |
| int4/int4 | `amct_pytorch/configs/w4a4.yaml` | `int` |
| mxfp4/mxfp4 | `amct_pytorch/configs/w4a4.yaml` | `mxfp` |

MoE quantization targets are `attn-linear` + `moe` (the adapter rejects `mlp`).

## How to Run

Environment: single NPU `npu:0`. Source your CANN env and install `amct_pytorch` (or set `PYTHONPATH` to the repo root).  
One-stop platform example: `source /home/developer/Ascend/cann/set_env.sh`. Set `HF_ENDPOINT` if you need a mirror for calibration/eval datasets.

- Model: local Qwen3.6-35B-A3B weights; `--model_name qwen3_6_moe`; `--trust_remote_code`
- Calibration: Pileval (`mit-han-lab/pile-val-backup`), default `nsamples=128`
- Evaluation: WikiText2 `wikitext-2-raw-v1` test; `--seq_len 4096`; `--granularity block`
- BF16 and quantized eval must use the same weight dir and eval settings
- Cover Attention (`attn-linear`) and MoE at eval (`--quant_target attn-linear moe`)
- **autoround PTQ runs on Attention only**; **MoE uses direct (fake) quantization** at eval (`--quant_target attn-linear moe`, no per-expert autoround params). Matches official [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md) and the guidance to fall back to MoE direct quant when full expert PTQ is too slow
- `ptq_mlp.sh` SKIPs by default; set `RUN_MOE_PTQ=1` only to force full expert autoround (not recommended as the sample default)
- Per format: extract → PTQ(attn) → quantized eval (load attn PTQ params; MoE direct quant via bit_config)
- Artifact paths are relative placeholders; do not commit data/weights

### Time estimate

Scope: 1× `npu:0` (not multi-NPU); `autoround`, `epochs=10`, `seq_len=4096`, `end_block_idx=40`.

| Step | Wall time | Basis |
| --- | --- | --- |
| extract (attn + moe) | ~1.5–2.5 h | Measured wall-clock (attn + moe extract runs combined; offline data ~162 GiB) |
| Attention autoround PTQ (per dtype) | ~2 h (~120 min) | Measured ~120 min for int / mxfp; result-table “Quant time” uses this |
| Full MoE autoround PTQ | **~40 days on 1 NPU** | See extrapolation below; skipped by default → MoE direct quant |
| Quantized eval | ~0.5–1 h | Measured wall-clock (block fake-quant forward + WikiText2 PPL) |

**Full MoE extrapolation (1× `npu:0`, mxfp trial):**

- Expert units: `40 layers × 256 experts ≈ 10240`
- Measured progress: wrote `layer_0_expert_205.pt`, running `expert_206` (~**206/256** of layer 0) after ~**20 h** wall-clock for that MoE PTQ run
- Per-expert average: `20 h / 206 ≈ 5.8 min/expert`
- Full layer 0: `20 × 256 / 206 ≈ 24.9 h/layer`
- All layers: `24.9 h/layer × 40 ≈ 996 h ≈ **41.5 days**` (or `10240 × 5.8 min ≈ 990 h`)

“Quant time” in the results table = wall-clock of **Attention autoround PTQ** for that format (exclude extract / eval / model download / env setup).

From the repository root:

```bash
source /path/to/cann/set_env.sh
export MODEL=/path/to/Qwen3.6-35B-A3B   # placeholder

bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/eval_bf16.sh
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/extract_ptq_data.sh

export QUANT_DTYPE=int
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/ptq_attn.sh
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/ptq_mlp.sh   # SKIP by default
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/eval_quant.sh

export QUANT_DTYPE=mxfp
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/ptq_attn.sh
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/eval_quant.sh
```

You can also use the equivalent `python3 -m ...` commands in this README; scripts print the full runtime arguments on start.

Main parameters:

| Parameter / env | Meaning |
|:--|:--|
| `MODEL` | Local weight directory (placeholder) |
| `--model_name qwen3_6_moe` | Qwen3.6 MoE path |
| `--seq_len 4096` | Calibration/eval sequence length |
| `--granularity block` | Block granularity |
| `--quant_dtype int\|mxfp` | int4/int4 or mxfp4/mxfp4 |
| `--algos autoround` | Quantization algorithm |
| `--bit_config` | `amct_pytorch/configs/w4a4.yaml` |
| `--quant_target` | One target per extract/PTQ call: `attn-linear` or `moe`; quantized eval loads both |
| `--nsamples` | Pileval samples, default 128 |
| `--epochs` / `--base_lr` | Sample values `10` / `1e-3` (CLI defaults are often `15` / `1e-5`; set in `scripts/common.sh`) |
| `--end_block_idx` | Sample value `40` (exclusive). Model has 40 layers (full coverage); CLI default `61` is clamped to layer count in PTQ |

Equivalent CLI example (int):

```bash
python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/qwen3.6-35b-a3b-w4a4/outputs/qwen3_6_35b_a3b/ptq_data/attn-linear \
  --quant_dtype int \
  --algos autoround \
  --bit_config amct_pytorch/configs/w4a4.yaml \
  --quant_target attn-linear \
  --start_block_idx 0 \
  --end_block_idx 40 \
  --epochs 10 \
  --base_lr 1e-3 \
  --output_dir examples/models/qwen3.6-35b-a3b-w4a4/outputs/qwen3_6_35b_a3b_w4a4_int
```

## Artifacts

Relative placeholder paths (default under this sample directory):

| Artifact | Path | Size |
|:--|:--|:--|
| PTQ offline data Attention | `./outputs/qwen3_6_35b_a3b/ptq_data/attn-linear` | 81 GiB |
| PTQ offline data MoE | `./outputs/qwen3_6_35b_a3b/ptq_data/moe` | 81 GiB |
| Offline data total (attn+moe) | sum of the two dirs above | 162 GiB |
| int PTQ params Attention | `./outputs/qwen3_6_35b_a3b_w4a4_int/ptq_params/qwen3_6_moe/attn-linear` | 4.8 GiB |
| int PTQ params MoE | (not generated by default; only with `RUN_MOE_PTQ=1`) | — |
| mxfp PTQ params Attention | `./outputs/qwen3_6_35b_a3b_w4a4_mxfp/ptq_params/qwen3_6_moe/attn-linear` | 5.1 GiB |
| mxfp PTQ params MoE | (not generated by default) | — |
| BF16 / int eval logs | `./outputs/qwen3_6_35b_a3b_w4a4_int/logs/` (e.g. `eval_qwen3_6_moe_bf16.log` / `eval_qwen3_6_moe_quant.log`) | — |
| mxfp eval logs | `./outputs/qwen3_6_35b_a3b_w4a4_mxfp/logs/` (e.g. `eval_qwen3_6_moe_quant.log`) | — |
| extract logs | `./outputs/qwen3_6_35b_a3b_w4a4_int/logs/` (default `QUANT_DTYPE=int`; e.g. `extract_ptq_data.log`; attn/moe append to the same file) | — |

Offline data size: sum of extract output dirs, in GiB. Eval / extract logs share the corresponding `--output_dir` (`OUTPUT_DIR`), so they do not land under the repo-root `./outputs/logs/`.

## Results

BF16 and quantized eval use the same `MODEL` and the same `seq_len=4096` / `granularity=block` settings.  
Eval covers `attn-linear` + `moe`: Attention loads autoround PTQ params; **MoE uses w4a4 direct fake-quant** (full expert autoround is too slow).

| Model | Data type | Algorithm | Quant format | BF16 PPL | Quant PPL | PPL delta | Quant time (min) | Offline data (GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | `w4a4` | `autoround` | `int4/int4` | 6.308547 | 264.116119 | 257.807572 | 120 | 162 |
| Qwen3.6-35B-A3B | `w4a4` | `autoround` | `mxfp4/mxfp4` | 6.308547 | 6.876903 | 0.568356 | 120 | 162 |

**Accuracy note**: The `int4/int4` PPL 264.116119 is the **expected** result for the default path (Attention autoround only + MoE int4 direct fake-quant), matching the corresponding `eval_quant` log — not a misconfiguration. Under the same setup, `mxfp4/mxfp4` is much friendlier to direct quant and reaches ~6.88. To improve int quality, set `RUN_MOE_PTQ=1` for full expert autoround (~41.5 days wall-clock; off by default).

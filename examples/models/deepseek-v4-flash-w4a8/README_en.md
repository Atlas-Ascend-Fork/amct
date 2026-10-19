# DeepSeek-V4-Flash w4a8 `lwc` + `lac` sample

Issue [#182](https://gitcode.com/cann/amct/issues/182) **task 25** (directory `deepseek-v4-flash-w4a8/`, separate from the W4A4 sample under `deepseekv4/`): `w4a8` + `lwc` + `lac` on DeepSeek-V4-Flash, **once for int4/int8 and once for mxfp4/mxfp8**.

CLI: `python3 -m amct_pytorch.eval` / `extract_ptq_data` / `ptq`.  
Bit widths reuse [`amct_pytorch/configs/w4a8.yaml`](../../../amct_pytorch/configs/w4a8.yaml) (routed experts `w4a8`, shared experts `w8a8`). `int` / `mxfp` is selected by `--quant_dtype`.

| Format | Config | CLI `--quant_dtype` |
| --- | --- | --- |
| int4/int8 | `amct_pytorch/configs/w4a8.yaml` | `int` |
| mxfp4/mxfp8 | `amct_pytorch/configs/w4a8.yaml` | `mxfp` |

Quant targets are `attn-linear` + `moe` (no dense `mlp`).

## How to run

Environment: **one** `npu:0` (64GB HBM). If two NPUs are visible, set `ASCEND_RT_VISIBLE_DEVICES=0` before eval so activations and expert weights stay on the same card. Install `amct_pytorch` or set `PYTHONPATH` to the repo root.  
Example: `source /home/developer/Ascend/cann/set_env.sh`. Set `HF_ENDPOINT` if calibration or eval data must be downloaded from a mirror.

- Weights: local DeepSeek-V4-Flash; `--model_name deepseek_v4`; `--trust_remote_code`
- The official checkpoint is mixed: Attention / shared experts are FP8 e4m3 with `float8_e8m0fnu` scales (128×128 blocks); routed experts are packed int8 MXFP4 with e8m0 scales (group 32 on the last dimension). Reported numbers used **load-time dequant**. No full bf16 copy was stored. To reproduce, point `MODEL` at a bf16 export from [DeepSeekV4-Flash-Walkthrough_en.md](../deepseekv4/DeepSeekV4-Flash-Walkthrough_en.md), or dequant with `weight_dequant` at load time
- Calibration: Pileval (`mit-han-lab/pile-val-backup`), `nsamples=128`
- Eval: WikiText2 `wikitext-2-raw-v1` test; `--seq_len 4096`; `--granularity block`
- BF16 and quant eval share the same weights and eval settings
- **`lwc` + `lac` trains Attention only**. **MoE is w4a8 direct fake-quant** (`--quant_target attn-linear moe`, no per-expert PTQ params)
- `ptq_mlp.sh` SKIPs unless `RUN_MOE_PTQ=1`
- Per format: extract → PTQ(attn) → quant eval
- Artifact paths are placeholders. Do not commit data or weights
- **Quant eval (single-NPU OOM + sample patch)**: The result table was measured on **one NPU + a sample-local monkeypatch**, not multi-NPU sharding. Bare `python3 -m amct_pytorch.eval` sets `sharded_block=True`; on one card `_build_block_device_map` returns `None` and the whole block (all 256 routed experts) lands on `npu:0` and OOMs. Per-expert moves exist only on the PTQ path (`llm_ptq.py`). This sample's `eval_quant.sh` calls [`scripts/eval_quant_expert_offload.py`](scripts/eval_quant_expert_offload.py): park routed experts on CPU after `_dispatch_block`, then `.to(npu:0)` one expert at a time in MoE `forward`. **No changes under `amct_pytorch/`**. Upstream: [Issue #248](https://gitcode.com/cann/amct/issues/248). The same-model [W4A4 sample](../deepseekv4/README_en.md) records 2×Atlas A3; this sample reproduces on one card via the patch above.

### Time

Single `npu:0`; `lwc` + `lac`, `epochs=15`, `nsamples=128`, `cali_bsz=1`, `seq_len=4096`, `end_block_idx=43`.

| Step | Wall clock | Basis |
| --- | --- | --- |
| extract (attn + moe) | ~5.3 h | Each target's block loop was ~2 h 39 min (43 layers, ~222 s/layer) |
| Attention `lwc`+`lac` (each dtype) | ~30 h (1806 min) | Stable stretch ~**42 min/layer** × 43 layers. This is the table's quant time |
| Full MoE `lwc`+`lac` | **weeks to hundreds of days (1 NPU)** | See below. Default is MoE direct quant |
| Quant eval | ~6.5–7 h / format | Measured mxfp block loop 6 h 39 min (one NPU + `eval_quant_expert_offload.py`, experts one at a time); int is the same order |
| BF16 eval | ~1.5–2 h | Measured block loop ~1 h 30 min |

**Full MoE extrapolation (one `npu:0`):**

- 43 layers. Routed experts: **256** per layer, same model as the [W4A4 sample](../deepseekv4/README_en.md)
- Attention is 1 PTQ unit/layer, measured ~42 min/layer (15 epochs × 128 steps, `cali_bsz=1`)
- MoE is one unit per expert: `43 × 256 = 11008` units
- If one expert costs as much as one Attention unit: `42 min × 11008 ≈ 462336 min` ≈ **321 days** / format
- At 1/10 of that cost, one format is still ~**32 days**. Default stays direct quant

Quant time in the result table is **Attention PTQ** only (excludes extract, eval, download, and environment setup).

From the repo root:

```bash
source /path/to/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL=/path/to/DeepSeek-V4-Flash

bash examples/models/deepseek-v4-flash-w4a8/scripts/eval_bf16.sh
bash examples/models/deepseek-v4-flash-w4a8/scripts/extract_ptq_data.sh

export QUANT_DTYPE=int
bash examples/models/deepseek-v4-flash-w4a8/scripts/ptq_attn.sh
bash examples/models/deepseek-v4-flash-w4a8/scripts/ptq_mlp.sh   # SKIP by default
bash examples/models/deepseek-v4-flash-w4a8/scripts/eval_quant.sh

export QUANT_DTYPE=mxfp
bash examples/models/deepseek-v4-flash-w4a8/scripts/ptq_attn.sh
bash examples/models/deepseek-v4-flash-w4a8/scripts/eval_quant.sh
```

Equivalent `python3 -m ...` commands are below. Scripts print the full runtime arguments.

| Arg / env | Meaning |
|:--|:--|
| `MODEL` | Local weight directory (placeholder) |
| `--model_name deepseek_v4` | DeepSeek-V4 path |
| `--seq_len 4096` | Calibration and eval length |
| `--granularity block` | Block granularity |
| `--quant_dtype int\|mxfp` | int4/int8 or mxfp4/mxfp8 |
| `--algos lwc lac` | Algorithms |
| `--bit_config` | `amct_pytorch/configs/w4a8.yaml` |
| `--quant_target` | extract/PTQ: one of `attn-linear` or `moe`; eval covers both |
| `--nsamples` | Sample value `128`. extract and PTQ must match |
| `--cali_bsz` | Sample value `1` (CLI default 4 OOMs on 64GB) |
| `--epochs` / `--base_lr` | Sample values `15` / `1e-3` (CLI default `base_lr` is `1e-5`) |
| `--end_block_idx` | Sample value `43` (exclusive). The model has 43 layers. CLI default `61` is clamped |

```bash
python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name deepseek_v4 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/deepseek-v4-flash-w4a8/outputs/deepseek_v4_flash/ptq_data/attn-linear \
  --quant_dtype int \
  --algos lwc lac \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_target attn-linear \
  --nsamples 128 \
  --cali_bsz 1 \
  --start_block_idx 0 \
  --end_block_idx 43 \
  --epochs 15 \
  --base_lr 1e-3 \
  --output_dir examples/models/deepseek-v4-flash-w4a8/outputs/deepseek_v4_flash_w4a8_int
```

## Artifacts

| Artifact | Path | Size |
|:--|:--|:--|
| PTQ offline data Attention | `./outputs/deepseek_v4_flash/ptq_data/attn-linear` | 173 GiB |
| PTQ offline data MoE | `./outputs/deepseek_v4_flash/ptq_data/moe` | 173 GiB |
| Offline data total | sum of the two dirs | 346 GiB |
| int PTQ params Attention | `./outputs/deepseek_v4_flash_w4a8_int/ptq_params/deepseek_v4/attn-linear` | 18 MiB |
| int PTQ params MoE | (not generated unless `RUN_MOE_PTQ=1`) | — |
| mxfp PTQ params Attention | `./outputs/deepseek_v4_flash_w4a8_mxfp/ptq_params/deepseek_v4/attn-linear` | 1.2 GiB |
| mxfp PTQ params MoE | (not generated by default) | — |
| BF16 / int eval logs | `./outputs/deepseek_v4_flash_w4a8_int/logs/` | — |
| mxfp eval logs | `./outputs/deepseek_v4_flash_w4a8_mxfp/logs/` | — |
| extract logs | `./outputs/deepseek_v4_flash_w4a8_int/logs/` (default `QUANT_DTYPE=int`; attn and moe append) | — |

Logs share `--output_dir` (`OUTPUT_DIR`) and do not land under the repo-root `./outputs/logs/`.

## Results

BF16 and quant eval share `MODEL`, `seq_len=4096`, and `granularity=block`.  
Eval covers `attn-linear` + `moe`: Attention loads `lwc`+`lac` params; **MoE is w4a8 direct fake-quant**.

| Model | Data type | Algorithm | Quant format | BF16 PPL | Quant PPL | PPL delta | Quant time (min) | Offline data (GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4-Flash | `w4a8` | `lwc` + `lac` | `int4/int8` | 4.145633 | 4.779413 | 0.633780 | 1806 | 346 |
| DeepSeek-V4-Flash | `w4a8` | `lwc` + `lac` | `mxfp4/mxfp8` | 4.145633 | 4.270695 | 0.125062 | 1806 | 346 |

**Accuracy note**: Both rows are the default path (Attention `lwc`+`lac` only + MoE w4a8 direct quant), matching the `eval_quant` log. The int delta 0.633780 and the mxfp delta 0.125062 are expected for that path. `RUN_MOE_PTQ=1` is the full-expert path (tens to hundreds of days per format; off by default).

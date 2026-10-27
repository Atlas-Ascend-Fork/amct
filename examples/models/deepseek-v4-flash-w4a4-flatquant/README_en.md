# DeepSeek-V4-Flash w4a4 `flatquant` Quantization Sample

For Issue [#182](https://gitcode.com/cann/amct/issues/182) **Task 30** (L3, `w4a4` + `flatquant`): runs `w4a4` + `flatquant` on DeepSeek-V4-Flash, **once for int4/int4 and once for mxfp4/mxfp4**.

CLI entry points: `amct_pytorch.eval` / `extract_ptq_data` / `ptq` (wrapped via `scripts/run_amct.py`).  
The bit-width config reuses the in-repo [`amct_pytorch/configs/w4a4.yaml`](../../../amct_pytorch/configs/w4a4.yaml) (`w_bits: 4, a_bits: 4`); `int` / `mxfp` is selected via CLI `--quant_dtype`. Mapping to the task-book config names: `w4a4_int_flatquant.yaml` ≡ `w4a4.yaml` + `--quant_dtype int`, `w4a4_mxfp_flatquant.yaml` ≡ `w4a4.yaml` + `--quant_dtype mxfp`.

## Loading the official checkpoint (load-time dequantization)

The official DeepSeek-V4-Flash checkpoint is mixed-format: attention / shared-expert linears are FP8 e4m3 with 128x128 e8m0 scale blocks, and routed experts are int8-packed MXFP4 with per-32-element e8m0 scales. A full bf16 re-export is ~600 GB and cannot coexist with the 149 GB source on a 300 GB disk, so this sample uses **load-time dequantization**: [`scripts/dequant_load_patch.py`](scripts/dequant_load_patch.py) proxies `safe_open` and dequants each tensor to bf16 on CPU as the per-block loader reads it (math identical to the in-repo `amct_pytorch/experimental/deepseek-v4/convert_model.py`), ~13 GB transient per layer, with zero changes to `amct_pytorch/`.

## Running

Environment: 2x Atlas A3 (64GB HBM). extract/PTQ occupy both cards via a single serialized sharded process (two concurrent sharded loaders on one host contend for the npu:0 stem and OOM). Configure the CANN environment first and install `amct_pytorch` (or point `PYTHONPATH` at the repo root). The Pileval calibration set (`mit-han-lab/pile-val-backup`) and WikiText2 (`wikitext-2-raw-v1` test) download automatically on first use (`HF_ENDPOINT` optional).

- Model: local official DeepSeek-V4-Flash weights; `--model_name deepseek_v4`; `--trust_remote_code`
- `flatquant` covers both `attn-linear` and `moe` quant targets; both load real PTQ parameters
- Per format: extract (attn + moe) -> PTQ (attn) -> PTQ (moe) -> quantized eval (loads both param sets)
- Environment variables and defaults are documented at the top of each script; all paths are relative/placeholder and every script prints its full runtime arguments

### Step 1: BF16 baseline (multi-card device map)

```shell
bash scripts/eval_bf16.sh
```

### Step 2: PTQ offline data extraction

`extract_ptq_data` supports a single `--quant_target` per invocation; run once per target:

```shell
QUANT_TARGET=attn-linear bash scripts/extract_ptq_data.sh
QUANT_TARGET=moe bash scripts/extract_ptq_data.sh
```

### Step 3: flatquant PTQ

```shell
QUANT_DTYPE=int bash scripts/ptq_attn.sh
QUANT_DTYPE=int bash scripts/ptq_moe.sh
QUANT_DTYPE=mxfp bash scripts/ptq_attn.sh
QUANT_DTYPE=mxfp bash scripts/ptq_moe.sh
```

### Step 4: Quantized accuracy evaluation (single-card expert-offload)

```shell
QUANT_DTYPE=int bash scripts/eval_quant.sh
QUANT_DTYPE=mxfp bash scripts/eval_quant.sh
```

## Artifacts

Output root `outputs/dsv4_flash_w4a4_flatquant/`:

| Artifact | Directory | Size |
| --- | --- | --- |
| BF16 baseline eval | `01_baseline_eval/` | logs (<0.1 GiB) |
| attn-linear calibration data | `ptq_data/attn-linear/` (`02_extract_attnlinear/`) | ~22 GiB |
| moe calibration data | `ptq_data/moe/` (`02_extract_moe/`) | ~22 GiB |
| int PTQ params (attn-linear + moe) | `ptq_params/int/{attn-linear,moe}/` | MB-scale |
| int PTQ run logs (moe) | `04_ptq_moe_int/` (run logs) | logs |
| int PTQ run logs (attn) | `03_ptq_attn_int/` (run logs) | logs |
| mxfp PTQ params (attn-linear + moe) | `ptq_params/mxfp/{attn-linear,moe}/` | MB-scale |
| mxfp PTQ params (moe) | `04_ptq_moe_mxfp/` (run logs) | logs |
| mxfp PTQ run logs (attn) | `03_ptq_attn_mxfp/` (run logs) | logs |
| int quantized eval | `05_eval_quant_int/` | logs (<0.1 GiB) |
| mxfp quantized eval | `05_eval_quant_mxfp/` | logs (<0.1 GiB) |

## Results

Eval environment: 2x Atlas A3 (64GB HBM), `seq_len=4096`, WikiText2 test split, Pileval calibration (`nsamples=16`).

| Model | Dtype | Algo | Format | BF16 PPL | Quant PPL | PPL delta | Quant time (min) | Offline data (GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4-Flash | `w4a4` | `flatquant` | `int4/int4` | 4.1456 | 26.0635 | 21.9179 | 572 | 44 |
| DeepSeek-V4-Flash | `w4a4` | `flatquant` | `mxfp4/mxfp4` | 4.1456 | 4.6865 | 0.5409 | 797 | 44 |

Metric definitions:

- **BF16 PPL**: WikiText2 PPL under `eval_mode=bf16`.
- **Quant PPL**: WikiText2 PPL of the quantized model with the flatquant PTQ parameters of attention and MoE loaded.
- **PPL delta**: `Quant PPL - BF16 PPL`.
- **Quant time**: wall time from PTQ start until all quantization parameters are produced (excluding model download and environment setup).
- **Offline data**: total size of the `extract_ptq_data` output, attention and moe data directories combined.

## Notes

- Environment: 2x Atlas A3 (64GB HBM). Actual per-step paths: `extract_ptq_data.sh` and `ptq_*.sh` run as a single process with sharded loading over both cards (the attn-linear and moe targets execute serially); `eval_bf16.sh` uses a multi-card device map; `eval_quant.sh` uses the single-card expert-offload path in `scripts/eval_quant_expert_offload.py` (`DEVICE=npu:0`, same scheme as the w4a8 sample) which a single 64GB card can reuse directly.
- Both result-table rows were measured in this environment: the BF16 baseline went through the multi-card device map while quantized eval used the expert-offload path — the two paths differ in memory behavior, so align card configs per step when reproducing.
- First flatquant run on DeepSeek-V4-Flash in this repo. Actual hyperparameters: `ptq_attn.sh` defaults to `EPOCHS=15`, `ptq_moe.sh` to `EPOCHS=3` (the two scripts differ; each result-table row ran with its own value); `BASE_LR=1e-3` and `CALI_BSZ=1` are shared defaults.

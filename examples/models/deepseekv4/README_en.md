# DeepSeek-V4-Flash W4A4 Quantization Example (lwc + lac)

This example quantizes DeepSeek-V4-Flash to `w4a4` (4-bit weights + 4-bit activations) with the `lwc` + `lac` post-training quantization algorithms, covering both the Attention and MoE quantization targets, and delivers a reproducible workflow for both the `int4/int4` and `mxfp4/mxfp4` quantization formats.

## How to Run

The quantization workflow has four steps: BF16 baseline eval → PTQ offline data extraction → PTQ training → quantized eval with PTQ params. All scripts live in `scripts/`; the bit-width config is the repo-level `amct_pytorch/configs/w4a4.yaml`.

### Prerequisites

- Model weights: download from [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (an FP8+FP4 mixed-format checkpoint, ~149 GB). The `deepseek_v4` adapter dequantizes 1-byte weights on load, so no bf16 conversion copy is needed:
  - Attention weights (FP8 e4m3) are dequantized with 128x128 block scales;
  - MoE expert weights (packed-int8 MXFP4) are unpacked and dequantized with per-row 32-element-group E8M0 scales;
  - the checkpoint's `.scale` keys are consumed and dropped on load, so `load_state_dict(strict=True)` is not blocked by the extra keys.
  This example also fixes two adapter loading defects: `_block_sharded` previously cast FP8 weights with a bare dtype conversion (numerically wrong) and built meta modules under an fp32 default dtype that silently upcast weights to fp32 (per-layer HBM 24.1G -> 12.1G); `_block_staged` previously applied a blanket `.bfloat16()` that broke fp32-declared HC/RMSNorm params.
- Calibration set: Pileval (`mit-han-lab/pile-val-backup`). Eval set: WikiText2 (`wikitext-2-raw-v1` test split), `seq_len=4096`. Both download automatically on first run.
- Hardware: a single card is sufficient; with two cards, PTQ can be parallelized by block via `--start_block_idx/--end_block_idx`. On a 64GB card, PTQ requires `--cali_bsz 1`.

### Step 1: BF16 baseline

```shell
bash scripts/eval_bf16.sh
```

Defaults: `MODEL_PATH` (model directory), `BIT_CONFIG=amct_pytorch/configs/bf16.yaml`, `SEQ_LEN=4096`, `DEVICE=npu:0`. The baseline PPL is the reference for quantization loss.

### Step 2: PTQ offline data extraction

`extract_ptq_data` supports a single `quant_target` per invocation, so run it once per target — once for Attention and once for MoE (default `NSAMPLES=16`; each extraction produces ~22 GiB of data, ~43 GiB for both targets — plan disk accordingly):

```shell
QUANT_TARGET=attn-linear bash scripts/extract_ptq_data.sh
QUANT_TARGET=moe bash scripts/extract_ptq_data.sh
```

`--nsamples` is bounded by disk budget: this model's hidden state is a 4D Hyper-Connection shape (`hc_mult=4`), roughly 134 MB per sample per layer; `nsamples=128` would need ~170 GiB per target, so this example uses 16.

### Step 3: PTQ training

Attention and MoE are trained separately (DeepSeek-V4 is a pure MoE model with no dense MLP branch; `quant_target=mlp` is rejected by the adapter):

```shell
# int4/int4 format
QUANT_DTYPE=int bash scripts/ptq_attn.sh
QUANT_DTYPE=int bash scripts/ptq_moe.sh

# mxfp4/mxfp4 format
QUANT_DTYPE=mxfp bash scripts/ptq_attn.sh
QUANT_DTYPE=mxfp bash scripts/ptq_moe.sh
```

Key parameters:

- `QUANT_DTYPE`: `int` or `mxfp`. The bit-width config is the repo-level `amct_pytorch/configs/w4a4.yaml` (`w_bits: 4 / a_bits: 4`); the quantization format is selected by `--quant_dtype`.
- `BASE_LR=1e-3`: learning rate. **Do not use the framework default of 1e-5** — with the default, the `lwc/lac` clip factors barely update (they stay near the initial value of 4.0) and quantization accuracy degrades significantly.
- `EPOCHS`: 15 for Attention, 3 for MoE. Each MoE layer has 256 routed experts trained one by one; 3 epochs already converge.
- `CALI_BSZ=1`: the minimum batch size for a 64GB card.

MoE training can be parallelized across cards by block (e.g. two cards, half the layers each):

```shell
ASCEND_RT_VISIBLE_DEVICES=0 START_BLOCK_IDX=0 END_BLOCK_IDX=22 QUANT_DTYPE=int bash scripts/ptq_moe.sh &
ASCEND_RT_VISIBLE_DEVICES=1 START_BLOCK_IDX=22 END_BLOCK_IDX=43 QUANT_DTYPE=int bash scripts/ptq_moe.sh &
wait
```

### Step 4: Quantized eval

Load both the Attention and MoE PTQ params and evaluate the quantized model:

```shell
QUANT_DTYPE=int bash scripts/eval_quant.sh
QUANT_DTYPE=mxfp bash scripts/eval_quant.sh
```

## Artifacts

| Artifact | Path | Size |
|---|---|---|
| PTQ offline data (Attention) | `${OUTPUT_DIR}/ptq_data/attn-linear/` (43 `block_{i}_attn_in.pkl` files) | 21.5 GiB |
| PTQ offline data (MoE) | `${OUTPUT_DIR}/ptq_data/moe/` (43 `block_{i}_moe_in.pkl` files) | 21.5 GiB |
| PTQ params (Attention, int/mxfp) | `${OUTPUT_DIR}/ptq_params/{int,mxfp}/attn-linear/` (43 `layer_{i}_attn.pt` files) | 18 MB / 1.2 GB |
| PTQ params (MoE, int/mxfp) | `${OUTPUT_DIR}/ptq_params/{int,mxfp}/moe/` (43×256 `layer_{i}_expert_{j}.pt` files) | 789 MB / 69 GB |

Note: `lwc` clip factors for the mxfp format are stored per 32-element group (int is per-channel), which is why the mxfp MoE params are far larger than int. Offline data and PTQ params of the two targets can be produced sequentially and reused: extract one target's data, run PTQ for both formats, then move on to the other target.

## Results

Evaluated on 2×Atlas A3 (64GB HBM), `seq_len=4096`, WikiText2 test split, Pileval calibration (`nsamples=16`), `base_lr=1e-3`.

| Model | Data type | Algorithm | Format | BF16 PPL | Quantized PPL | PPL delta | PTQ time (min) | Offline data (GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4-Flash | `w4a4` | `lwc` + `lac` | `int4/int4` | 4.146 | 12.700 | 8.554 | 273 | 43.0 |
| DeepSeek-V4-Flash | `w4a4` | `lwc` + `lac` | `mxfp4/mxfp4` | 4.146 | 4.538 | 0.393 | 380 | 43.0 |

Metric definitions:

- **BF16 PPL**: WikiText2 PPL under `eval_mode=bf16`.
- **Quantized PPL**: WikiText2 PPL of the quantized model with Attention and MoE PTQ params loaded.
- **PPL delta**: `quantized PPL - BF16 PPL`.
- **PTQ time**: wall time from the start of PTQ to all quantization params being generated (Attention and MoE run in parallel across two cards; model download and environment setup excluded).
- **Offline data size**: total size of the `extract_ptq_data` output, Attention plus MoE data directories.

Interpretation: `mxfp4/mxfp4` retains near-baseline accuracy even with 4-bit activations (PPL delta 0.393). The symmetric int4 activation quantization (16 levels) is sensitive to this model's outlier-heavy activation distributions, giving a PPL delta of 8.554; prefer the mxfp format when accuracy matters.

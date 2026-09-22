# DeepSeek-V4-Flash W4A4 Quantization Example (lwc + lac)

This example quantizes DeepSeek-V4-Flash to `w4a4` (4-bit weights + 4-bit activations) with the `lwc` + `lac` post-training quantization algorithms, covering both the Attention and MoE quantization targets, and delivers a reproducible workflow for both the `int4/int4` and `mxfp4/mxfp4` quantization formats.

## How to Run

The quantization workflow has four steps: BF16 baseline eval → PTQ offline data extraction → PTQ training → quantized eval with PTQ params. All scripts live in `scripts/`; the bit-width config is the repo-level `amct_pytorch/configs/w4a4.yaml`.

Run the commands below from the sample directory `examples/models/deepseekv4/` (the scripts anchor the bit-width config and the artifact directory to absolute paths, so invoking them from the repo root or any other cwd works as well). All four steps must reuse the same `OUTPUT_DIR` (default `<sample dir>/outputs/dsv4_flash_w4a4`): the step-2 calibration data and the step-3 PTQ params are both derived from it, so switching cwd mid-workflow leaves step 3 unable to find the step-2 data.

### Prerequisites

- Model weights: download from [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (~149 GB). The official checkpoint mixes two 1-byte formats: Attention weights are FP8 e4m3 (with 128x128 block scales) and MoE expert weights are packed-int8 MXFP4 (with per-row 32-element-group E8M0 scales). Both must be multiplied back by their paired `.scale` keys to recover the real values.
- **The `deepseek_v4` adapter on master does not dequantize at load time**: `_block_sharded` only applies a bare dtype cast to floating-point tensors (no block scale, so FP8 values come out wrong) and does not even cast packed-int8 MXFP4; the PTQ path collects every key under the `layers.{i}.` prefix and then calls `load_state_dict(strict=True)`, so the extra `.scale` keys raise. Running the four steps of this sample directly on the official checkpoint therefore yields wrong numbers or a load failure.
- The result table in this sample was produced with an extra **load-time dequantization glue layer**: a `_dequant_tensor` helper plus a `load_layer_weight` override in the adapter, which fetches each `*.weight` together with its paired `.scale` and calls AMCT's existing `from amct_pytorch.quantization.dtypes.fp_impl import weight_dequant` (source file `amct_pytorch/quantization/dtypes/fp_impl.py`; the MX unpack/dequant helpers live in that same file — there is no `mxfp_impl.py` in the repo) to restore bf16 (`block_size=128` for FP8; `block_size=32, is_mx=True, is_packed=True` for MXFP4, matching the adapter's `block_size` convention of 32 for int8 weights and 128 otherwise), dropping the consumed `.scale` keys. That glue is sample-side work and is **not part of this PR — `amct_pytorch/` is unchanged**. The equivalent FP8 / MXFP4 / MXFP8 / NVFP4 / HiF4 dispatch already exists in the export path `amct_pytorch/common/models/llm/common/deploy_export.py` (its `convert_state_dict` is the reference implementation of exactly this call) and can be mirrored into the load path.
- Pick one before reproducing the results:
  - **(a) Convert the weights to bf16 first (recommended, no code change)**: follow section 2 of [DeepSeekV4-Flash-Walkthrough_en.md](DeepSeekV4-Flash-Walkthrough_en.md) and export a bf16 copy with the `deploy` entry point using `--granularity tensor`, then point `MODEL_PATH` at that copy;
  - **(b) Add the load-time dequantization glue described above to the adapter yourself**, after which `MODEL_PATH` can point directly at the official checkpoint.
- Calibration set: Pileval (`mit-han-lab/pile-val-backup`). Eval set: WikiText2 (`wikitext-2-raw-v1` test split), `seq_len=4096`. Both download automatically on first run.
- Hardware: a single card is sufficient; with two cards, PTQ can be parallelized by block via `--start_block_idx/--end_block_idx`. On a 64GB card, PTQ requires `--cali_bsz 1`.

### Step 1: BF16 baseline

```shell
bash scripts/eval_bf16.sh
```

Defaults: `MODEL_PATH` (model directory), `BIT_CONFIG=<repo root>/amct_pytorch/configs/bf16.yaml`, `SEQ_LEN=4096`, `DEVICE=npu:0`. The baseline PPL is the reference for quantization loss.

### Step 2: PTQ offline data extraction

`extract_ptq_data` supports a single `quant_target` per invocation, so run it once per target — once for Attention and once for MoE (default `NSAMPLES=16`; each extraction produces ~22 GiB of data, ~43 GiB for both targets — plan disk accordingly):

```shell
QUANT_TARGET=attn-linear bash scripts/extract_ptq_data.sh
QUANT_TARGET=moe bash scripts/extract_ptq_data.sh
```

`--nsamples` is bounded by disk budget: the capture hooks persist the outputs of `attn_norm` / `ffn_norm`, and `Block.forward` first collapses the Hyper-Connection `hc` dimension (`hc_mult=4`) into `[b, s, d]` through `hc_pre`, after which RMSNorm casts back to bf16 via `.to(dtype)`. The saved tensors are therefore 3D bf16 — roughly 32 MiB per sample per layer (`seq_len 4096 × hidden 4096 × 2 B`); `nsamples=128` would need ~172 GiB per target, so this example uses 16 (16 samples × 43 layers ≈ 21.5 GiB).

> **Step 3 must reuse the very same `NSAMPLES` value** (the scripts default to 16 consistently): the PTQ cosine LR scheduler sizes its period as `T_max = epochs × (nsamples // cali_bsz)`, while the real number of optimization steps is determined by the samples extracted here. If step 3 falls back to the CLI default `nsamples=128`, the period becomes 8× the real step count, so training ends with the learning rate still at ~96% of `base_lr` and the `lwc/lac` clip factors barely anneal.

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
- `NSAMPLES=16`: must match the value used by `extract_ptq_data.sh` in step 2 (see the note above); otherwise the cosine schedule period no longer matches the real step count.

Both Attention and MoE training can be parallelized across cards by block (e.g. two cards, half the layers each):

```shell
ASCEND_RT_VISIBLE_DEVICES=0 START_BLOCK_IDX=0 END_BLOCK_IDX=22 QUANT_DTYPE=int bash scripts/ptq_attn.sh &
ASCEND_RT_VISIBLE_DEVICES=1 START_BLOCK_IDX=22 END_BLOCK_IDX=43 QUANT_DTYPE=int bash scripts/ptq_attn.sh &
wait

ASCEND_RT_VISIBLE_DEVICES=0 START_BLOCK_IDX=0 END_BLOCK_IDX=22 QUANT_DTYPE=int bash scripts/ptq_moe.sh &
ASCEND_RT_VISIBLE_DEVICES=1 START_BLOCK_IDX=22 END_BLOCK_IDX=43 QUANT_DTYPE=int bash scripts/ptq_moe.sh &
wait
```

When sharding, two constraints are mandatory:

- Every card must reuse the same `PARAM_DIR` / `OUTPUT_DIR` (the script defaults already agree), so that the `layer_{i}_*.pt` files of all shards land in one place for step 4 to load.
- The per-card `[START_BLOCK_IDX, END_BLOCK_IDX)` ranges must be **complementary and non-overlapping**, together covering `[0, 43)` (as in `[0, 22)` and `[22, 43)` above). Only then does each shard write just the files of its own layers.

Counter-example: setting these variables for one card only. The card that has them set runs its sub-range while the other falls back to the script default `0..43` and runs all 43 layers, so the ranges overlap — the overlapping layers get written concurrently into the same `layer_{i}_*.pt` by two processes (`amct_pytorch/workflows/llm_ptq.py` skips a unit merely when its file already exists, and `torch.save` is not atomic), which can leave a truncated param file that only surfaces when step 4 loads it. So when a target is not sharded, leave both cards unset.

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

Evaluated on 2×Atlas A3 (64GB HBM), `seq_len=4096`, WikiText2 test split, Pileval calibration (`nsamples=16`), `base_lr=1e-3`. Weights were dequantized to bf16 at load time per prerequisite (b), which is equivalent to the pre-converted bf16 copy of route (a).

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

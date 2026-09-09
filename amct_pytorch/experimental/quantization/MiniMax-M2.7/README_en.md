# MiniMax-M2.7 OSPlus SmoothQuant Sample

This sample provides a two-stage **OSPlus SmoothQuant** quantization pipeline for
MiniMax-M2.7. Unlike classic SmoothQuant, which only balances the activation/weight
dynamic range via a single `alpha` hyperparameter, OSPlus **re-parameterizes the
per-channel scaling as a single-parameter threshold search** and **directly uses the
per-layer output reconstruction error under W4A4 MXFP4 fake quantization** as the
search objective, so the resulting equivalent scale targets the actual numerical
format used on the deployment side.

The three stages are:

1. `scripts/run_stage1.sh` (calibration & search): run a single-batch eager forward
   with the BF16 model as backend, collect calibration activations at two positions
   via forward hooks — the output of `input_layernorm` (Group 1, for q/k/v projection
   scaling) and the input of `self_attn.o_proj` (Group 2, for the v→o path scaling) —
   then run the OSPlus threshold search layer-by-layer to produce two groups of
   SmoothQuant equivalent scales per layer.
2. `scripts/run_stage2_bf16.sh` (fusion & export): read the original BF16 checkpoint
   and all stage-1 scales, fuse the equivalent scaling layer-by-layer in fp32 and cast
   back to BF16, exporting a **fused BF16 HuggingFace checkpoint** (with no
   `quantization_config`) that has the same shape and dtype as the original model.
3. `scripts/run_stage3_mxfp4.sh` (MXFP4 conversion): read the fused BF16 checkpoint from
   stage 2, apply round-to-nearest (RTN) MXFP4 quantization and packing to the
   quantizable Linear weights, and export a deployable **MXFP4 HuggingFace checkpoint**
   (`config.json` carries a quark-style `quantization_config`; see the appendix for
   Quark-style details).

> The stage-2 fused BF16 checkpoint is mathematically equivalent to the original model
> up to BF16 rounding error, and can serve as a standard BF16 starting point for
> subsequent quantization (MXFP4 / W4A8 / W8A8 / INT4 ...); stage 3 performs the RTN
> weight conversion and packing to MXFP4 on top of it. If you only want the fused BF16
> model, running through stage 2 is sufficient — stage 3 is optional.

## Directory Layout

- `scripts/`: entry scripts (stage-1 calibration & search, stage-2 fusion & export,
  stage-3 MXFP4 conversion)
- `src/`: stage-1 / stage-2 / stage-3 core implementation
  - `stage1_calibrate.py`: load model → one forward to collect activations → per-layer
    OSPlus search → dump scales (channel-wise scale only; no classic OS+
    channel-wise shift)
  - `run_search_from_cache.py`: in stage-1 "record-only" mode, run the scale search
    offline and in parallel from cached activations + safetensors weights
  - `stage2_export_bf16.py`: read scales, fuse in fp32 and export the fused BF16 checkpoint
  - `export_rtn_mxfp4.py`: convert the fused BF16 checkpoint to Quark-style MXFP4 via RTN
    and export (see the appendix for Quark-style details)
  - `mxfp4_fake_quant.py`: the MXFP4 fake-quant operators used by the OSPlus objective
  - `common.py`: architecture constants and helpers used by the MXFP4 export
- `mxfp4_quantizer/`: pure-torch MXFP4 quantize/dequantize aligned with the Ascend
  MXFP4 dequant path
- `requirements.txt`: Python dependencies

## Environment

Recommended container: `quay.io/ascend/vllm-ascend:v0.18.0rc1-a3`

Use the container-bundled `torch` / `torch_npu`; install the remaining Python
dependencies from `requirements.txt`:

```bash
cd /workspace/amct/amct_pytorch/experimental/quantization/MiniMax-M2.7
pip install -r requirements.txt
```

We recommend running inside the container and mounting the following in advance:

- code directory: e.g. mounted at `/workspace/amct`
- BF16 MiniMax-M2.7 model directory: e.g. mounted at `/model/MiniMax-M2.7-bf16`
- calibration data file: e.g. mounted at `/data/minimax/calib.json`

The examples below assume you are already in the sample directory:

```bash
cd /workspace/amct/amct_pytorch/experimental/quantization/MiniMax-M2.7
```

## Required External Inputs

### `MODEL_DIR`

`MODEL_DIR` is the **BF16** MiniMax-M2.7 model directory, shared by stage 1 and stage 2.

- It must contain model weights and configs loadable by `transformers` (including
  `model.safetensors.index.json`, tokenizer, `configuration_minimax_m2.py` /
  `modeling_minimax_m2.py`, etc.)
- Mounting the BF16 model directory directly is recommended

Example:

```bash
export MODEL_DIR=/model/MiniMax-M2.7-bf16
```

### `CALIB_DATA`

`CALIB_DATA` is the `jsonl` calibration corpus used by stage 1: the script reads it
line-by-line, preferring the `messages` field rendered via the chat template; if a line
has no `messages` but a `text` field, `text` is used directly. Single-line example:

```json
{"messages":[{"role":"user","content":"Explain the basics of quantization."}]}
```

## Stage 1: Calibration & OSPlus Search

Stage 1 runs a single-batch eager forward with the BF16 model as backend, hooks two
positions to collect activations — the output of `input_layernorm` (Group 1) and the
input of `self_attn.o_proj` (Group 2) — runs the OSPlus threshold search
layer-by-layer, and dumps each layer's two scale groups (plus an activation cache) to
`RECORD_DIR`.

It supports two working modes (via `RECORD_ONLY`) and resumable runs (via `RESUME`):

- `RECORD_ONLY=1` (script default): **only record and cache activations, skip search**.
  This decouples "one forward to collect activations" from "offline parallel search";
  the built-in multi-worker section (`run_search_from_cache.py` + `numactl` core
  binding) then runs the search in parallel on CPU.
- `RECORD_ONLY=0`: after one forward to collect activations, run the OSPlus search
  **in the same process** via a thread pool.

### Launch Example

```bash
MODEL_DIR=/model/MiniMax-M2.7-bf16 \
CALIB_DATA=/data/minimax/calib.json \
RECORD_DIR=$(pwd)/data/record_data \
NUM_CALIB_DATA=512 \
SEQ_LEN=32768 \
BATCH_SIZE=1 \
MAX_TOKENS_PER_LAYER=4096 \
LOAD_DEVICE_MAP=auto \
RECORD_ONLY=0 \
RESUME=1 \
bash scripts/run_stage1.sh
```

For the "record first, search in parallel later" approach, set `RECORD_ONLY=1` to
record first, then reuse the built-in parallel search section (which binds cores per
`NUM_WORKERS` / `THREADS_PER_WORKER` / `NUMA_NODES` / `CORES_PER_NUMA`).

### Outputs

After stage 1, `RECORD_DIR` contains:

- `layer_{i}_attn_scale.pt`: Group-1 scale, shape `[hidden_size]` (62 layers)
- `layer_{i}_oproj_scale.pt`: Group-2 scale, shape `[num_attn_heads * head_dim]`
  (GQA-folded, 62 layers)
- `activations/layer_{i}_attn_act.pt` / `activations/layer_{i}_oproj_act.pt`: bf16 activation cache
- `metadata.json`: config metadata for this calibration & search run

### Parameters

- `MODEL_DIR`: MiniMax-M2.7 BF16 model directory (**required**).
- `CALIB_DATA`: calibration `jsonl` file path (**required**).
- `RECORD_DIR`: stage-1 output directory, read directly by stage 2.
- `NUM_CALIB_DATA`: number of calibration samples.
- `SEQ_LEN`: truncation length for calibration samples.
- `BATCH_SIZE`: forward batch size.
- `MAX_TOKENS_PER_LAYER`: per-layer token-dimension cap (tail truncation) to bound the
  per-layer activation cache.
- `LOAD_DEVICE_MAP`: model loading strategy, `auto` (auto placement) or `cpu`.
- `RECORD_ONLY`: `1` records activations and skips search; `0` searches in-process after recording.
- `RESUME`: `1` skips layers whose scale files already exist (resumable).
- `NUM_WORKERS` / `THREADS_PER_WORKER` / `NUMA_NODES` / `CORES_PER_NUMA` / `NUM_LAYERS`:
  only used by the parallel search section when `RECORD_ONLY=1`, to control CPU
  parallelism and NUMA core binding.

## Stage 2: Fuse & Export BF16

Stage 2 does no forward validation; it reads all stage-1 scales, fuses the SmoothQuant
equivalent scaling layer-by-layer in fp32 and casts back to BF16, exporting a plain BF16
HuggingFace safetensors checkpoint.

### Launch Example

```bash
MODEL_DIR=/model/MiniMax-M2.7-bf16 \
RECORD_DIR=$(pwd)/data/record_data \
OUTPUT_DIR=$(pwd)/data/exported/MiniMax-M2.7-osplus_sq_fused_bf16 \
bash scripts/run_stage2_bf16.sh
```

### Outputs

After stage 2, `OUTPUT_DIR` contains a standard HuggingFace export:

- `model-00001-of-xxxxx.safetensors` shards
- `model.safetensors.index.json`
- `config.json` (**without** `quantization_config`)
- `generation_config.json`
- tokenizer and modeling files

### Parameters

- `MODEL_DIR`: source BF16 model directory (**required**).
- `RECORD_DIR`: stage-1 scale directory (must contain 62 `*_attn_scale.pt` and 62
  `*_oproj_scale.pt`).
- `OUTPUT_DIR`: output directory for the fused BF16 model; must be empty or not exist.
- `MODEL_FILES_DIR`: source directory for tokenizer / config / modeling files exported
  alongside the weights; defaults to `MODEL_DIR` (the BF16 model directory already
  contains these). Only set it if those files are not in `MODEL_DIR`.

### Resource Requirements

- BF16 reference model under `MODEL_DIR` (MiniMax-M2.7-bf16 is ~427 GB);
- ~430 GB free RAM (the full BF16 model stays resident in CPU memory during fusion);
- ~430 GB free disk under `OUTPUT_DIR`.

## Stage 3: MXFP4 Conversion (RTN)

Stage 3 reads the fused BF16 checkpoint exported by stage 2, applies round-to-nearest
(RTN) MXFP4 quantization and packing to the quantizable Linear weights, and exports a
deployable MXFP4 HuggingFace checkpoint. This step uses **no calibration data**; modules
such as the MoE gate (`*block_sparse_moe.gate*`) and `lm_head` keep their original dtype
and are not quantized.

### Launch Example

The default input is the stage-2 default output directory, so it usually needs no
explicit override:

```bash
BF16_MODEL_DIR=$(pwd)/data/exported/MiniMax-M2.7-osplus_sq_fused_bf16_hf \
OUTPUT_DIR=$(pwd)/data/exported/MiniMax-M2.7-osplus_sq_mxfp4_hf \
QUANT_DEVICE=auto \
bash scripts/run_stage3_mxfp4.sh
```

You can also apply it to any loadable BF16 MiniMax-M2.7 checkpoint (e.g. the original
BF16 model) by pointing `BF16_MODEL_DIR` at that directory.

### Outputs

After stage 3, `OUTPUT_DIR` contains a Quark-style MXFP4 export (see the appendix):

- `model-00001-of-xxxxx.safetensors` shards (quantizable Linear weights stored as
  `*.weight` packed uint8 FP4 payload + `*.weight_scale` uint8 e8m0 scale; other tensors
  keep their original dtype)
- `model.safetensors.index.json`
- `config.json` (**carries** a quark-style `quantization_config`)
- tokenizer and modeling files

### Parameters

- `BF16_MODEL_DIR`: input BF16 checkpoint directory for stage 3; defaults to the stage-2
  default output directory.
- `OUTPUT_DIR`: output directory for the MXFP4 model; must be empty or not exist.
- `LOAD_DEVICE_MAP`: model loading strategy, `cpu` (default) or `auto`.
- `DEVICE_MAP_FILE`: optional device map JSON to explicitly map submodules to devices.
- `QUANT_DEVICE`: device(s) for MXFP4 quantization; `auto` uses available NPU / CUDA
  (falling back to CPU), or a comma-separated list like `npu:0,npu:1` for multi-device
  parallel quantization.
- `MAX_INFLIGHT_JOBS`: max in-flight jobs during multi-device parallel quantization.
- `ROW_CHUNK_SIZE`: per-weight row chunk size for quantization, to bound peak memory.

## Running Tips

- Stage order: stage 1 → stage 2 → (optional) stage 3; stage 2 depends on stage-1
  scales, and stage 3 depends on the stage-2 fused BF16 checkpoint. If you only need the
  fused BF16 model, running through stage 2 is sufficient.
- `SEQ_LEN`, `NUM_CALIB_DATA`, `BATCH_SIZE` significantly affect stage-1 device memory
  usage and runtime; tune them to your device count and memory.
- For the `RECORD_ONLY=1` parallel search section, set `NUM_WORKERS` /
  `THREADS_PER_WORKER` etc. according to the host NUMA topology to avoid cross-NUMA contention.
- Stage 3 can use NPU / CUDA to accelerate quantization (`QUANT_DEVICE`), falling back to
  CPU when no accelerator is available; reduce `ROW_CHUNK_SIZE` if memory is tight.

## OSPlus SmoothQuant Principle

### 1. The Basic Form of SmoothQuant Equivalent Transformation

For any Linear op $Y = X W^\top$ (with $X \in \mathbb{R}^{N \times C_{\text{in}}}$),
introducing a positive vector $s \in \mathbb{R}_{>0}^{C_{\text{in}}}$ along the input
channel dimension yields the strictly equivalent decomposition:

$$Y = X W^\top = \big(X / s\big)\big(W \odot s\big)^\top,$$

where $X / s$ and $W \odot s$ denote per-channel division and multiplication by $s$.

Under the MXFP4 block quantization format, elements within a block share a single E8M0
exponent scale, so an extreme magnitude in a single channel significantly raises that
block's scale and depresses the effective precision of other elements in the same block.
The key observation of SmoothQuant is that the per-channel magnitude distribution of
activations is far more uneven than that of weights; therefore, choosing an appropriate
$s > 1$ to **shrink activations and amplify weights** — without changing the model
function — significantly reduces the combined MXFP4 quantization error of activations
and weights.

This project applies the transformation at two positions, whose insertion points,
scaling dimensions, and absorption directions are shown below.

**Table: The two insertion positions of SmoothQuant equivalent scaling**

| Group | Insertion Position | Scaling Dimension | Absorption Direction | Notes |
|-------|--------------------|-------------------|----------------------|-------|
| Group 1 | between `input_layernorm` and q/k/v projections | hidden dim (`hidden_size = 3072`) | layernorm weight divided per-channel by $s$; q/k/v projection weights multiplied per-input-channel by $s$ | outlier suppression on the Attention QKV input side |
| Group 2 | between v projection and o projection | attention query dim (`num_attn_heads × head_dim = 48 × 128`) | v-proj weight divided on the output-channel dim by a KV-head-granular $s$; o-proj weight multiplied on the input-channel dim by a Q-head-granular $s$ | under the GQA constraint, the 6 repeated query heads within the same KV head must share one scale, otherwise the v side cannot absorb it as a single vector |

Since all of the above is fused in fp32 before returning to BF16, and the GQA folding of
Group 2 is explicitly modeled during the search stage (see below), the entire equivalent
transformation holds strictly in math, introducing only the rounding error of the BF16
representation itself; the fused model's logits should match the original model up to
BF16 rounding precision.

### 2. The OSPlus Search Objective on MXFP4

Directly applying the classic SmoothQuant per-channel scaling parameterized as
$s_i = \max|X_{:,i}|^{\alpha} / \max|W_{i,:}|^{1-\alpha}$ can only balance the overall
dynamic range of activations and weights via the single hyperparameter $\alpha$, and
cannot directly minimize the actual reconstruction error under a W4A4 MXFP4
configuration. OSPlus re-parameterizes this scaling as a **single-parameter threshold
search**: given a threshold $s_t > 0$, the scale for each channel $i$ is built as

$$s_i(s_t) = \max\!\Big( \max\big(\tfrac{c_{\max,i}}{s_t},\, 1\big),\; \max\big(\tfrac{-c_{\min,i}}{s_t},\, 1\big) \Big),$$

where $c_{\max,i}$ and $c_{\min,i}$ are the per-channel max and min of channel $i$ over
the calibration activations. This parameterization only scales channels whose magnitude
significantly exceeds $s_t$, keeping $s_i = 1$ for the rest; the entire candidate scale
is determined by a single scalar $s_t$, so the search space collapses from $C_{\text{in}}$
dimensions to **one**.

Over the threshold $s_t$, OSPlus directly uses the MXFP4-quantized output reconstruction
error as its objective. For the current layer's calibration activation $X$ and weight $W$:

$$\mathcal{L}(s_t) = \frac{1}{N}\sum_{n=1}^{N}\Big\|\,\widetilde{Q}_{a}\!\big(X_n / s(s_t)\big)\;\widetilde{Q}_{w}\!\big(W \odot s(s_t)\big)^{\!\top} - X_n W^{\!\top}\Big\|_2^2,$$

where $\widetilde{Q}_a$ and $\widetilde{Q}_w$ are MXFP4 fake-quant operators along the
reduction axis, using E2M1 element encoding and an E8M0 shared exponent scale, and
implemented with rounding rules strictly consistent with the Ascend MXFP4 GEMM dequant
path (see `src/mxfp4_fake_quant.py` and `mxfp4_quantizer/`). This loss is equivalent to
the layer-level output MSE after chaining SmoothQuant equivalent scaling with W4A4 MXFP4
fake quantization; hence its optimum $s^\star$ directly targets the actual numerical
format on the deployment side rather than merely balancing statistical magnitudes.

During the search, $s_t$ is uniformly sampled into 200 candidates over
$[0.1,\; \max(|c_{\max}|, |c_{\min}|)]$ and traversed in descending order, recording the
$s^\star$ that minimizes $\mathcal{L}(s_t)$. The two scale groups are implemented by a
general searcher (`OSPlusMigrator`) and a GQA-constrained searcher
(`OSPlusMigratorGQA`); the latter additionally applies GQA folding
$\mathrm{fold}(s)_{h,r,d} = \max_{r'} s_{h,r',d}$ at each candidate scale, so that the
"6 repeated query heads within the same KV head share the scale" constraint is already
satisfied when evaluating $\mathcal{L}(s_t)$, avoiding illegal scales that are found by
the search but cannot be absorbed during fusion.

### 3. Implementation of the Calibration & Search Stage (Stage 1)

The calibration & search stage uses the BF16 model as backend, renders the calibration
corpus into input sequences via the chat template, and runs a single-batch eager
forward. During the forward, hooks collect activation samples at two positions
simultaneously: Group-1 activations on the output side of `input_layernorm`, and
Group-2 activations on the input side of the o projection. Since the OSPlus search only
depends on the channel-level magnitude distribution of activations, each layer's
collected results are tail-truncated along the token dimension (`MAX_TOKENS_PER_LAYER`),
which both stabilizes per-channel extremum estimation and bounds the per-layer
activation cache.

All layers' activations are collected in parallel within a single forward pass, avoiding
repeated model loading; then the OSPlus searcher is called layer-by-layer, bounding
concurrency via a semaphore on the working device, ultimately producing two scale
tensors per layer and saving them to `RECORD_DIR`. This stage supports resumable runs
(`RESUME`) and a "record activations only, skip search" mode (`RECORD_ONLY`), used
respectively for failure retries and for offline reuse of cached activations with
parallel search.

### 4. Implementation of the Fusion & Export Stage (Stage 2)

The fusion & export stage reads the original BF16 checkpoint and all scale tensors from
the previous sub-stage (62 layers × 2 groups), fuses them layer-by-layer in fp32 per the
equivalent transformation of Section 1 and casts back to BF16, finally outputting a BF16
HuggingFace checkpoint with the same shape and dtype as the original model.

Here the v projection absorbs both Group-1 and Group-2 scaling: it is first multiplied on
the input-channel dim by the Group-1 scale, then divided on the output-channel dim by a
KV-head-granular vector formed from the first element of each group of the Group-2 scale
along the GQA dimension; the o projection is multiplied on the input-channel dim by the
full Q-head-granular Group-2 scale. GQA folding guarantees that this KV-head-granular
vector and the full Q-head-granular vector are strictly equal across each group of 6
repeats, so the two absorption steps do not conflict.

All arithmetic during fusion is done in fp32 before casting back to BF16, to avoid BF16
accumulated rounding error polluting the precision of the equivalent transformation. The
exported checkpoint carries no `quantization_config`; its external behavior matches the
original BF16 model up to BF16 rounding precision, and it can be consumed directly as a
new BF16 starting point by subsequent quantization-aware training and MSE-guided MXFP4
weight conversion stages, with no special loading logic. This decoupling at the weight
checkpoint level lets this stage serve as a standard BF16 input for later stages while
retaining independent-run and rollback capabilities.

## Preliminary Results

After recording activations and searching scales over ~4906 tokens, the preliminary
W4A4 results of the OSPlus SmoothQuant model are as follows:

| Method | HumanEval+ | GSM8K | MATH500 | LongBench v2 | GPQA Diamond |
|--------|-----------|-------|---------|--------------|--------------|
| OSPlus SmoothQuant | 89.02 | 95.20 | 91.00 | 51.29 | 82.58 |
| M2.7 baseline (BF16) | 91.60 | 95.55 | 91.26 | 54.87 | 88.88 |

The accuracy evaluation above uses the inference framework
[cann-recipes-infer / minimax_m2.5_mxfp4](https://gitcode.com/cann/cann-recipes-infer/tree/master/integration/vllm/minimax_m2.5_mxfp4),
with MXFP4 quantization of the KV cache additionally enabled: KV is quantized token-wise
with MXFP4 group size 32; the first 32 tokens' KV stay in BF16, and subsequent tokens' KV
are quantized to MXFP4; Q and the K cache also apply a block=32 Hadamard rotation to reduce
quantization error.

> Note: the above are preliminary accuracies for OSPlus SmoothQuant equivalent scaling +
> W4A4 MXFP4. Evaluation is affected by sampling temperature, randomness, and inference
> backend implementation differences; reproduced numbers may fluctuate slightly under the
> same configuration. We recommend averaging over multiple runs or reproducing under the
> same framework version.

## Appendix: Quark-style MXFP4 vs Common HF MXFP4 Storage

"Quark-style" refers to the checkpoint packaging convention (tensor naming +
`quantization_config`), not a different numerical encoding. The underlying format remains
OCP MXFP4: E2M1 elements, one shared E8M0 scale per group of 32, and two FP4 values packed
into one `uint8`.

Differences from the common HF / GPT-OSS MXFP4 layout (`quant_method: mxfp4`) are mainly:

| | This sample (Quark-style) | Common HF / GPT-OSS MXFP4 |
|---|---|---|
| Weight tensor name | Still `*.weight` (packed uint8) | `*_blocks` (packed uint8) |
| Scale tensor name | Sibling `*.weight_scale` (uint8 e8m0) | Sibling `*_scales` |
| `quant_method` | `"quark"` | `"mxfp4"` |
| Config schema | Quark nested schema (`global_quant_config`, `export.pack_method=reorder`, `weight_format=real_quantized`, `scale_format=e8m0`, etc.) | Simpler, typically `modules_to_not_convert` + `quant_method` |
| Consumers | Quark / Quark-aligned inference paths (e.g. cann-recipes-infer MiniMax MXFP4) | Transformers MXFP4 / GPT-OSS loaders |

In short: the numerical format is the same; the difference is key naming, config shape, and
who loads it. Stage 3 deliberately matches AMD Quark's HF export, hence "Quark-style".

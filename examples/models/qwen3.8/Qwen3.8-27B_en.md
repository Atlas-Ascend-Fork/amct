# Qwen3.8-27B Quant-Result Artifact and Deployment

[中文](Qwen3.8-27B.md)

This guide covers the complete Qwen3.8-27B flow: BF16 baseline, calibration-data extraction, PTQ artifact publication, reload in a fresh process, deploy-checkpoint export, and inference on Ascend 910C. The `attn-linear` and `mlp` targets are exported separately and both artifacts are loaded for evaluation and deployment.

## 1. Validation scope

The source model is the BF16 [Qwen/Qwen3.8-27B](https://www.modelscope.ai/models/Qwen/Qwen3.8-27B) checkpoint. Its Transformers configuration type is `qwen3_5`, so AMCT commands use `--model_name qwen3_5`.

The text backbone has 64 decoder layers: 48 linear-attention layers and 16 full-attention layers. This example quantizes 496 Linear modules:

| Target | Modules | Count |
| --- | --- | ---: |
| `mlp` | `gate_proj`, `up_proj`, and `down_proj` in every layer | 192 |
| `attn-linear` | Five projections in each linear-attention layer and four in each full-attention layer | 304 |
| Total | Linear modules in the text decoder | 496 |

Embeddings, norms, lm_head, vision modules, MTP, convolutions, attention state updates, and other non-Linear structures remain in their source precision.

## 2. Design choices

### 2.1 Start from BF16 weights

This example uses the BF16 checkpoint rather than Qwen3.8-27B-FP8. PTQ needs weights that have not already been quantized by this workflow to estimate clipping and scales, and BF16 provides a common accuracy baseline. Applying INT8 or INT4 PTQ to an FP8 checkpoint would combine the error from two quantization stages, making it impossible to attribute the measured loss. Converting FP8 storage to BF16 does not recover information removed by FP8 quantization, so it is not treated as an equivalent source.

### 2.2 Use blockwise processing and separate attention from MLP

Keeping all float weights, calibration inputs, and algorithm state resident at once is not practical for a 27B model. Blockwise PTQ constructs one layer, records its statistics and artifact unit, then releases its intermediate objects. Peak memory is therefore determined by one layer rather than the full model.

`attn-linear` and `mlp` use separate artifacts because the extraction and PTQ CLIs process one `quant_target` per run, and the two targets use different input hooks and parameter sets. Combining them would couple two independent failure boundaries and prevent the two NPU jobs from running concurrently. Evaluation and deployment still require both artifacts to bind the same checkpoint and share the same construction config before merging them.

### 2.3 Use `gptq_diag` for the deployable result

MinMax is fast and needs no calibration optimization, but its W8A8 PPL in this run is `7.0270099640`, compared with `6.9715690613` for `gptq_diag`. AWQ gives the lowest W8A8 PPL, `6.9421286583`, but it requires an input-channel transform at runtime. The current compressed-tensors dense Linear schema has no field for that transform, so AWQ supports artifact-reload PPL but not this dense deploy path.

The repository's existing `gptq` keeps a full Hessian, whose compute and state grow quadratically with input width. Across 64 layers and 496 Linear modules, that materially increases calibration memory and time. `gptq_diag` accumulates only the diagonal second moment and searches a clipping factor, so each QuantLinear needs to persist only `clip_factor`. It was selected for the result that must be saved, reloaded, and deployed. The separate name is intentional: the approximation must not be presented as full-Hessian GPTQ.

### 2.4 Use mixed W4A8 rather than uniform W4 or uniform W8

W4 MLP alone remained within the exploratory threshold, but W4 attention produced `PPL=8.0783901215`, and applying W4 to both MLP and attention produced `8.3023319244`. Both exceed the `0.5` selection threshold. Sensitivity runs showed that the main QKV projections account for most of this loss, so they remain W8 while MLP, auxiliary gates, and output projections use W4. The final policy reaches `PPL=7.0721402168`, a `0.1518216133` increase over BF16.

Uniform W8 is more conservative but uses 27.5743 GB of NPU weight memory in this run; mixed W4A8 uses 19.0806 GB. The mixed policy therefore derives its W4 coverage from measured PPL rather than a preset percentage. The runtime fuses `in_proj_qkv` with `in_proj_z`, forcing 48 numerical-W4 modules to use the W8 ABI. They are not labelled as W4-kernel modules because doing so would make metadata disagree with the fused module's parameter contract.

### 2.5 Keep artifacts separate from deploy checkpoints

An artifact stores algorithm results and reconstruction config, not a second copy of the 27B BF16 weights. The same source checkpoint and artifact can therefore be used for PPL and for deploy export, and the algorithm result can be validated before allocating space for another full checkpoint. Writing the final runtime checkpoint directly from PTQ would combine two separate questions -- whether quantization state can be restored and whether one runtime format can load -- into one operation. Keeping the stages separate also leaves room for another deploy backend to consume the same quantization result.

## 3. Validated environment

| Component | PTQ/PPL environment | Deploy runtime environment |
| --- | --- | --- |
| Hardware | 2 x Ascend910_9362 | 1 x Ascend910_9362 |
| Driver / CANN | 25.5.5 / 9.1.0 | 25.5.5 / 9.1.0 |
| PyTorch / torch-npu | 2.7.1 / 2.7.1.post4 | 2.10.0 / 2.10.0.post4 |
| Transformers | 5.12.1 | 5.5.4 |
| vLLM / vLLM-Ascend | - | 0.23.0 / 0.23.0 |

PTQ and deployment may use separate Python environments. Both must load the matching CANN environment, and Transformers must include Qwen3.5 model support. See the repository [README](../../../README_en.md#installation-verification) for AMCT installation.

## 4. Prepare paths

Download the BF16 model from ModelScope:

```bash
python - <<'PY'
from modelscope import snapshot_download

snapshot_download("Qwen/Qwen3.8-27B", local_dir="/data/models/Qwen3.8-27B")
PY
```

From the AMCT repository root, define the paths used below:

```bash
export MODEL_DIR=/data/models/Qwen3.8-27B
export RUN_DIR=outputs/qwen3_8_27b_gptq_diag_mixed_w4a8
export BIT_CONFIG=examples/models/qwen3.8/qwen3_8_27b_mixed_w4a8.yaml

mkdir -p "${RUN_DIR}"
```

The model directory must contain `config.json`, `model.safetensors.index.json`, and every referenced safetensors shard. The artifact binds their SHA-256 values; a different checkpoint requires new artifacts.

## 5. BF16 PPL baseline

```bash
python -m amct_pytorch.eval \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml \
  --seq_len 4096 \
  --seed 0 \
  --output_dir "${RUN_DIR}/bf16_eval"
```

The validated WikiText-2 test result is `PPL=6.920318603515625`.

## 6. Extract calibration data for both targets

`extract_ptq_data` accepts one `quant_target` per run. These commands use eight 512-token samples and extract layer inputs for attention Linear and MLP separately:

```bash
python -m amct_pytorch.extract_ptq_data \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:0 \
  --quant_target attn-linear \
  --nsamples 8 \
  --seq_len 512 \
  --data_dir "${RUN_DIR}/ptq_data/attn-linear" \
  --output_dir "${RUN_DIR}/extract/attn-linear"

python -m amct_pytorch.extract_ptq_data \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:1 \
  --quant_target mlp \
  --nsamples 8 \
  --seq_len 512 \
  --data_dir "${RUN_DIR}/ptq_data/mlp" \
  --output_dir "${RUN_DIR}/extract/mlp"
```

Run them concurrently on two NPUs or sequentially on one. Calibration data and the WikiText-2 test set must remain separate.

## 7. Run PTQ and publish quant-result artifacts

This example uses `gptq_diag`. It accumulates the diagonal of the input second moment, searches a weight-clipping factor in `[0.5, 1.0]` for each output channel, and persists `clip_factor`. `gptq_diag` is a diagonal-Hessian approximation designed for the 27B blockwise flow. The repository's `gptq` entry is the full-Hessian implementation and is a different algorithm.

```bash
python -m amct_pytorch.ptq \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:0 \
  --quant_target attn-linear \
  --quant_dtype int \
  --algos gptq_diag \
  --bit_config "${BIT_CONFIG}" \
  --data_dir "${RUN_DIR}/ptq_data/attn-linear" \
  --end_block_idx 64 \
  --quant_result_dir "${RUN_DIR}/artifacts/attn-linear" \
  --output_dir "${RUN_DIR}/ptq/attn-linear"

python -m amct_pytorch.ptq \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:1 \
  --quant_target mlp \
  --quant_dtype int \
  --algos gptq_diag \
  --bit_config "${BIT_CONFIG}" \
  --data_dir "${RUN_DIR}/ptq_data/mlp" \
  --end_block_idx 64 \
  --quant_result_dir "${RUN_DIR}/artifacts/mlp" \
  --output_dir "${RUN_DIR}/ptq/mlp"
```

Each target produces 64 unit safetensors files and one `quant_manifest.json`. Unit files hold the `clip_factor` values for the QuantLinear modules in that layer. The manifest records the source fingerprint, construction config, unit index, file sizes, file SHA-256 values, and entry digests.

The writer atomically publishes every unit, verifies all units by reading them back, and publishes the manifest atomically and last. A directory with unit files but no `quant_manifest.json` is an interrupted run and is rejected by evaluation and deployment.

## 8. Reload artifacts and evaluate PPL

Start a fresh evaluation process. Quant targets, bit policy, algorithms, and construction parameters are restored from the artifacts rather than from PTQ process memory:

```bash
python -m amct_pytorch.eval \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:0 \
  --eval_mode quant \
  --seq_len 4096 \
  --seed 0 \
  --quant_result_dir \
    "${RUN_DIR}/artifacts/attn-linear" \
    "${RUN_DIR}/artifacts/mlp" \
  --output_dir "${RUN_DIR}/artifact_eval"
```

Before building a quantized block, the reader validates the manifest schema, directory file set, unit hashes and digests, shared configuration across both artifacts, full 64-layer coverage, and the current BF16 checkpoint fingerprint.

## 9. Export the deploy checkpoint

```bash
python -m amct_pytorch.deploy \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:0 \
  --quant_result_dir \
    "${RUN_DIR}/artifacts/attn-linear" \
    "${RUN_DIR}/artifacts/mlp" \
  --output_dir "${RUN_DIR}/deploy"
```

Deployment builds one quantized block at a time from the same BF16 checkpoint, loads each unit's `clip_factor`, produces INT4/INT8 weights and scales, and writes a new sharded checkpoint. Unquantized tensors are copied unchanged, then the weight index and compressed-tensors `quantization_config` are regenerated.

The Qwen3.8 vLLM runtime fuses `in_proj_qkv` and `in_proj_z` into `in_proj_qkvz`. The former is W8 and the latter is W4 in this policy, but one fused module cannot use two parameter ABIs. Therefore, 48 `in_proj_z` modules keep W4 numerical values and scales while using W8 metadata ABI. The exporter reports this fallback explicitly; it is incorrect to claim that all 400 numerical-W4 modules use the W4 kernel.

## 10. vLLM-Ascend smoke test

Load the exported directory in the vLLM-Ascend environment:

```bash
export VLLM_PLUGINS=ascend
export ASCEND_RT_VISIBLE_DEVICES=0

python - <<'PY'
import os
from vllm import LLM, SamplingParams

llm = LLM(
    model=os.path.join(os.environ["RUN_DIR"], "deploy"),
    tensor_parallel_size=1,
    max_model_len=256,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 0, "video": 0},
)
result = llm.generate(
    ["The capital of France is"],
    SamplingParams(temperature=0.0, max_tokens=8),
)[0].outputs[0]
print(result.text)
print(list(result.token_ids))
PY
```

The validated token IDs are `[11751, 13, 198, 760, 6511, 314, 9564, 369]`. This smoke test proves real 910C load and generation, not production throughput, long-context, or concurrency performance.

## 11. Results

All PPL values use the same BF16 checkpoint, WikiText-2 test set, seed 0, and `seq_len=4096`:

| Bit policy | Algorithm | PPL | Delta from BF16 |
| --- | --- | ---: | ---: |
| BF16 | - | 6.9203186035 | 0 |
| W8A8 | MinMax | 7.0270099640 | +0.1066913605 |
| W8A8 | AWQ | 6.9421286583 | +0.0218100548 |
| W8A8 | `gptq_diag` | 6.9715690613 | +0.0512504578 |
| mixed W4A8 (lower W4 coverage) | MinMax | 7.3077268600 | +0.3874082565 |
| mixed W4A8 (lower W4 coverage) | AWQ | 7.2863488197 | +0.3660302162 |
| mixed W4A8 (lower W4 coverage) | `gptq_diag` | 7.0944299698 | +0.1741113663 |
| this mixed W4A8 policy | `gptq_diag` | 7.0721402168 | +0.1518216133 |

This example contains 400 numerical-W4 Linear modules and 96 native-W8 Linear modules. The runtime classification is:

| Runtime class | Count | Meaning |
| --- | ---: | --- |
| Native W4 runtime | 352 | W4 values and the W4 kernel parameter contract |
| W4 values with W8 ABI fallback | 48 | Linear-attention `in_proj_z` |
| Native W8 | 96 | Main QKV and other sensitive projections |

Static validation checked 65 shards and 2799 index keys, with 0 missing, extra, duplicate, or routing errors. All 703 untouched tensors are element-wise identical to the BF16 source. Measured NPU weight memory decreased from 27.5743 GB for W8A8 to 19.0806 GB, a reduction of about 30.803%. The W4 checkpoint is about 0.96% larger on disk because logical INT4 values are stored in INT8 containers and packed to 4 bits by vLLM at load time. The benefit is NPU memory, not checkpoint size.

## 12. Changing the algorithm or bit policy

- Uniform W8A8: set `BIT_CONFIG` to `amct_pytorch/configs/w8a8.yaml` and rerun PTQ, artifact evaluation, and deployment in a new output directory.
- MinMax: change `--algos gptq_diag` to `--algos minmax`.
- AWQ: change it to `--algos awq`. The current compressed-tensors dense Linear schema cannot represent AWQ's input-channel transform, so AWQ supports artifact-reload PPL but not this dense deploy path.
- Full-Hessian GPTQ: use `--algos gptq`. Its state and computational cost differ from `gptq_diag`; it cannot consume the diagonal-approximation artifacts from this example.

Changes to the algorithm, bit policy, source weights, or quantization targets require new artifacts. Do not edit a manifest to reuse incompatible results.

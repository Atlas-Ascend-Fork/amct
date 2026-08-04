# Kimi-VL-A3B-Thinking Quantization on NPU
## Overview
Kimi-VL-A3B-Thinking, released by Moonshot AI, is a multimodal (vision-language) MoE model with 16B total parameters, 2.8B activated per token, and roughly 32GB of BF16 weights. This practice uses the quantization tooling in amct_pytorch to run quantization, data extraction, and PTQ on its **language-side MoE backbone**.

The model has two parts: the MoonViT vision tower and a DeepSeek-V3 style language backbone (MLA attention, 64 routed experts, 2 shared experts, with layer 0 kept dense). This sample quantizes only the feed-forward path of the language backbone; the vision tower and the multimodal projector stay in BF16.

Compared with the other one-stop samples in this repository, the distinguishing property here is that **the entire flow fits on a single card**: 32GB of BF16 weights load onto one Atlas A3 card as a whole model, so the baseline evaluation needs no multi-card sharding.

---

## Hardware Requirements
Product model: Atlas A3 Pod series

Operating system: Linux ARM

Image version: amct_llm_images:v1

Driver version: Ascend HDK 25.5.1
> Use `npu-smi info` to check whether the Ascend NPU firmware and driver are correctly installed, and confirm the version is `25.5.1`. If it is not installed or the version differs, download the [firmware and driver package](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=9.0.0-beta.2&driver=Ascend+HDK+25.5.1) and install it following the [guide](https://hiascend.com/document/redirect/CannCommunityInstSoftware).

---

## One-Stop Platform Guide

The one-stop platform ships a preconfigured runtime. When using it, follow this section instead of the docker-related steps in the standard flow.

- **Model support**: the one-stop platform provides a single-card Atlas A3 environment. At roughly 32GB of BF16 weights, this model fits comfortably on one card.
- **Environment setup**: the runtime is already in place — no docker image to pull, no container to start.
- **CANN path**: CANN is installed at `/home/developer/Ascend/cann`.

> The standard operations in the Quick Start below apply to non-one-stop environments; one-stop users should adjust each step according to the differences above.

---

## Supported Scope

| quant_target | Supported | Notes |
|---|---|---|
| `moe` | Yes | Routed experts (`moe.routed`) and shared experts (`moe.shared`) in layers 1-26 |
| `mlp` | Yes | The dense MLP of layer 0 (`first_k_dense_replace=1`) |
| `attn-linear` | No | The language backbone uses MLA with `q_lora_rank=None` and needs its own attention quant module, which is not adapted yet |
| `attn-cache` | No | Same as above |

Passing `attn-linear` / `attn-cache` raises an explicit error with the message above rather than silently skipping.

---

## Quick Start

### Download the source

  ```shell
  mkdir -p /home/code; cd /home/code/
  git clone https://gitcode.com/cann/amct.git
  cd amct
  ```

### Download the dataset
  The required dataset is downloaded automatically when running eval in amct_pytorch.

### Download the weights

  Download the [moonshotai/Kimi-VL-A3B-Thinking original weights](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking) and place them under a fixed path, for example `/data/models/Kimi-VL-A3B-Thinking`.

  These weights are already `bfloat16`, so unlike FP8/FP4 checkpoints they need no format conversion before the flow below. The model loads with `trust_remote_code`, so the weight directory must contain the official `modeling_kimi_vl.py` and `configuration_kimi_vl.py`.

### Build locally
  For the local build flow see [Installation & Verification](../../../README_en.md#installation-verification)

### Baseline evaluation

After the local build, a baseline run verifies the environment end to end and provides the reference numbers for the direct-quantization and PTQ evaluations that follow.

  ```shell
python -m amct_pytorch.eval \
     --model /data/models/Kimi-VL-A3B-Thinking \
     --model_name kimi_vl \
     --seq_len 4096 \
     --granularity block \
     --device npu:0 \
     --eval_mode bf16 \
     --bit_config amct_pytorch/configs/bf16.yaml
  ```
Key arguments:
- seq_len: input sequence length used for calibration and evaluation; adjust to fit device memory
- granularity: blockwise or modelwise inference; currently block is supported
- eval_mode: in `quant` mode `bit_config` must be set as well; in `bf16` it may be omitted or taken from the sample
- bit_config: quantization configuration file

Baseline accuracy:
`Wikitext2-ppl=30.0658`

> Note: the evaluation dataset is text-only Wikitext2, so this metric reflects the language backbone; the vision side does not participate.
> This value is much higher than that of a text-only model of comparable size (Qwen3.6-35B-A3B scores 6.2825). That is a property of
> the model, not a broken flow: Kimi-VL-A3B-Thinking is trained for multimodal long chain-of-thought with a 163840-token vocabulary,
> so plain-English language-modeling perplexity is expected to be poor. It was cross-checked along a fully independent path — the whole
> model loaded on one card and scored with a standard sliding window — which gives `30.2493`, within 0.6% of the blockwise number here
> (the gap comes from blockwise bf16 accumulation order and last-token handling). Use this baseline to measure quantization degradation;
> do not compare it against the ppl of text-only models.

For more details see [Parameter Description](../../../docs/zh/AMCT_Pytorch_LLM.md#31-通用参数)

### Direct-quantization accuracy evaluation
Run the accuracy test under the bit configuration in the YAML and compare against the baseline. The default scheme applies uniform A8W4 int quantization to `quant-target`:
  ```shell
python -m amct_pytorch.eval \
    --model /data/models/Kimi-VL-A3B-Thinking \
    --model_name kimi_vl \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --eval_mode quant \
    --quant_target moe \
    --quant_dtype int \
    --bit_config amct_pytorch/configs/w4a8.yaml
  ```
Key arguments:
- quant_target: quantization target; here the linear layers inside the MoE experts
- quant_dtype: quantization data format; int and mxfp are currently supported

Direct-quantization accuracy:
`Wikitext2-ppl=29.7433`

> Slightly below the BF16 baseline (30.0658), i.e. A8W4 direct quantization is effectively lossless on this model.
> Perplexity dropping by ~1% after quantization is normal jitter on this model's loss surface, not a sign that quantization
> did nothing — running the same path with `w4a4.yaml` (activations squeezed to 4 bit as well) raises ppl to `78.8130`,
> which shows the quantizers really are acting on the expert linears. Only if your reproduction returns a value essentially
> identical to the baseline should you suspect that `quant_target` failed to match the experts.

For more details see [Parameter Description](../../../docs/zh/AMCT_Pytorch_LLM.md#32-ppl-评估参数)

### PTQ data extraction
Extract the PTQ calibration dataset for the chosen `quant_target`:
  ```shell
python -m amct_pytorch.extract_ptq_data \
    --model /data/models/Kimi-VL-A3B-Thinking \
    --model_name kimi_vl \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/kimi_vl/moe \
    --quant_target moe
  ```
Key arguments:
- data_dir: directory for the extracted data

For more details see [Parameter Description](../../../docs/zh/AMCT_Pytorch_LLM.md#33-数据提取参数)

### Post-Training Quantization
Apply a quantization algorithm to reduce the loss introduced by quantization, using autoround as the example:
#### Single-card environment
  ```shell
python -m amct_pytorch.ptq \
    --model /data/models/Kimi-VL-A3B-Thinking \
    --model_name kimi_vl \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/kimi_vl/moe \
    --quant_dtype int \
    --algos autoround \
    --bit_config amct_pytorch/configs/w4a8.yaml \
    --base_lr 1e-3 \
    --quant_target moe \
    --epochs 10 \
    --output_dir ptq_result/
  ```
Key arguments:
- base_lr: learning rate; tune per model and algorithm
- algos: quantization algorithms; lwc/lac/omniquant/autoround are currently supported
- output_dir: where PTQ results are saved
- epochs: number of iterations; adjust per algorithm and observed gain

> PTQ completes on a single card for this model. To speed training up you may still follow the multi-card script [ptq_multi_npu](../../ptq_multi_npu.sh).

For more details see [Parameter Description](../../../docs/zh/AMCT_Pytorch_LLM.md#35-ptq-参数)

### Direct-quantization accuracy evaluation with PTQ
After PTQ, add the quantization algorithm to the evaluation and compare against both the baseline and the no-algorithm direct-quantization result to verify the gain:
  ```shell
python -m amct_pytorch.eval \
  --model /data/models/Kimi-VL-A3B-Thinking \
  --model_name kimi_vl \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode quant \
  --quant_target moe \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_dtype int \
  --algos autoround \
  --moe_mlp_param_dir ptq_result/ptq_params/kimi_vl/moe
  ```
Key arguments:
- moe_mlp_param_dir: where the algorithm coefficients are stored when `quant_target` is `moe` or `mlp`

### Exporting quantized weights
After all the steps above, export the quantized weights for loading. The resulting weight_map matches the official HuggingFace layout.
  ```shell
python -m amct_pytorch.deploy \
  --model /data/models/Kimi-VL-A3B-Thinking \
  --model_name kimi_vl \
  --granularity block \
  --quant_target moe \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_dtype int \
  --algos autoround \
  --moe_mlp_param_dir ptq_result/ptq_params/kimi_vl/moe \
  --output_dir ./output/Kimi-VL-A3B-Thinking-A8W4-INT
  ```
Key arguments:
- output_dir: where the exported weights are saved

> The export writes through every tensor it did not replace, so the vision tower and the multimodal projector keep their original BF16 values under the official `language_model.` / `vision_tower.` prefixes.

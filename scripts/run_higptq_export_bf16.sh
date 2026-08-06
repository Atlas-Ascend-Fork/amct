#!/bin/bash
source ~/Ascend/ascend-toolkit/set_env.sh

# use conda environment: conda activate amct

# Bakes the HiGPTQ-calibrated weights (produced by run_higptq.sh) into a plain
# BF16 HF checkpoint, deployable as-is on engines without a HiFloat4 kernel
# (e.g. vLLM). Unlike amct_pytorch.deploy, this doesn't need real 4-bit
# packing -- it reuses the same fake-quant simulation run_eval_higptq.sh uses.
#
# Prerequisite: run scripts/run_ptq_extract_data.sh then scripts/run_higptq.sh
# first (both for attn-linear and mlp) so ptq_result/ptq_params/qwen3_5/{attn-linear,mlp}
# below actually exist.

python scripts/export_higptq_bf16.py \
    --model /home/ma-user/work/z84448474/models/Qwen3.5-9B \
    --model_name qwen3_5 \
    --device npu:0 \
    --quant_target attn-linear mlp \
    --quant_dtype hifp \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp \
    --output_dir ptq_result/bf16_export/qwen3_5

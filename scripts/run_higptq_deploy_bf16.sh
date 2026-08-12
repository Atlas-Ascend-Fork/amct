#!/bin/bash
source ~/Ascend/ascend-toolkit/set_env.sh

# use conda environment: conda activate amct

# Two-step alternative to run_higptq_export_bf16.sh: first `amct_pytorch.deploy`
# packs the HiGPTQ-calibrated weights into a real HiFloat4 checkpoint (uint8
# "<name>.weight" + "<name>.scale" pairs, same artifact you'd ship to an engine
# with a HiFloat4 kernel), then
# examples/algorithms/higptq/src/dequantize_higptq_deploy.py unpacks that
# checkpoint back into a plain BF16 HF/vLLM-loadable checkpoint. hif4_unpack is
# the exact inverse of hif4_pack, so the final weights are numerically identical
# to run_higptq_export_bf16.sh's single-script fake-quant path -- this version
# just reuses the standard deploy CLI for the packing step instead of
# re-simulating it, at the cost of an extra on-disk packed copy in between.
#
# Prerequisite: run scripts/run_ptq_extract_data.sh then scripts/run_higptq.sh
# first (both for attn-linear and mlp) so ptq_result/ptq_params/qwen3_5/{attn-linear,mlp}
# below actually exist.

# step 1: pack the calibrated weights into a real HiFloat4 deploy checkpoint
python -m amct_pytorch.deploy \
    --model /home/ma-user/work/z84448474/models/Qwen3.5-9B \
    --model_name qwen3_5 \
    --granularity block \
    --device npu:0 \
    --quant_target attn-linear mlp \
    --quant_dtype hifp \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp \
    --output_dir ptq_result/deploy/qwen3_5

# step 2: unpack the HiFloat4 checkpoint back into plain BF16
python examples/algorithms/higptq/src/dequantize_higptq_deploy.py \
    --input_dir ptq_result/deploy/qwen3_5 \
    --output_dir ptq_result/bf16_export/qwen3_5

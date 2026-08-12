#!/bin/bash
source ~/Ascend/ascend-toolkit/set_env.sh

# use conda environment: conda activate amct

# Bakes plain (uncalibrated) HiFloat4 RTN fake-quant weights into a BF16 HF
# checkpoint, deployable as-is on engines without a HiFloat4 kernel (e.g.
# vLLM). No calibration run needed -- unlike run_higptq_export_bf16.sh, this
# does not require scripts/run_ptq_extract_data.sh / scripts/run_higptq.sh.

python scripts/export_hif4_bf16.py \
    --model /home/ma-user/work/z84448474/models/Qwen3.5-9B \
    --model_name qwen3_5 \
    --device npu:0 \
    --quant_target attn-linear mlp \
    --quant_dtype hifp \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --output_dir ptq_result/hif4_bf16_export/qwen3_5

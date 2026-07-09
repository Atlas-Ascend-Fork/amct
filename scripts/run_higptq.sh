#!/bin/bash
source ~/Ascend/ascend-toolkit/set_env.sh

# use conda environment: conda activate amct

python -m amct_pytorch.ptq \
    --model /home/ma-user/work/z84448474/models/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/attn-linear \
    --quant_dtype hifloat4 \
    --algos higptq \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --quant_target attn-linear \
    --output_dir ptq_result/

#!/bin/bash
source ~/Ascend/ascend-toolkit/set_env.sh
export PILEVAL_LOCAL_PATH="/home/ma-user/work/z84448474/data/pile-val-backup"

# use conda environment: conda activate amct

python -m amct_pytorch.extract_ptq_data \
    --model /home/ma-user/work/z84448474/models/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/attn-linear \
    --quant_target attn-linear

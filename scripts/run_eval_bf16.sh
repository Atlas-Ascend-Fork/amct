#!/bin/bash
source ~/Ascend/ascend-toolkit/set_env.sh
export WIKITEXT2_LOCAL_PATH="/home/ma-user/work/z84448474/data/wikitext"

# use conda environment: conda activate amct

python -m amct_pytorch.eval \
     --model /home/ma-user/work/z84448474/models/Qwen3.5-9B \
     --model_name qwen3_5 \
     --seq_len 4096 \
     --granularity block \
     --device npu:0 \
     --eval_mode bf16 \
     --bit_config amct_pytorch/configs/bf16.yaml

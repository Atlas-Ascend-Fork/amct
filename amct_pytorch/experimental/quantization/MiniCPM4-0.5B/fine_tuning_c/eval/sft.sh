#! /usr/bin/env bash

export NUMEXPR_MAX_THREADS=8
export HF_ALLOW_CODE_EVAL="1"
export PYTHONPATH=$PWD/$PYTHONPATH
export COMPASS_DATA_CACHE=/home/wanghaoyu2/miniconda3/envs/vq/lib/python3.10/site-packages/opencompass/

set -ue

checkpoint_path=""
unpack_path=""
overwrite_config_path=""
log_path="logs"


python scripts/unpack_model.py \
    --model_path $checkpoint_path/checkpoint-$ckpt \
    --output_path $unpack_path \
    --overwrite_config_path $overwrite_config_path/config.json

lm_eval \
    --model hf \
    --model_args pretrained=${unpack_path},dtype=bfloat16,trust_remote_code=True \
    --tasks arc_challenge,arc_easy,boolq,piqa,hellaswag,winogrande,ceval-valid,cmmlu,mmlu,m3ke,openbookqa,wplc \
    --batch_size 16 \
    --output_path $log_path/result.json

lm_eval \
    --model vllm \
    --model_args pretrained=${unpack_path},trust_remote_code=True,dtype=bfloat16,max_model_len=3072,gpu_memory_utilization=0.95 \
    --batch_size 4 \
    --tasks wplc \
    --log_samples \
    --output_path $log_path/result_wplc.json
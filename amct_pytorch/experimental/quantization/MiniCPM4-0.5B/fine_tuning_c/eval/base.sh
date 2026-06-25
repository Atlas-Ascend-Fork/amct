#! /usr/bin/env bash

export NUMEXPR_MAX_THREADS=8
export HF_ALLOW_CODE_EVAL="1"
export PYTHONPATH=$PWD/$PYTHONPATH
export COMPASS_DATA_CACHE=/home/wanghaoyu2/miniconda3/envs/vq/lib/python3.10/site-packages/opencompass/

 
log_path="logs/1.7B/sft_no_init"

#! /usr/bin/env bash

export NUMEXPR_MAX_THREADS=8
export HF_ALLOW_CODE_EVAL="1"
export PYTHONPATH=$PWD/$PYTHONPATH
export COMPASS_DATA_CACHE=/home/wanghaoyu2/miniconda3/envs/vq/lib/python3.10/site-packages/opencompass/

set -ue

model_path=/data/groups/QY_LLM_Other/wanghaoyu/pretrained_models/BitCPM4_1B
log_path="logs/BitCPM4_1B"

python eval/run_compass.py \
    eval/opencompass_config_mmlu.py \
    --hf-path $model_path \
    -w $log_path \
    -a vllm \
    --max-num-workers 2 \
    --generation-kwargs "temperature=0.0"
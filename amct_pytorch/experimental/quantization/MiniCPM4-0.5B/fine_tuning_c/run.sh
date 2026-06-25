#! /usr/bin/env bash

export TOKENIZERS_PARALLELISM=true
export CACHE_BASE_DIR="./.cache"

train_data_config_path="configs/longsft.json"
batch_size=2
gradient_acc_steps=1
lr=1.0e-5
identity=basic
model_name_or_path="/data/groups/QY_LLM_Other/wanghaoyu/exp/tvq/MiniCPM4-1B"
exp_dir="/data/groups/QY_LLM_Other/wanghaoyu/exp/tvq_training/MiniCPM4-1B/debug2"
max_steps=4000
model_max_length=4096

set -ue

if [ -d $exp_dir/$identity/tblogs ]; then
    echo "$exp_dir/$identity/tblogs exists; delete it (y/n)"
    read -r confirm
    if [ "$confirm" = "y" ]; then
        rm -rf $exp_dir/$identity/tblogs
        echo "deleting $exp_dir/$identity/tblogs"
    fi
fi

torchrun \
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 5869 \
    -m train \
    --model_name_or_path $model_name_or_path \
    --train_data_config_path $train_data_config_path \
    --bf16 true \
    --output_dir "$exp_dir/$identity" \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_acc_steps \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 20 \
    --learning_rate $lr \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.1 \
    --stable_ratio 0.4 \
    --lr_scheduler_type "warmup_stable_decay" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --logging_dir "$exp_dir/$identity/tblogs" \
    --dataloader_num_workers 4 \
    --max_grad_norm 1.0 \
    --max_steps $max_steps \
    --model_max_length $model_max_length \

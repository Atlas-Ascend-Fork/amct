#! /usr/bin/env bash 

python convert_ckpt.py \
    --fp16_model_path /data/groups/QY_LLM_Other/wanghaoyu/pretrained_models/MiniCPM4-1B \
    --init_ckpt_path /data/groups/QY_LLM_Other/wanghaoyu/exp/quip_sharp/MiniCPM4-1B/hf_guassian \
    --output_path /data/groups/QY_LLM_Other/wanghaoyu/exp/tvq/MiniCPM4-1B

    
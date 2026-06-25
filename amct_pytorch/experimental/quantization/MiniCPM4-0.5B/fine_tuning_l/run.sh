#! /usr/bin/env bash

export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=true

LCVQ_MODEL_PATH=/data/groups/QY_LLM_Other/wanghaoyu/exp/tvq/MiniCPM4-1B/
DATASET_PATH=/data/groups/QY_LLM_Other/wanghaoyu/resources/datasets/Fineweb
OUTPUT_PATH=/data/groups/QY_LLM_Other/wanghaoyu/exp/tvq_training/MiniCPM4-1B

datasets="shard_1"

python cli/llama_factory_cli.py train \
    ./configs/example.yaml \
    stage=pt \
    model_name_or_path=$LCVQ_MODEL_PATH \
    dataset_dir=$DATASET_PATH \
    output_dir=$OUTPUT_PATH \
    dataset=$datasets \
    max_samples=10000 2>&1 | tee log.txt
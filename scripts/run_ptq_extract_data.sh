#!/bin/bash
source ~/Ascend/ascend-toolkit/set_env.sh
export PILEVAL_LOCAL_PATH="/home/ma-user/work/z84448474/data/pile-val-backup"

# use conda environment: conda activate amct

# --quant_target only accepts a single value per run (LlmExtractPtqDataWorkflow
# raises if len(args.quant_target) != 1). qwen3_5 is a dense model, so the two
# targets we care about are attn-linear and mlp; extract calibration data for
# each into its own --data_dir, then run_higptq.sh consumes them the same way.

# attn-linear
python -m amct_pytorch.extract_ptq_data \
    --model /home/ma-user/work/z84448474/models/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/attn-linear \
    --quant_target attn-linear

# mlp
python -m amct_pytorch.extract_ptq_data \
    --model /home/ma-user/work/z84448474/models/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/mlp \
    --quant_target mlp

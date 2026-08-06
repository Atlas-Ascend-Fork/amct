#!/bin/bash
source ~/Ascend/ascend-toolkit/set_env.sh
export WIKITEXT2_LOCAL_PATH="/home/ma-user/work/z84448474/data/wikitext"

# use conda environment: conda activate amct

# Unlike ptq/extract_ptq_data, eval accepts multiple --quant_target values in
# one run: it loads each target's PTQ params from its own dir independently
# (LlmModel.load_selected_layer_ptq_params) and applies them together. This
# loads the attn-linear and mlp params produced by scripts/run_higptq.sh.
#
# Prerequisite: run scripts/run_ptq_extract_data.sh then scripts/run_higptq.sh
# first (both for attn-linear and mlp) so ptq_result/ptq_params/qwen3_5/{attn-linear,mlp}
# below actually exist.

python -m amct_pytorch.eval \
    --model /home/ma-user/work/z84448474/models/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --eval_mode quant \
    --quant_target mlp attn-linear \
    --quant_dtype hifp \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp

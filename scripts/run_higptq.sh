#!/bin/bash
source ~/Ascend/ascend-toolkit/set_env.sh

# use conda environment: conda activate amct

# --quant_target only accepts a single value per run (LlmPtqWorkflow raises if
# len(args.quant_target) != 1). qwen3_5 is a dense model, so the two targets
# we care about are attn-linear and mlp; each is a separate PTQ run below,
# saved to its own param dir. run_eval_higptq.sh loads both dirs together via
# --attn_linear_param_dir / --moe_mlp_param_dir.

# attn-linear
python -m amct_pytorch.ptq \
    --model /home/ma-user/work/z84448474/models/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/attn-linear \
    --quant_dtype hifp \
    --algos higptq \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --quant_target attn-linear \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --output_dir ptq_result/

# mlp
python -m amct_pytorch.ptq \
    --model /home/ma-user/work/z84448474/models/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/mlp \
    --quant_dtype hifp \
    --algos higptq \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --quant_target mlp \
    --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp \
    --output_dir ptq_result/

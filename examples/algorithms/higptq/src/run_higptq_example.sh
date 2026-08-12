#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

# End-to-end HiGPTQ example matching examples/algorithms/higptq/README.md 3.1-3.4.
# Run this from the repository root:
#   bash examples/algorithms/higptq/src/run_higptq_example.sh

MODEL_PATH="/path/to/Qwen3.5-9B"
MODEL_NAME="qwen3_5"
BIT_CONFIG="amct_pytorch/configs/hifloat4.yaml"

# --quant_target only accepts a single value per run for extract_ptq_data/ptq
# (LlmExtractPtqDataWorkflow / LlmPtqWorkflow raise if len(quant_target) != 1).

# 3.1 extract PTQ calibration data
python -m amct_pytorch.extract_ptq_data \
    --model "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/attn-linear \
    --quant_target attn-linear

python -m amct_pytorch.extract_ptq_data \
    --model "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/mlp \
    --quant_target mlp

# 3.2 run HiGPTQ PTQ (Hessian calibration)
python -m amct_pytorch.ptq \
    --model "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/attn-linear \
    --quant_dtype hifp \
    --algos higptq \
    --bit_config "$BIT_CONFIG" \
    --quant_target attn-linear \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --output_dir ptq_result/

python -m amct_pytorch.ptq \
    --model "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/mlp \
    --quant_dtype hifp \
    --algos higptq \
    --bit_config "$BIT_CONFIG" \
    --quant_target mlp \
    --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp \
    --output_dir ptq_result/

# 3.3 quantization accuracy evaluation (eval accepts multiple --quant_target at once)
python -m amct_pytorch.eval \
    --model "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --eval_mode quant \
    --quant_target mlp attn-linear \
    --quant_dtype hifp \
    --bit_config "$BIT_CONFIG" \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp

# 3.4 export a BF16 deployment checkpoint (deploy -> dequantize)
python -m amct_pytorch.deploy \
    --model "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --granularity block \
    --device npu:0 \
    --quant_target attn-linear mlp \
    --quant_dtype hifp \
    --bit_config "$BIT_CONFIG" \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp \
    --output_dir ptq_result/deploy/qwen3_5

python examples/algorithms/higptq/src/dequantize_higptq_deploy.py \
    --input_dir ptq_result/deploy/qwen3_5 \
    --output_dir ptq_result/bf16_export/qwen3_5

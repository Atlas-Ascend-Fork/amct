#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com

# w4a4 int4 autoround algorithm inference
# python -m amct_pytorch.eval \
#   --trust_remote_code \
#   --model /workspace/models/Qwen3-0.6B \
#   --model_name qwen3 \
#   --device npu:0 \
#   --granularity block \
#   --eval_mode quant \
#   --quant_target attn-linear mlp \
#   --quant_dtype int \
#   --algos autoround \
#   --bit_config configs/w4a4_int_autoround.yaml \
#   --attn_linear_param_dir outputs/ptq_params/qwen3/attn-linear \
#   --moe_mlp_param_dir outputs/ptq_params/qwen3/mlp \
#   --seq_len 4096

# w4a4 mxfp4 autoround algorithm inference
python -m amct_pytorch.eval \
  --trust_remote_code \
  --model /workspace/models/Qwen3-0.6B \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --eval_mode quant \
  --quant_target attn-linear mlp \
  --quant_dtype mxfp \
  --algos autoround \
  --bit_config configs/w4a4_mxfp_autoround.yaml \
  --attn_linear_param_dir outputs/mxfp/ptq_params/qwen3/attn-linear \
  --moe_mlp_param_dir outputs/mxfp/ptq_params/qwen3/mlp \
  --seq_len 4096

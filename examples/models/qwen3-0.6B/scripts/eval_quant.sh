# -*- coding: UTF-8 -*-
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
#!/bin/bash

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
: "${MODEL:?Please set MODEL to the local Qwen3-0.6B weight directory}"

QUANT_DTYPE="${QUANT_DTYPE-mxfp}"

case "${QUANT_DTYPE}" in
  int)
    BIT_CONFIG="configs/w4a4_int_autoround.yaml"
    PARAM_ROOT="./outputs/int/ptq_params/qwen3"
    ;;
  mxfp)
    BIT_CONFIG="configs/w4a4_mxfp_autoround.yaml"
    PARAM_ROOT="./outputs/mxfp/ptq_params/qwen3"
    ;;
  *)
    echo "ERROR: QUANT_DTYPE must be int or mxfp; got '${QUANT_DTYPE}'" >&2
    exit 1
    ;;
esac

# Print the complete evaluation command before execution.
set -x
python -m amct_pytorch.eval \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --eval_mode quant \
  --quant_target attn-linear mlp \
  --quant_dtype "${QUANT_DTYPE}" \
  --algos autoround \
  --bit_config "${BIT_CONFIG}" \
  --attn_linear_param_dir "${PARAM_ROOT}/attn-linear" \
  --moe_mlp_param_dir "${PARAM_ROOT}/mlp" \
  --seq_len 4096

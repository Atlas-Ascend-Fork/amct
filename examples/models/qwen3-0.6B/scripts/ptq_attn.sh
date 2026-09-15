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

: "${MODEL:?Please set MODEL to the local Qwen3-0.6B weight directory}"

QUANT_DTYPE="${QUANT_DTYPE-mxfp}"

case "${QUANT_DTYPE}" in
  int)
    BIT_CONFIG="configs/w4a4_int_autoround.yaml"
    OUTPUT_DIR="./outputs/int"
    ;;
  mxfp)
    BIT_CONFIG="configs/w4a4_mxfp_autoround.yaml"
    OUTPUT_DIR="./outputs/mxfp"
    ;;
  *)
    echo "ERROR: QUANT_DTYPE must be int or mxfp; got '${QUANT_DTYPE}'" >&2
    exit 1
    ;;
esac

# Print the complete PTQ command before execution.
set -x
python -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir outputs/ptq_data/attn-linear \
  --quant_dtype "${QUANT_DTYPE}" \
  --algos autoround \
  --bit_config "${BIT_CONFIG}" \
  --quant_target attn-linear \
  --epochs 15 \
  --base_lr 1e-3 \
  --output_dir "${OUTPUT_DIR}"
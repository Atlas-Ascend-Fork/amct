# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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


#!/bin/bash
INPUT_WEIGHT_PATH=""
OUTPUT_WEIGHT_PATH=""
QUANT_MODE=""
MLA_PARAM_PATH=""
MOE_PARAM_PATH=""

case "${QUANT_MODE,,}" in
    bfloat16)
        echo "Convert to bfloat16 weights..."
        python ./deploy.py \
            --input_weight_path "$INPUT_WEIGHT_PATH" \
            --output_weight_path "$OUTPUT_WEIGHT_PATH"
        ;;
    w8a8c16)
        echo "Convert to w8a8c16 weights..."
        python ./deploy.py \
            --input_weight_path "$INPUT_WEIGHT_PATH" \
            --output_weight_path "$OUTPUT_WEIGHT_PATH" \
            --quant_type "w8a8c16"
        ;;
    w8a8c8)

        echo "Convert to w8a8c8 weights..."
        python ./deploy.py \
            --input_weight_path "$INPUT_WEIGHT_PATH" \
            --output_weight_path "$OUTPUT_WEIGHT_PATH" \
            --quant_type "w8a8c8" \
            --clip \
            --mla_param_path "$MLA_PARAM_PATH" \
            --moe_param_path "$MOE_PARAM_PATH" \
        ;;
    w4a8c8)

        echo "Convert to w4a8c8 weights..."
        python ./deploy.py \
            --input_weight_path "$INPUT_WEIGHT_PATH" \
            --output_weight_path "$OUTPUT_WEIGHT_PATH" \
            --quant_type "w4a8c8" \
            --clip \
            --mla_param_path "$MLA_PARAM_PATH" \
            --moe_param_path "$MOE_PARAM_PATH" \
        ;;
    *)
        echo "Error: Unsupport Quant_mode: $QUANT_MODE"
        echo "Supported Mode: bfloat16, w8a8c16, w8a8c8, w4a8c8"
        exit 1
        ;;
esac

echo "Output path: $OUTPUT_WEIGHT_PATH"
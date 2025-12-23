# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

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
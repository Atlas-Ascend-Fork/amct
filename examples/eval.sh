# -*- coding: UTF-8 -*-
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

#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0

# bf16 inference
python -m amct_pytorch.eval \
  --model /path/to/model \
  --model_name qwen3_5 \
  --device npu:0 \
  --granularity block \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml \
  --seq_len 4096

# replace quant module but turn off quant flag, bf16 inference
python -m amct_pytorch.eval \
  --model /path/to/model \
  --model_name qwen3_5 \
  --device npu:0 \
  --granularity block \
  --eval_mode quant \
  --quant_target mlp attn-linear \
  --bit_config amct_pytorch/configs/w8a8.yaml \
  --seq_len 4096
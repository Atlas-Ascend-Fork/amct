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
python amct_pytorch/cli/llm/deploy.py \
  --model /path/to/model \
  --model_name qwen3_5 \
  --device npu:0 \
  --granularity block \
  --quant_target mlp attn-linear \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/w8a8.yaml \
  --output_dir ./deploy_out


#!/bin/bash
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

python amct_pytorch/cli/llm/deploy.py \
  --trust_remote_code \
  --model /path/to/model \
  --model_name qwen3_6 \
  --device cpu \
  --granularity tensor \
  --quant_layers_config amct_pytorch/configs/deploy/qwen/qwen3_6/qwen3_6_w8a8_int.json \
  --output_dir ./deploy_out

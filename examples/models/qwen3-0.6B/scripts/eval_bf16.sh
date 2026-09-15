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

# Print the complete baseline evaluation command before execution.
set -x
python -m amct_pytorch.eval \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --eval_mode bf16 \
  --bit_config configs/bf16.yaml \
  --seq_len 4096
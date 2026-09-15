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
# 这个文件用于生成校准张量 输出目录是：--data_dir 这个参数
#!/bin/bash

set -euo pipefail

python -m amct_pytorch.extract_ptq_data \
  --trust_remote_code \
  --model /workspace/models/Qwen3-0.6B \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --quant_target attn-linear \
  --nsamples 128 \
  --data_dir outputs/ptq_data/attn-linear

python -m amct_pytorch.extract_ptq_data \
  --trust_remote_code \
  --model /workspace/models/Qwen3-0.6B \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --quant_target mlp \
  --data_dir outputs/ptq_data/mlp \
  --seq_len 4096 \
  --nsamples 128

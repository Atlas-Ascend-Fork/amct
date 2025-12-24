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
model_path=“” # 模型路径
data_path=“” # dump的数据路径
output_path=“” # 量化系数保存路径

NUM_BLOCKS=61
NUM_TASKS=8
NUM_NPUS=8

avg=$((NUM_BLOCKS / NUM_TASKS))
rem=$((NUM_BLOCKS % NUM_TASKS))

start=0

for ((i=0; i<NUM_TASKS; i++)); do
  if [ $i -lt $rem ]; then
    len=$((avg + 1))
  else
    len=$avg
  fi

  end=$((start + len))
  npu_id=$((i % NUM_NPUS))

  echo "Launching task $i: blocks [$start, $end), npu $npu_id"
  ASCEND_RT_VISIBLE_DEVICES=$npu_id python ./main.py \
     --model $model_path \
     --w_bits 8 --a_bits 8 \
     --q_bits 8 --k_bits 8 --v_bits 8 \
     --cali_bsz 1 --epoch 25 --base_lr 1e-2 \
     --lwc --lac \
     --cls c8 \
     --output_dir $output_path --data_dir $data_path \
     --start_block_idx $start --end_block_idx $end --train_mode mla --dev 0 &

  start=$end
done

wait
echo "All tasks launched."
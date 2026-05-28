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
model_path="" # 模型路径
model_name="" # 模型名
data_path="" # dump 的数据路径
output_path="" # 量化系数保存路径

NUM_BLOCKS=61 # 模型有多少个 decoder layers
NUM_TASKS=8  # 同时启动多少个 PTQ 进程
CARDS=(0 1 2 3 4 5 6 7) # 可用的 NPU ID
NUM_NPUS=${#CARDS[@]}   # 可用 NPU 数
LAUNCH_INTERVAL=60 # 相邻任务启动间隔，单位秒

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
  logic_id=${CARDS[$npu_id]}

  echo "Launching task $i: blocks [$start, $end), npu $logic_id"
  ASCEND_RT_VISIBLE_DEVICES=$logic_id python -m amct_pytorch.ptq \
      --model $model_path \
      --model_name $model_name \
      --data_dir $data_path \
      --device npu:0 \
      --granularity block \
      --start_block_idx $start \
      --end_block_idx $end \
      --quant_target moe \
      --quant_dtype int \
      --bit_config amct_pytorch/configs/w4a4.yaml \
      --algos learnable_had \
      --output_dir $output_path &

  start=$end
  if [ $i -lt $((NUM_TASKS - 1)) ]; then
    sleep $LAUNCH_INTERVAL
  fi
done

wait
echo "All tasks launched."

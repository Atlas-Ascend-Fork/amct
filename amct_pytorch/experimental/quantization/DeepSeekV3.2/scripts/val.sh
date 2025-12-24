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


model_path=“” # 模型路径
output_path=“” # 推理中间结果，推理完可删除
wikitext_out=“” # 网络最后输出结果，用于计算ppl
mla_param_dir=“” # c8量化系数路径
moe_param_dir=“” # moe量化系数路径


python3 ./eval.py \
    --a_bits 8 \
    --w_bits 4 \
    --seq_len 4096 \
    --cls c8 \
    --model $model_path \
    --train_mode block \
    --output_dir $output_path \
    --wikitext_final_out $wikitext_out \
    --lac --lwc \
    --start_block_idx 0 --end_block_idx 61 \
    --mla_param_dir $mla_param_dir \
    --moe_param_dir $moe_param_dir


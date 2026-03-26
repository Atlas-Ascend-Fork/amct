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
import torch

from amct_pytorch.quantize_op.utils import get_weight_min_max_by_granularity, calculate_scale_offset


WEIGHT_SCALE_FACTOR = 16.0
ACTUAL_PERCENTILE = 0.99
PREVIOUS_PERCENTILE = 0.01


def calculate_scale_offset_by_granularity(weight, quant_config):
    """
    Function: calculate scale and offset by granularity for percentile quantization
    Args:
        weight: torch.tensor, weight data
        quant_config: dict, quantization configuration
    Returns:
        scale_w: torch.tensor, weight scale
        offset_w: torch.tensor, weight offset
    """
    weight_min, weight_max = get_weight_min_max_by_granularity(weight, quant_config)
    
    weight_max = torch.clamp(weight_max, max=WEIGHT_SCALE_FACTOR)
    weight_min = torch.clamp(weight_min, min=-WEIGHT_SCALE_FACTOR)
    
    scale_w, offset_w = calculate_scale_offset(weight_max, weight_min, 
                                               quant_config.get('weights_cfg').get('symmetric'), 
                                               quant_config.get('weights_cfg').get("quant_type"))

    return scale_w, offset_w


def update_percentile_stats(current_max, current_min, previous_max, previous_min):
    """
    Function: update percentile statistics using exponential moving average
    Args:
        current_max: torch.tensor, current batch max
        current_min: torch.tensor, current batch min
        previous_max: torch.tensor, previous accumulated max
        previous_min: torch.tensor, previous accumulated min
    Returns:
        new_max: torch.tensor, updated max
        new_min: torch.tensor, updated min
    """
    if previous_max is None:
        return current_max, current_min
    
    new_max = current_max * ACTUAL_PERCENTILE + previous_max * PREVIOUS_PERCENTILE
    new_min = current_min * ACTUAL_PERCENTILE + previous_min * PREVIOUS_PERCENTILE
    
    return new_max, new_min

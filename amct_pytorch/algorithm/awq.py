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

from amct_pytorch.utils.quant_util import pad_zero_by_group
from amct_pytorch.quantize_op.utils import convert_to_dst_shape
from amct_pytorch.quantize_op.utils import get_weight_min_max_by_granularity, calculate_scale_offset
from amct_pytorch.utils.quant_util import convert_to_per_group_shape, quant_dequant_tensor
from amct_pytorch.utils.vars import MXFP4_E2M1

TOKENS_NUM = 512
MAX_SHRINK = 0.5
GRIDS_NUM = "grids_num"


def process_weights_for_layers(layers, scale_awq, quant_config):
    """
    Function: processing awq quantization of weights for layers list
    Args:
        layers: list of torch.nn.module
        scale_awq: awq scale
        quant_config: quant parameters
    """
    wts_type = quant_config.get('weights_cfg').get("quant_type")
    wts_granularity = quant_config.get('weights_cfg').get("strategy")
    group_size = quant_config.get('weights_cfg').get("group_size")
    for layer in layers:
        layer.weight.data = layer.weight.data.mul(scale_awq)
        if wts_type in (MXFP4_E2M1,):
            layer.weight.data = quant_dequant_tensor(layer.weight.data, wts_type, group_size=group_size)
        else:
            scale, offset = calculate_scale_offset_by_granularity(layer.weight.data, quant_config)
            layer.weight.data = quant_dequant_tensor(layer.weight.data, wts_type, scale, offset, group_size)
        layer.weight.data = layer.weight.data / scale_awq


def search_scale(inputs, layers, block, quant_config, kwargs=None):
    """
    Function: use AWQ search scale
    Args:
        inputs: torch.tensor
        layers: list of torch.nn.module
        block: evaluate unit
        quant_config: quant parameters
    Returns:
        scale: torch.tensor
    """
    algo_params = quant_config.get("algorithm")
    wts_type = quant_config.get('weights_cfg').get("quant_type")

    if not torch.isfinite(inputs).all():
        raise RuntimeError("Run AWQ error! Invalid value(nan or inf) in activation!")
    for layer in layers:
        if not torch.isfinite(layer.weight.data).all():
            raise RuntimeError("Run AWQ error! Invalid value(nan or inf) in weight!")

    kwargs = dict() if kwargs is None else kwargs
    with torch.no_grad():
        ori_out = block(inputs, **kwargs)
        if isinstance(ori_out, tuple):
            ori_out = ori_out[0]

    min_loss = torch.tensor(float('inf'))
    best_scale = None
    inputs_mean = inputs.abs().contiguous().view(-1, inputs.shape[-1]).mean(0)  # (cin)
    grids_num = algo_params.get('awq').get('grids_num')
    ori_state = {name: tensor.detach().cpu() for name, tensor in block.state_dict().items()}
    for grid in range(grids_num):
        ratio = grid / grids_num
        scale_awq = inputs_mean.pow(ratio).clamp(min=1e-4).to(torch.float64).view(1, -1) # (1, cin)
        scale_awq = scale_awq / (scale_awq.max() * scale_awq.min()).sqrt()
        scale_awq = torch.clamp(scale_awq.to(inputs.dtype), 1e-4, torch.finfo(inputs.dtype).max)
        scale_awq = scale_awq.to(next(block.parameters()).device)
        process_weights_for_layers(layers, scale_awq, quant_config)

        quant_out = block(inputs, **kwargs)
        if isinstance(quant_out, tuple):
            quant_out = quant_out[0]

        loss = (ori_out - quant_out).float().pow(4).mean()
        if not torch.isnan(loss) and loss < min_loss:
            min_loss = loss
            best_scale = scale_awq

        block.load_state_dict(ori_state)

    if best_scale is None or torch.isinf(min_loss):
        raise RuntimeError("Run AWQ error! Got invalid loss.")

    return best_scale


def apply_scale(scale_awq, ori_module, input_data):
    """
    Function: apply scale to weight and activation
    Args:
        scale_awq: torch.tensor, awq algo scale factor
        ori_module: original module
        input_data: input data
    """
    scale = scale_awq.to(ori_module.weight.device)
    ori_module.weight.data.mul_(scale)
    input_data.div_(scale)


def calculate_scale_offset_by_granularity(weight, quant_config):
    weight_min, weight_max = get_weight_min_max_by_granularity(weight, quant_config)
    scale_w, offset_w = calculate_scale_offset(weight_max, weight_min, quant_config.get('weights_cfg').get('symmetric'), 
                                               quant_config.get('weights_cfg').get("quant_type"))

    return scale_w, offset_w

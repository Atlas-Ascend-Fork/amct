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
import torch

from amct_pytorch.utils.vars import HIFLOAT8
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.utils.quant_util import convert_dtype, quant_tensor
from amct_pytorch.quantize_op.utils import get_weight_min_max_by_granularity, process_scale


class NpuHIF8CastLinear(torch.nn.Module):
    """
    Function: quantization operator to cast weight/activation to low precision (HIF8)
    Features: 
        1. Weight: direct cast to target dtype without calibration
        2. Activation: insert cast operator in forward for online conversion
        3. Support HIF8 data types
    APIs: forward.
    """
    def __init__(self, ori_module, layer_name, quant_config):
        """
        Function: init objective.
        Args:
            ori_module: torch module. Quantized module instance.
            layer_name: str. Original module's name.
            quant_config: dict. Calibration algorithm parameters.
                          - weights_cfg: weight quantization config
                          - inputs_cfg: activation quantization config
        """
        super().__init__()
        self.ori_module_type = type(ori_module).__name__
        self.ori_dtype = ori_module.weight.dtype
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        self.device = ori_module.weight.device
        self.weight = ori_module.weight
        self.layer_name = layer_name
        self.act_quant_type = quant_config.get('inputs_cfg').get('quant_type')
        self.weight_compress_only = True
        if quant_config.get('inputs_cfg').get('enable_quant') is None or \
            quant_config.get('inputs_cfg').get('enable_quant') == True:
            self.weight_compress_only = False
        
        weight_min, weight_max = get_weight_min_max_by_granularity(self.weight, quant_config)
        scale_w = (weight_max / 16).to(torch.float32)
        scale_w, _ = process_scale(scale_w, None, quant_config.get('weights_cfg').get('symmetric'))
        scale_w = scale_w.reshape(-1, 1) if len(scale_w.shape) == 1 else scale_w
        scale_w = scale_w.repeat(self.weight.shape[0], 1) if scale_w.shape[0] == 1 else scale_w
        scale_w = scale_w.reshape(-1)
        weight, _ = quant_tensor(self.weight.transpose(1, 0), self.wts_type, scale=scale_w)
        self.register_buffer('quantized_weight', weight.to(self.device))
        self.register_buffer('scale_w', scale_w)

        if ori_module.bias is not None:
            bias = ori_module.bias
            if self.act_quant_type == HIFLOAT8:
                bias = bias.to(torch.float32) / scale_w
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x):
        """
        Function: NpuHIF8CastLinear forward function (core feature 3: activation online cast)
        Args:
            x: torch.tensor. Input data for forward computation.
        Return:
            torch.tensor. Output after activation cast and original module forward.
        """
        import torch_npu
        if self.weight_compress_only:
            # input shape reshape to 2d for npu op
            ori_shape = x.shape
            x = x.reshape(-1, x.shape[-1]).to(self.device)
            output = torch_npu.npu_weight_quant_batchmatmul(x, self.quantized_weight,
                self.scale_w.to(x.dtype), bias=self.bias, weight_dtype=torch_npu.hifloat8)
            output = output.reshape(*ori_shape[:-1], -1)
        else: 
            quantized_x = convert_dtype(x, self.act_quant_type)
            output = torch_npu.npu_quant_matmul(quantized_x, self.quantized_weight, self.scale_w,
                x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8,
                pertoken_scale=None, bias=self.bias, output_dtype=x.dtype)
            LOGGER.logd(f"Cast activation of layer '{self.layer_name}' to {self.act_quant_type} success",
                'NpuHIF8CastLinear')
        return output

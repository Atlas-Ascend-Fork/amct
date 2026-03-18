# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
import torch
from torch import nn
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.quantize_op.utils import calculate_scale_offset
from amct_pytorch.utils.data_utils import check_linear_input_dim
from amct_pytorch.utils.vars import HIFLOAT8
from amct_pytorch.utils.log import LOGGER


class HIF8CastQuant(BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the HIF8 cast quantization class.
    This implements offline weight quantization to HIF8 and online activation clip+cast.
    APIs: forward
    """
    def __init__(self,
                 ori_module,
                 layer_name,
                 quant_config):
        """
        Function: init objective.
        Args:
            ori_module: torch module. Original Linear module.
            layer_name: ori_module's name.
            quant_config: calibration algorithm parameters.
        """
        super().__init__(ori_module, layer_name, quant_config)
        self.ori_module_type = type(ori_module).__name__
        self.weight = ori_module.weight
        self.bias = ori_module.bias
        self.layer_name = layer_name
        self.ori_module = ori_module
        
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        if self.wts_type != HIFLOAT8:
            raise RuntimeError(f'HIF8CastQuant only supports hifloat8 weight type, got {self.wts_type}')
        
        self.wts_symmetric = quant_config.get('weights_cfg').get('symmetric', True)
        self.weight_granularity = quant_config.get('weights_cfg').get('strategy', 'channel')
        
        self.scale_w, self.offset_w = self._calculate_weights_scale_factor(self.weight.data)
        
        self.act_type = quant_config.get('inputs_cfg').get('quant_type', HIFLOAT8)
        if self.act_type != HIFLOAT8:
            LOGGER.logw(f'HIF8CastQuant recommends hifloat8 activation type, got {self.act_type}')
        
        LOGGER.logd(f'HIF8CastQuant initialized for layer {layer_name} with weight shape {self.weight.shape}')
    
    def _calculate_weights_scale_factor(self, weight_data):
        """
        Function: calculate weights's quant factor for HIF8
        Args:
            weight_data: torch.tensor, weight data
        Returns:
            scale_w: torch.tensor, scale factor for weight
            offset_w: torch.tensor, offset factor for weight
        """
        if self.weight_granularity == 'tensor':
            weight_max = torch.max(weight_data)
            weight_min = torch.min(weight_data)
        elif self.weight_granularity == 'channel':
            weight_max = torch.max(weight_data, dim=1, keepdim=True)[0]
            weight_min = torch.min(weight_data, dim=1, keepdim=True)[0]
        else:
            raise RuntimeError(f'HIF8CastQuant only supports tensor/channel granularity, got {self.weight_granularity}')
        
        scale_w, offset_w = calculate_scale_offset(weight_max, weight_min, 
                                                 self.wts_symmetric, self.wts_type)
        return scale_w, offset_w
    
    def forward(self, inputs):
        """
        Function: HIF8CastQuant forward function.
        Implements online activation clip+cast during calibration.
        Args:
            inputs: data used for calibration in torch.tensor (BF16/FP16)
        Returns:
            torch.tensor: output in BF16/FP16
        """
        check_linear_input_dim(inputs)
        
        import torch_npu
        
        ori_shape = inputs.shape
        inputs = inputs.to(self.weight.device)
        
        inputs_reshaped = inputs.reshape(-1, inputs.shape[-1])
        
        clip_min = torch.tensor(-240.0, device=inputs.device, dtype=inputs.dtype)
        clip_max = torch.tensor(240.0, device=inputs.device, dtype=inputs.dtype)
        
        clipped_inputs = torch.clamp(inputs_reshaped, clip_min, clip_max)
        
        cast_inputs = torch_npu.npu_dtype_cast(clipped_inputs, dtype=torch_npu.hifloat8)
        
        fp_out = self.ori_module(inputs)
        
        return fp_out
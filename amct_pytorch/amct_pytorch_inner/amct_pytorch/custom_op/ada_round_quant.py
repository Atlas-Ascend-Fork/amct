#!/usr/bin/env python3
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
from torch import nn
import torch.nn.functional as F

from ...amct_pytorch.ada_round.kl_divergence import find_scale_and_offset_by_kl

GAMMA = -0.1
ZETA = 1.1
ANNEAL_COEFFICIENT = 0.5


class AdaRoundQuant(torch.nn.Module):
    """
    Function: Run AdaRoundQuant process for quantization of the given layer.
    APIs: forward, get_quantize_weight, get_scale_offset
    """
    def __init__(self, module, wts_param, data_tensor, tensor_balance_factor):
        '''
        Function:
        init AdaRoundQuant

        Arguments:
        module: torch.nn.Module, Modules to be quantized by the adaround.
        wts_param: Weight Quantization Parameter.
        data_tensor: Weight to be quantified.
                If the DMQ feature is enabled, DMQ conversion is performed. 
                If module is a deconv operator, Adjust ConvTranspose weight shape to fit group param is performed. 
        tensor_balance_factor: for DMQ feature. Adjust the activation value

        Return: None
        '''
        super().__init__()
        self.ada_round_optimize = False
        self.wts_param = wts_param
        self.module = module
        self.weight = module.weight
        self.tensor_balance_factor = tensor_balance_factor
        self.scale, self.offset = find_scale_and_offset_by_kl(data_tensor, self.wts_param['channel_wise'])
        self.alpha = nn.Parameter(self._init_alpha(), requires_grad=True)

    def forward(self, inputs):
        """
        Function: If ada_round_optimize is true, fixed-point inference is performed. 
                  If ada_round_optimize is false, floating-point inference is performed.
        """
        # dmq
        if self.tensor_balance_factor is not None:
            self.tensor_balance_factor = self.tensor_balance_factor.to(inputs.device)
            inputs = inputs / self.tensor_balance_factor
        
        # adjust weight
        if self.ada_round_optimize:
            weight_data = self.get_quantize_weight().to(self.weight.device)
        else:
            weight_data = self.weight.data

        if type(self.module).__name__ == 'Conv2d':
            output = F.conv2d(inputs, weight_data, self.module.bias, 
                              self.module.stride, self.module.padding, self.module.dilation, self.module.groups)
        elif type(self.module).__name__ == 'ConvTranspose2d':
            output = F.conv_transpose2d(inputs, weight_data, self.module.bias, self.module.stride, 
                                        self.module.padding, self.module.output_padding, 
                                        self.module.groups, self.module.dilation)
        elif type(self.module).__name__ == 'Linear':
            output = F.linear(inputs, weight_data, self.module.bias)

        return output

    def get_quantize_weight(self):
        """
        Function: Obtains the weight with quantization loss.
        """
        h_alpha = torch.clamp(torch.sigmoid(self.alpha) * (ZETA - GAMMA) + GAMMA, 0, 1)

        # symmetric quant & unsigned quantize
        weight_q = torch.floor(self.weight / self.scale)
 
        # optimize weight by h_alpha
        weight_q = torch.clamp(weight_q + h_alpha, -128, 127)
        quantize_weight = ((weight_q) * self.scale).to(dtype=self.weight.data.dtype)

        return quantize_weight
    
    def get_scale_offset(self):
        """
        Function: Obtains the quantization factor list.
        """
        # self.scale shape:(input_channel, output_channel/groups, 1, 1)
        # --> Target shape: (output_channel)
        if isinstance(self.module, nn.ConvTranspose2d) and self.module.groups > 1 and self.wts_param['channel_wise']:
            step = self.scale.shape[0] // self.module.groups
            scale_list = self.scale[::step].reshape([-1]).cpu().numpy().tolist()
        else:
            scale_list = self.scale.reshape([-1]).cpu().numpy().tolist()
        offset_list = self.offset.reshape([-1]).cpu().numpy().tolist()
        return scale_list, offset_list

    def _init_alpha(self):
        """
        Function: Initializes the alpha factor.
        """
        # ConvTranspose2d weight has adjusted before
        if isinstance(self.module, nn.Conv2d):
            self.scale = self.scale.reshape([-1, 1, 1, 1])
        if isinstance(self.module, nn.ConvTranspose2d):
            self.scale = self.scale.reshape([1, -1, 1, 1])

            # prechannel weight conversion Process: (in_channel, out_channel/groups, kern[0], kern[1])
            # --> (groups, in_channel/groups, out_channel/groups, kern[0], kern[1])
            # --> (groups, out_channel/groups,  in_channel/groups, kern[0], kern[1])
            # --> (out_channel, in_channel/groups, kern[0], kern[1])
            # So the scale original shape: (1, output_channel, 1, 1)
            # scale target shape: (input_channel, output_channel/groups, 1, 1)
            if self.module.groups > 1 and self.wts_param['channel_wise']:
                in_channel = self.module.weight.shape[0]
                out_channel_div_groups = self.module.weight.shape[1]
                self.scale = self.scale.reshape(self.module.groups, out_channel_div_groups, 1, 1) \
                             .repeat_interleave(in_channel // self.module.groups, dim=0)

        w_floor = torch.floor(self.weight / self.scale)
        w_quant = self.weight / self.scale
        diff = w_quant - w_floor
        alpha = -torch.log((ZETA - GAMMA) / (diff - GAMMA) - 1)
        return alpha

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
from torch import nn
import torch.nn.functional as F
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.utils.data_utils import check_linear_input_dim
from amct_pytorch.quantize_op.utils import calculate_scale_offset
from amct_pytorch.quantize_op.utils import calculate_progressive_weights_scale_factor
from amct_pytorch.utils.vars import FLOAT8_E4M3FN, FLOAT4_E2M1
from amct_pytorch.utils.log import LOGGER


class SmoothQuant(BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the SmoothQuant class.
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 layer_name,
                 quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        layer_name: ori_module's name.
        quant_config: calibration algorithm parameters.
        """
        super().__init__(ori_module, layer_name, quant_config)
        self.ori_module_type = type(ori_module).__name__
        self.weight = ori_module.weight
        self.bias = ori_module.bias
        self.layer_name = layer_name
        self.quant_config = quant_config
        self.group_size = quant_config.get('group_size')
        self.batch_num = quant_config.get('batch_num')
        self.act_granularity = quant_config.get('inputs_cfg').get('strategy')
        self.wts_granularity = quant_config.get('weights_cfg').get('strategy')
        self.act_symmetric = quant_config.get('inputs_cfg').get('symmetric')
        self.wts_symmetric = quant_config.get('weights_cfg').get('symmetric')
        self.act_type = quant_config.get('inputs_cfg').get('quant_type')
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        self.group_size = quant_config.get('weights_cfg').get('group_size', None)
        self.cur_batch = 0
        self.device = ori_module.weight.device

        self.data_max = torch.ones((1, self.weight.shape[-1]), device=self.weight.device, \
            dtype=self.weight.dtype) * torch.inf * -1
        self.data_min = torch.ones((1, self.weight.shape[-1]), device=self.weight.device, \
            dtype=self.weight.dtype) * torch.inf
        self.batch_input = None
        self.scale_w1 = None
        self.scale_w2 = None

    @torch.no_grad()
    def forward(self, inputs):
        """
        Function: SmoothQuant forward function.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        check_linear_input_dim(inputs)
        inputs = inputs.to(self.weight.device)
        output = F.linear(inputs, self.weight, self.bias)

        self.cur_batch += 1
        if self.cur_batch > self.batch_num:
            return output
        batch_max = inputs.reshape(-1, inputs.shape[-1]).amax(dim=0, keepdim=True)
        batch_min = inputs.reshape(-1, inputs.shape[-1]).amin(dim=0, keepdim=True)
        if self.act_granularity == 'token':
            if self.batch_input is None:
                self.batch_input = inputs.cpu()
            else:
                self.batch_input = torch.cat((self.batch_input, inputs.cpu()), dim=-1)

        # update data_max and data_min to calculate smooth factor
        self.data_max = self.data_max.to(inputs.device)
        self.data_min = self.data_min.to(inputs.device)
        self.data_max = torch.where(batch_max > self.data_max, batch_max, self.data_max)
        self.data_min = torch.where(batch_min < self.data_min, batch_min, self.data_min)

        if self.cur_batch == self.batch_num:
            act_max = torch.max(self.data_max.abs(), self.data_min.abs())
            smooth_factor = self.calculate_smooth(act_max)
            weight = self.weight * smooth_factor

            # only FP8 * FP4 do progressive scale
            if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
                self.scale_w1, self.scale_w2 = calculate_progressive_weights_scale_factor(
                    weight.data, group_size=self.group_size)
            else:
                self.scale_w, self.offset_w = self.calculate_weights_scale_factor(weight, \
                    self.quant_config.get('weights_cfg').get('strategy'))

            if self.act_granularity == 'tensor':
                scale_d, offset_d = self._calculate_per_tensor_params(smooth_factor)
            elif self.act_granularity == 'token':
                scale_d, offset_d = self._calculate_per_token_params(smooth_factor)

            self.scale_d = scale_d
            self.offset_d = offset_d
            self.scale = smooth_factor

            LOGGER.logd("Calculate smooth quant params of layer '{}' success!".format(self.layer_name), 'SmoothQuant')
        return output

    def calculate_smooth(self, act_max):
        """
        Function: smooth the input tensor by smooth_strength
        Parameters: inputs: max activation per channel
        Return: output_tensor: smooth factor
        """

        # weight shape is [cout, cin], wts_max shape is [1,cin]
        wts_max = self.weight.abs().amax(dim=0, keepdim=True)

        smooth_strength = self.quant_config.get('algorithm').get('smoothquant').get('smooth_strength')
        smooth_factor = (act_max ** smooth_strength) / (wts_max ** (1 - smooth_strength))
        # set smooth_factor to 1 if nan or inf or 0
        zero_mask = smooth_factor < torch.finfo(act_max.dtype).eps
        invalid_mask = ~torch.isfinite(smooth_factor)
        if zero_mask.any().item() or invalid_mask.any().item():
            smooth_factor = torch.where(zero_mask | invalid_mask,
                torch.tensor(1.0, dtype=smooth_factor.dtype, device=smooth_factor.device), smooth_factor)
            LOGGER.logd("The smooth factor is calculated as abnormal and set to 1")

        return smooth_factor

    def calculate_weights_scale_factor(self, weight, weight_granularity):
        """
        Function: calculate weights's quant factor
        Parameters: 
        weight: weight data
        weight_granularity: weight granularity
        """
        weight_data = weight.data

        if weight_granularity == 'channel':
            weight_max = weight_data.max(dim=1, keepdim=True).values
            weight_min = weight_data.min(dim=1, keepdim=True).values
        elif weight_granularity == 'tensor':
            weight_max = weight_data.max().reshape(1, 1)
            weight_min = weight_data.min().reshape(1, 1)
        scale_w, offset_w = calculate_scale_offset(weight_max, weight_min, 
                                            self.wts_symmetric, self.wts_type)

        return scale_w, offset_w

    def _calculate_per_token_params(self, smooth_factor):
        """
        Function: calculate per token scale_d and offset_d
        Parameters: 
        smooth_factor: smooth factor
        """
        smooth_input = self.batch_input / smooth_factor.cpu().repeat(1, self.batch_num)
        cout_axis = -2
        axis_list = list(range(0, smooth_input.dim()))
        axis_list.pop(cout_axis)
        axis = tuple(axis_list)
        per_token_max = torch.amax(smooth_input, dim=axis, keepdim=True)
        per_token_min = torch.amin(smooth_input, dim=axis, keepdim=True)
        return calculate_scale_offset(per_token_max, per_token_min, self.act_symmetric, self.act_type)

    def _calculate_per_tensor_params(self, smooth_factor):
        """
        Function: calculate per tensor scale_d and offset_d
        Parameters: 
        smooth_factor: smooth factor
        """
        max_val = (self.data_max / smooth_factor).max().reshape(-1)
        min_val = (self.data_min / smooth_factor).min().reshape(-1)
        return calculate_scale_offset(max_val, min_val, self.act_symmetric, self.act_type)


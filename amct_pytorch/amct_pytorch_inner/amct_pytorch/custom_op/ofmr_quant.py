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

from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.data_utils import convert_precision

EC_CAND = tuple(range(-5, 6))


class OFMRQuant(nn.Module):
    """
    Function: Customized torch.nn.Module of the OFMR quantization class.
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 record_module,
                 layer_name,
                 quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Quantized_type.
        record_module: customed record module. To read and write.
        layer_name: ori_module's name.
        quant_config: calibration algorithm parameters.
        """
        super().__init__()
        self.ori_module = ori_module
        self.module_type = type(ori_module).__name__
        self.post_quant_record_module = record_module
        self.layer_name = layer_name
        self.batch_num = quant_config.get('batch_num')
        self.act_type = quant_config.get('act_type')
        self.wts_type = quant_config.get('wts_type')
        self.round_mode = quant_config.get('round_mode')
        self.weight_compress_only = quant_config.get('weight_compress_only')
        self.device = ori_module.weight.device
        self.act_quant_loss = torch.zeros(len(EC_CAND), dtype=torch.float32, device=self.device)
        self.cout = 1
        if quant_config.get('weight_granularity') == 'PER_CHANNEL':
            self.cout = self.ori_module.weight.shape[0]
        self.wts_quant_loss = torch.zeros((len(EC_CAND), self.cout), dtype=torch.float32, device=self.device)
        self.cur_batch = 0

    @staticmethod
    def compute_mse(a, b, reduce_axis=None):
        """
        Function: compute mse loss.
        Args:
        a: torch.tensor
        b: torch.tensor
        reduce_axis: eg: a.shape=(7,8,9,10), reduce_axis=(0,2,3), sum.shape=(8,)
        """
        if reduce_axis:
            loss = torch.sum((a - b).to(torch.float64).pow(2), dim=reduce_axis)
        else:
            loss = torch.sum((a - b).to(torch.float64).pow(2))
        return (loss / torch.numel(a)).to(torch.float32)

    def forward(self, inputs):
        """
        Function: OFMRQuant foward funtion.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        with torch.no_grad():
            if isinstance(self.ori_module, nn.Linear) and \
                (len(inputs.shape) > 6 or len(inputs.shape) < 2):
                raise RuntimeError("Linear quant only support dim from 2 to 6")
            inputs = inputs.to(self.device)
            fp_out = self.ori_module(inputs)

            self.cur_batch += 1
            if self.cur_batch > self.batch_num:
                return fp_out

            self._calc_weight_quant_loss(inputs, fp_out)
            if not self.weight_compress_only: 
                self._calc_act_quant_loss(inputs, fp_out)

            if self.cur_batch == self.batch_num:
                self._record_quant_factor()

            return fp_out

    def _record_quant_factor(self):
        """
        Function: Write the quantization factor to the record file.
        """
        scale_w = self._calc_scale_w()
        if not self.weight_compress_only:
            scale_d = self._calc_scale_d()
            quant_params = {
                self.layer_name[0]: {
                    'quant_factors': {
                        'scale_w': scale_w,
                        'scale_d': scale_d
                    }
                }
            }
            quant_info = {'act_type': self.act_type,
                            'wts_type': self.wts_type,
                            'round_mode': self.round_mode}
        else:
            quant_params = {
                self.layer_name[0]: {
                    'quant_factors': {
                        'scale_w': scale_w
                    }
                }
            }
            quant_info = {'wts_type': self.wts_type,
                            'round_mode': self.round_mode,
                            'weight_compress_only': True}
        self.post_quant_record_module(self.layer_name, quant_info, quant_params)

    def _calc_scale_d(self):
        """
        Function: Calculate the scale of the activation value by act_quant_loss
        Return: scale_d list, activation scale
        """
        self.act_quant_loss = self.act_quant_loss.masked_fill(torch.isnan(self.act_quant_loss), float('inf'))
        min_value, min_index = torch.min(self.act_quant_loss, dim=0)
        if not torch.isfinite(min_value):
            raise RuntimeError(
                "{}'s activation quant loss has invalid value, inf or nan. "
                "Please check activation value.".format(self.layer_name))
        act_ec = EC_CAND[min_index.tolist()]
        scale_d = torch.Tensor([2 ** act_ec])
        return scale_d

    def _calc_scale_w(self):
        """
        Function: Calculate the scale of the weight value by wts_quant_loss
        Return: scale_w, list, weight scale
        """
        self.wts_quant_loss = self.wts_quant_loss.masked_fill(torch.isnan(self.wts_quant_loss), float('inf'))
        min_values, min_indices = torch.min(self.wts_quant_loss, dim=0)
        if not all(torch.isfinite(min_values)):
            raise RuntimeError(
                "{}'s weight quant loss has invalid value, inf or nan. "
                "Please check weight value.".format(self.layer_name))
        wts_ecs = [EC_CAND[min_index.tolist()] for min_index in min_indices]
        scale_w = torch.Tensor([2 ** wts_ec for wts_ec in wts_ecs])
        if type(self.ori_module).__name__ == 'Linear' and len(scale_w) > 1:
            scale_w = scale_w.reshape(-1, 1)
        return scale_w

    def _calc_weight_quant_loss(self, inputs, fp_out):
        """
        Function: calculate loss in weight quant for candidates in EC_CAND
        Args:
        inputs: torch.tensor quant op's input
        fp_out: torch.tensor quant op's original output
        """
        ori_dtype = self.ori_module.weight.dtype
        for ec in EC_CAND:
            scaled_weights = self.ori_module.weight / (2 ** ec)
            scaled_weights = torch.clamp(scaled_weights, torch.finfo(ori_dtype).min, torch.finfo(ori_dtype).max)
            quant_weights = convert_precision(scaled_weights, self.wts_type, self.round_mode)
            quant_weights = quant_weights * (2 ** ec)
            if self.module_type == 'Linear':
                quant_out = F.linear(inputs, quant_weights, self.ori_module.bias)
                cout_axis = -1 # NC/...C
            else:
                quant_out = F.conv2d(inputs, quant_weights, self.ori_module.bias, self.ori_module.stride,
                    self.ori_module.padding, self.ori_module.dilation, self.ori_module.groups)
                cout_axis = -3 # NCHW/CHW
            axis = None
            if self.cout > 1:
                axis_list = list(range(0, len(fp_out.shape))) # get all axis, e.g conv2d: [0,1,2,3]/[0,1,2]
                axis_list.pop(cout_axis)
                axis = tuple(axis_list)
            self.wts_quant_loss[ec - EC_CAND[0]] += self.compute_mse(fp_out, quant_out, axis)

    def _calc_act_quant_loss(self, inputs, fp_out):
        """
        Function: calculate loss in activation quant for candidates in EC_CAND
        Args:
        inputs: torch.tensor quant op's input
        fp_out: torch.tensor quant op's original output
        """
        ori_dtype = inputs.dtype
        for ec in EC_CAND:
            scaled_inputs = inputs / (2 ** ec)
            scaled_inputs = torch.clamp(scaled_inputs, torch.finfo(ori_dtype).min, torch.finfo(ori_dtype).max)
            quant_inputs = convert_precision(scaled_inputs, self.act_type, self.round_mode)
            quant_inputs = quant_inputs * (2 ** ec)
            if self.module_type == 'Linear':
                quant_out = F.linear(quant_inputs, self.ori_module.weight, self.ori_module.bias)
            else:
                quant_out = F.conv2d(quant_inputs, self.ori_module.weight, self.ori_module.bias,
                    self.ori_module.stride, self.ori_module.padding, self.ori_module.dilation,
                    self.ori_module.groups)
            self.act_quant_loss[ec - EC_CAND[0]] += self.compute_mse(fp_out, quant_out)
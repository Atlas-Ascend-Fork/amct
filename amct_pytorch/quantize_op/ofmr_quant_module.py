# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details at
# http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F

from amct_pytorch.utils.quant_util import quant_dequant_tensor
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule

EC_CAND = tuple(range(-5, 6))


class OfmrQuant(BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the OFMR quantization class.
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
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
        super().__init__(ori_module, layer_name, quant_config)
        self.ori_module = ori_module
        self.ori_module_type = type(ori_module).__name__
        self.module_type = type(ori_module).__name__
        self.layer_name = layer_name

        self.weight = ori_module.weight
        self.bias = ori_module.bias

        self.batch_num = quant_config.get('batch_num')
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        self.act_type = quant_config.get('inputs_cfg').get('quant_type')

        self.weight_compress_only = True if quant_config.get('inputs_cfg').get('enable_quant') == False else False
        self.device = ori_module.weight.device
        self.act_quant_loss = torch.zeros(len(EC_CAND), dtype=torch.float32, device=self.device)
        self.cout = 1
        if quant_config.get('weights_cfg').get('strategy') == 'channel':
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
                self.scale_w = self._calc_scale_w()
                if not self.weight_compress_only:
                    self.scale_d = self._calc_scale_d()

            return fp_out

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
            scale_weight = torch.Tensor([2 ** ec])
            quant_weights = quant_dequant_tensor(self.ori_module.weight, self.wts_type, scale=scale_weight)
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
            scale_inputes = torch.Tensor([2 ** ec]).to(torch.float32)
            quant_inputs = quant_dequant_tensor(inputs, self.act_type, scale=scale_inputes)
            if self.module_type == 'Linear':
                quant_out = F.linear(quant_inputs, self.ori_module.weight, self.ori_module.bias)
            else:
                quant_out = F.conv2d(quant_inputs, self.ori_module.weight, self.ori_module.bias,
                    self.ori_module.stride, self.ori_module.padding, self.ori_module.dilation,
                    self.ori_module.groups)
            self.act_quant_loss[ec - EC_CAND[0]] += self.compute_mse(fp_out, quant_out)

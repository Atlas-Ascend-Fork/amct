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
SCALE_W_DIVISOR = 16


class OfmrQuant(BaseQuantizeModule):
    """
    OFMR Quantization Module.
    
    Weight scale: scale_w = weight_abs_max / SCALE_W_DIVISOR
    Activation scale: searched through EC_CAND using quantized weight
    """

    def __init__(self, ori_module, layer_name, quant_config):
        """Initialize OFMR quantization module with config parameters."""
        super().__init__(ori_module, layer_name, quant_config)
        self.ori_module = ori_module
        self.module_type = type(ori_module).__name__
        self.layer_name = layer_name

        self.batch_num = quant_config.get('batch_num')
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        self.act_type = quant_config.get('inputs_cfg').get('quant_type')
        self.weight_strategy = quant_config.get('weights_cfg').get('strategy')

        self.weight_compress_only = (
            quant_config.get('inputs_cfg').get('enable_quant') == False
        )
        self.device = ori_module.weight.device
        self.cur_batch = 0

        self.scale_w = self._calc_scale_w()
        self.quant_weight = quant_dequant_tensor(
            self.ori_module.weight, self.wts_type, scale=self.scale_w
        )

        if not self.weight_compress_only:
            self.act_quant_loss = torch.zeros(
                len(EC_CAND), dtype=torch.float32, device=self.device
            )

    @staticmethod
    def compute_mse(a, b, reduce_axis=None):
        """Compute MSE loss between two tensors."""
        if reduce_axis:
            loss = torch.sum((a - b).to(torch.float64).pow(2), dim=reduce_axis)
        else:
            loss = torch.sum((a - b).to(torch.float64).pow(2))
        return (loss / torch.numel(a)).to(torch.float32)

    def forward(self, inputs):
        """Forward pass with calibration for scale calculation."""
        with torch.no_grad():
            if isinstance(self.ori_module, nn.Linear) and \
                (len(inputs.shape) > 6 or len(inputs.shape) < 2):
                raise RuntimeError("Linear quant only support dim from 2 to 6")

            inputs = inputs.to(self.device)
            fp_out = self.ori_module(inputs)

            self.cur_batch += 1
            if self.cur_batch > self.batch_num:
                return fp_out

            if not self.weight_compress_only:
                self._calc_act_quant_loss(inputs, fp_out)

            if self.cur_batch == self.batch_num and not self.weight_compress_only:
                self.scale_d = self._calc_scale_d()

            return fp_out

    def _calc_scale_w(self):
        """Calculate weight scale: scale_w = weight_abs_max / SCALE_W_DIVISOR."""
        weight_abs = self.ori_module.weight.abs()

        if self.weight_strategy == 'tensor':
            scale_w = torch.tensor(
                weight_abs.max().item() / SCALE_W_DIVISOR,
                dtype=torch.float32,
                device=self.device
            )
        else:
            if self.module_type == 'Linear':
                max_per_channel = weight_abs.max(dim=1).values
            else:
                max_per_channel = weight_abs.amax(dim=(1, 2, 3))

            scale_w = max_per_channel / SCALE_W_DIVISOR

            if self.module_type == 'Linear' and scale_w.numel() > 1:
                scale_w = scale_w.reshape(-1, 1)

        return scale_w.detach()

    def _calc_scale_d(self):
        """Calculate activation scale by searching EC_CAND for optimal MSE."""
        self.act_quant_loss = self.act_quant_loss.masked_fill(
            torch.isnan(self.act_quant_loss), float('inf')
        )
        min_value, min_index = torch.min(self.act_quant_loss, dim=0)

        if not torch.isfinite(min_value):
            raise RuntimeError(
                "{}'s activation quant loss has invalid value, inf or nan. "
                "Please check activation value.".format(self.layer_name)
            )

        act_ec = EC_CAND[int(min_index.item())]
        scale_d = torch.tensor([2 ** act_ec], dtype=torch.float32)
        return scale_d

    def _calc_act_quant_loss(self, inputs, fp_out):
        """Calculate activation quantization loss using quantized weight."""
        for ec in EC_CAND:
            scale_inputs = torch.tensor([2 ** ec], dtype=torch.float32, device=self.device)
            quant_inputs = quant_dequant_tensor(inputs, self.act_type, scale=scale_inputs)

            if self.module_type == 'Linear':
                quant_out = F.linear(quant_inputs, self.quant_weight, self.ori_module.bias)
            else:
                quant_out = F.conv2d(
                    quant_inputs, self.quant_weight, self.ori_module.bias,
                    self.ori_module.stride, self.ori_module.padding,
                    self.ori_module.dilation, self.ori_module.groups
                )

            self.act_quant_loss[ec - EC_CAND[0]] += self.compute_mse(fp_out, quant_out)

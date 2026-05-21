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
import torch.nn as nn


class BaseQuantizeModule(nn.Module):
    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__()
        self.ori_module_type = None
        self.act_type = None
        self.wts_type = None
        self.scale_w = None
        self.scale_d = None
        self.offset_w = None
        self.offset_d = None
        self.group_size = None

        self.scale = None
        self.clip_max = None

    def _quantizable_weight(self):
        if self.scale is not None:
            return self.weight.data * self.scale.to(device=self.weight.device,
                                                    dtype=self.weight.dtype)
        return self.weight.data

    def _quantizable_input(self, inputs):
        if self.scale is not None:
            return inputs / self.scale.to(device=inputs.device, dtype=inputs.dtype)
        return inputs

    @torch.no_grad()
    def _fake_quant_forward(self, inputs):
        import torch.nn.functional as F
        from amct_pytorch.utils.quant_util import quant_dequant_tensor

        if self.scale_w is None and getattr(self, 'scale_w1', None) is not None \
                and getattr(self, 'scale_w2', None) is not None:
            from amct_pytorch.quantize_op.utils import apply_progressive_quant_dequant
            dequant_w = apply_progressive_quant_dequant(
                self._quantizable_weight(), self.scale_w1, self.scale_w2)
        else:
            dequant_w = quant_dequant_tensor(
                self._quantizable_weight(),
                self.wts_type,
                self.scale_w,
                self.offset_w,
                self.group_size)

        if self.scale_d is not None:
            dequant_x = quant_dequant_tensor(
                self._quantizable_input(inputs),
                self.act_type,
                self.scale_d,
                self.offset_d)
        else:
            dequant_x = inputs

        if isinstance(getattr(self, 'ori_module', None), nn.Conv2d):
            conv = self.ori_module
            return F.conv2d(dequant_x, dequant_w, self.bias,
                            conv.stride, conv.padding, conv.dilation, conv.groups)
        return F.linear(dequant_x, dequant_w, self.bias)

    def forward(self, inputs):
        pass
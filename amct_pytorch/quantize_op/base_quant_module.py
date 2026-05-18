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

        # smooth_factor (SmoothQuant) or AWQ pre-scale; distinct from scale_w
        self.scale = None
        self.clip_max = None

    def _quantizable_weight(self):
        # scale_w was computed from weight * smooth_factor for SmoothQuant,
        # so the same tensor must be passed to quant_dequant_tensor
        if self.scale is not None:
            return self.weight.data * self.scale.to(device=self.weight.device,
                                                    dtype=self.weight.dtype)
        return self.weight.data

    def _quantizable_input(self, inputs):
        # scale_d was computed from inputs / smooth_factor for SmoothQuant
        if self.scale is not None:
            return inputs / self.scale.to(device=inputs.device, dtype=inputs.dtype)
        return inputs

    @torch.no_grad()
    def _fake_quant_forward(self, inputs):
        import torch.nn.functional as F
        from amct_pytorch.utils.quant_util import quant_dequant_tensor

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

        return F.linear(dequant_x, dequant_w, self.bias)

    def forward(self, inputs):
        pass
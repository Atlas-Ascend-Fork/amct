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
import torch.nn.functional as F

try:
    import hifloat8_cast
except ImportError as e:
    print(f'{e}, please check it')
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule


class Hifloat8FakequantLinear(BaseQuantizeModule):
    '''
    input cast to hifloat8
    weight quantize to hifloat8, per-tensor
    '''
    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__(ori_module, layer_name, quant_config)
        self.device = ori_module.weight.device
        self.layer_name = layer_name
        
        # quantize weight, per-tensor
        hifloat8_quant_range = 16   # hifloat8 high precision range
        scale_w = (ori_module.weight.max() / hifloat8_quant_range)
        hif8_weight = hifloat8_cast.float_to_hifloat8((ori_module.weight / scale_w).cpu())
        self.fakequant_weight = hifloat8_cast.hifloat8_to_float32(hif8_weight).to(ori_module.weight) * scale_w

        # keep bias unchanged
        self.bias = ori_module.bias

    @torch.no_grad()
    def forward(self, x):
        # input cast to hifloat8
        quantized_x = hifloat8_cast.float_to_hifloat8(x.cpu())
        fakequant_x = hifloat8_cast.hifloat8_to_float32(quantized_x).to(x)
        return F.linear(fakequant_x, self.fakequant_weight, self.bias)


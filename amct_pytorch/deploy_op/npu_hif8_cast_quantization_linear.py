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

from amct_pytorch.utils.vars import FLOAT8_E4M3FN, HIFLOAT8
from amct_pytorch.utils.log import LOGGER


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
        self.weight = ori_module.weight
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        self.device = ori_module.weight.device
        self.layer_name = layer_name
        self.ori_module = ori_module
        self.act_quant_type = quant_config.get('inputs_cfg').get('quant_type')
        self.weight_compress_only = True
        if quant_config.get('inputs_cfg').get('enable_quant') is None or \
            quant_config.get('inputs_cfg').get('enable_quant') == True:
            self.weight_compress_only = False
        if self.wts_type == HIFLOAT8:
            import torch_npu
            self.weight = torch_npu.npu_dtype_cast(self.weight, self.wts_type)
        self.register_buffer('quantized_weight', self.weight)

    @torch.no_grad()
    def forward(self, x):
        """
        Function: NpuHIF8CastLinear forward function (core feature 3: activation online cast)
        Args:
            x: torch.tensor. Input data for forward computation.
        Return:
            torch.tensor. Output after activation cast and original module forward.
        """
        if not self.weight_compress_only: 
            # 转换激活到目标数据类型
            import torch_npu
            x = torch_npu.npu_dtype_cast(x, self.act_quant_type)
            LOGGER.logd(f"Cast activation of layer '{self.layer_name}' to {self.act_quant_type} success", 'NpuHIF8CastLinear')

        output = self.ori_module(x)
        return output
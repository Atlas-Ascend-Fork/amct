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
import copy
import torch
import torch.nn.functional as F

from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.common.utils.data_utils import check_linear_input_dim
from amct_pytorch.algorithms.quant.awq import (
    search_scale,
    apply_scale,
    calculate_scale_offset_by_granularity,
)
from amct_pytorch.common.utils.quant_util import quant_dequant_weight
from amct_pytorch.common.utils.vars import INT4, INT8, FLOAT4_E2M1
from amct_pytorch.common.utils.log import LOGGER


class LinearAWQuant(BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the LinearAWQuant class.
    APIs: forward.
    """

    def __init__(self, ori_module, layer_name, quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        layer_name: ori_module's name.
        quant_config: calibration algorithm parameters.
        """
        super().__init__(ori_module, layer_name, quant_config)
        self.ori_module_type = type(ori_module).__name__
        self.weight = copy.deepcopy(ori_module.weight)
        self.bias = ori_module.bias
        self.layer_name = layer_name
        self.quant_config = quant_config
        self.wts_symmetric = quant_config.get('weights_cfg').get('symmetric')
        self.ori_module = ori_module
        self.wts_type = self.quant_config.get('weights_cfg').get('quant_type')
        if self.quant_config.get('weights_cfg').get("group_size") is not None:
            self.group_size = self.quant_config.get('weights_cfg').get("group_size")
        self.calc_done = False
        if (
            quant_config.get("inputs_cfg").get("enable_quant") is None
            or quant_config.get("inputs_cfg").get("enable_quant") == True
        ):
            self.act_granularity = quant_config.get('inputs_cfg').get('strategy')

    @torch.no_grad()
    def forward(self, inputs):
        """
        Function: LinearAWQuant foward funtion.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        input_data = inputs.clone()
        with torch.no_grad():
            check_linear_input_dim(input_data)
            input_data = input_data.to(self.weight.device)
            output = F.linear(input_data, self.weight, self.bias)
        if self.calc_done:
            return self.fake_quant_forward(input_data)

        scale_awq = search_scale(
            input_data, [self.ori_module], self.ori_module, self.quant_config
        )
        apply_scale(scale_awq, self.ori_module, input_data)
        self.scale = 1 / scale_awq.detach()

        if self.quant_config.get("weights_cfg").get("quant_type") in (
            INT4,
            INT8,
            FLOAT4_E2M1,
        ):
            self.scale_w, self.offset_w = calculate_scale_offset_by_granularity(
                self.ori_module.weight.data, self.quant_config
            )
        self.calc_done = True
        LOGGER.logd(
            "Calculate awq quant params of layer '{}' success!".format(self.layer_name),
            "LinearAWQuant",
        )
        return output

    @torch.no_grad()
    def fake_quant_forward(self, inputs):
        if not getattr(self, 'fake_quant_cache_ready', False):
            self.cached_dq_w = quant_dequant_weight(
                self.ori_module.weight.data,
                self.wts_type,
                self.scale_w,
                self.offset_w,
                getattr(self, "group_size", None),
            )
            self.fake_quant_cache_ready = True
        x = inputs * self.scale.to(device=inputs.device, dtype=inputs.dtype)
        return F.linear(x, self.cached_dq_w, self.bias)

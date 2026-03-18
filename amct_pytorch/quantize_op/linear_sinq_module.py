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

from amct_pytorch.algorithm.sinq import generate_sinq_params
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.utils.data_utils import check_linear_input_dim
from amct_pytorch.utils.log import LOGGER


class LinearSINQQuant(BaseQuantizeModule):
    """
    Weight-only SINQ module. Parameters are computed eagerly for plain SINQ.
    If calibration is explicitly enabled before convert(), a prepare forward can
    overwrite them with activation-assisted SINQ parameters.
    """
    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__(ori_module, layer_name, quant_config)
        self.ori_module_type = type(ori_module).__name__
        self.weight = copy.deepcopy(ori_module.weight)
        self.bias = ori_module.bias
        self.layer_name = layer_name
        self.ori_module = ori_module
        self.quant_config = quant_config
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        self.wts_symmetric = quant_config.get('weights_cfg').get('symmetric', False)
        self.group_size = quant_config.get('weights_cfg').get('group_size')
        self.method = quant_config.get('algorithm', {}).get('sinq', {}).get('method', 'sinq')
        self.calc_done = False
        self.calibration_enabled = False
        self.calibration_applied = False
        self.calculate_params()

    def calculate_params(self, activations=None):
        self.scale, self.scale_w, self.offset_w, _ = generate_sinq_params(
            self.weight.data, self.quant_config, activations=activations
        )
        self.scale = self.scale.to(dtype=self.weight.dtype, device=self.weight.device)
        self.calc_done = True
        LOGGER.logd("Calculate sinq quant params of layer '{}' success!".format(
            self.layer_name), 'LinearSINQQuant')

    @torch.no_grad()
    def forward(self, inputs):
        input_data = inputs.to(self.weight.device)
        output = F.linear(input_data, self.weight, self.bias)
        if self.calibration_enabled and not self.calibration_applied:
            check_linear_input_dim(input_data)
            self.calculate_params(activations=input_data.detach())
            self.calibration_applied = True
        return output

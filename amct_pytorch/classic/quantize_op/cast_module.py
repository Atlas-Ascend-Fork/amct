# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import torch
import torch.nn.functional as F

from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.classic.quantize_op.utils import calculate_hifloat8_weight_scale
from amct_pytorch.common.utils.data_utils import check_linear_input_dim
from amct_pytorch.common.utils.quant_util import quant_dequant_weight, hifloat8_fake_quant
from amct_pytorch.common.utils.log import LOGGER


class HIF8CastQuant(BaseQuantizeModule):
    """
    Function: fake-quant operator for the HIF8 cast algorithm (no calibration).

    Mirrors NpuHIF8CastLinear's weight scale (weight_max / 16) but simulates the
    cast in FP so accuracy can be evaluated off-NPU. The FP round trip goes through
    hifloat8_fake_quant, which falls back to amct_ops when torch_npu lacks a native
    hifloat8 cast -- that fallback is the reason this stage is split out from the
    native deploy op NpuHIF8CastLinear. The deploy op is rebuilt from this module at
    convert time and keeps its own native compute path unchanged.
    """
    def __init__(self, ori_module, layer_name, quant_config):
        """
        Function: init objective.
        Args:
            ori_module: torch module. Quantized module instance.
            layer_name: str. Original module's name.
            quant_config: dict. Per-layer quantization config (kept for the deploy op
                          to reconstruct itself at convert time).
        """
        super().__init__(ori_module, layer_name, quant_config)
        self.weight = ori_module.weight
        self.bias = ori_module.bias
        self.layer_name = layer_name
        # Retained so the deploy op can read inputs_cfg.quant_type when rebuilding from this module.
        self.quant_config = quant_config
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')

        self.weight_compress_only = True
        if quant_config.get('inputs_cfg').get('enable_quant') is None or \
                quant_config.get('inputs_cfg').get('enable_quant') is True:
            self.weight_compress_only = False

        # Channel/tensor weight scale (1-D), mirroring NpuHIF8CastLinear.
        strategy = quant_config.get('weights_cfg').get('strategy')
        self.scale_w = calculate_hifloat8_weight_scale(self.weight, strategy)
        self.cached_dq_w = quant_dequant_weight(self.weight.data, self.wts_type, self.scale_w)
        LOGGER.logd("Calculate cast quant params of layer '{}' success!".format(self.layer_name), 'HIF8CastQuant')

    @torch.no_grad()
    def forward(self, inputs):
        """Fake-quant forward: dequantized weight matmul, optional activation cast round trip."""
        check_linear_input_dim(inputs)
        inputs = inputs.to(self.weight.device)
        if self.weight_compress_only:
            dq_x = inputs
        else:
            # Native path casts the activation scaleless (convert_dtype + pertoken_scale=None);
            # mirror it with a scaleless hif8 round trip so the amct_ops fallback applies here too.
            dq_x = hifloat8_fake_quant(inputs).to(inputs.dtype)
        return F.linear(dq_x, self.cached_dq_w, self.bias)

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
from amct_pytorch.classic.quantize_op.quant_result_mixin import QuantResultSupportMixin
from amct_pytorch.common.utils.data_utils import check_linear_input_dim
from amct_pytorch.algorithms.quant.awq import (
    search_scale,
    apply_scale,
    calculate_scale_offset_by_granularity,
)
from amct_pytorch.common.utils.quant_util import quant_dequant_weight
from amct_pytorch.common.utils.vars import INT4, INT8, FLOAT4_E2M1
from amct_pytorch.common.utils.log import LOGGER


class LinearAWQuant(QuantResultSupportMixin, BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the LinearAWQuant class.
    APIs: forward.
    """

    # ``weight`` is deliberately absent: AWQ never modifies its own weight, it
    # scales the original module's weight by 1/scale. Both calibration fake-quant
    # and deploy recompute that transform from the original weight and ``scale``,
    # so persisting it would only duplicate the source model.
    _quant_param_keys = ('scale_w', 'offset_w', 'scale')
    supports_quant_result = True

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
        # True only on the reload path, where ori_module.weight was never scaled
        # in place and the transform has to be reapplied from weight and scale.
        self._scale_reapplied = False
        enable_quant = quant_config.get("inputs_cfg").get("enable_quant")
        if enable_quant is None or enable_quant:
            self.act_granularity = quant_config.get('inputs_cfg').get('strategy')

    @torch.no_grad()
    def forward(self, inputs):
        """
        Function: LinearAWQuant forward function.
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
        self.maybe_dump_quant_params()
        return output

    @torch.no_grad()
    def scaled_weight(self):
        """The search-scaled weight AWQ fake-quant works on.

        During calibration ``apply_scale`` multiplies ``ori_module.weight`` in
        place, so that tensor is authoritative and is used as-is. After a reload
        there is no in-place result to read: the source model is untouched, so the
        transform is reapplied from ``self.weight`` (the original weight) and
        ``self.scale``.

        The two are close but not bit-identical. ``self.scale`` is ``1/scale_awq``
        rounded to the weight precision, and dividing by it does not generally
        recover ``weight * scale_awq`` -- on bf16 a few hundred of 2048 elements
        differ by one ulp. Calibration therefore never takes the derived path, so
        ``quant_result_dir=None`` behaves exactly as before this was added.
        """
        original = getattr(self, 'ori_module', None)
        if original is not None and not self._scale_reapplied:
            return original.weight.data
        scale = self.scale.to(device=self.weight.device)
        return (self.weight.data / scale).to(self.weight.dtype)

    @torch.no_grad()
    def fake_quant_forward(self, inputs):
        if not getattr(self, 'fake_quant_cache_ready', False):
            self.cached_dq_w = quant_dequant_weight(
                self.scaled_weight(),
                self.wts_type,
                self.scale_w,
                self.offset_w,
                getattr(self, "group_size", None),
            )
            self.fake_quant_cache_ready = True
        x = inputs * self.scale.to(device=inputs.device, dtype=inputs.dtype)
        return F.linear(x, self.cached_dq_w, self.bias)

    def _on_quant_params_loaded(self):
        """Reloaded params are final: never search a scale again.

        ``scale`` is what makes the module usable, so a unit without it is an
        incomplete artifact rather than something to re-calibrate around. The
        fake-quant cache is dropped so it is rebuilt from the loaded params.
        """
        self.require_loaded_quant_params('scale')
        # Same per-dtype rule as GPTQ: forward() only computes scale_w/offset_w for
        # these types, so only they may be demanded back. Failing here beats
        # entering calc_done with scale_w=None and breaking at fake-quant time.
        if self.wts_type in (INT4, INT8, FLOAT4_E2M1):
            self.require_loaded_quant_params('scale_w')
            if self.wts_symmetric is False:
                self.require_loaded_quant_params('offset_w')
        self.calc_done = True
        # ori_module here is an untouched float layer, so the in-place scaled
        # weight does not exist and must be derived instead.
        self._scale_reapplied = True
        self.fake_quant_cache_ready = False
        self.cached_dq_w = None

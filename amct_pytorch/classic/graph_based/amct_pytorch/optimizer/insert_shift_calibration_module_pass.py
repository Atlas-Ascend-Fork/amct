#!/usr/bin/env python3
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
"""Insert shift calibration module pass.

Wrap the FP16-path target modules with ``ShiftCalibrationModule`` so that
the user calibration forward passes calibrate the per-layer
``fixed_shift_value`` and write it into the record file.

Match rules (``enable_shift_value`` comes from the quant config):
- 'off' (default): no module is wrapped;
- 'all': every FIXED_SHIFT_VALUE_TYPES module is wrapped on the fp16_fp16
  regime. The target layers have already been dropped from the quant layers
  in the config parsing stage, so they run the FP16 non-quantized path;
- 'auto': modules explicitly skipped by the user (skip_layers /
  quant_enable=false, i.e. no longer present in the quant config) are
  wrapped on the fp16_fp16 regime; weight-only modules (whose
  calibration_config carries a wts algo but no act algo: the
  activation is carried unquantized + 8-bit weight, the same
  semantics as the LLM config system's unconfigured inputs) are
  wrapped on the fp16_int8 regime. Modules whose activation is
  quantized (INT16 / INT8 fake quant) never join the fp16_int8
  search.
"""

from ...amct_pytorch.configuration.configuration import Configuration
from ...amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from ...amct_pytorch.nn.module.quantization.shift_calibration import (
    ShiftCalibrationModule,
)
from ...amct_pytorch.nn.module.quantization.shift_calibration import FP16_FP16
from ...amct_pytorch.nn.module.quantization.shift_calibration import FP16_INT8
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.vars import FIXED_SHIFT_VALUE_TYPES
from ...amct_pytorch.utils.vars import ENABLE_SHIFT_VALUE
from ...amct_pytorch.utils.vars import ACTIVATION_QUANT_PARAMS
from ...amct_pytorch.utils.vars import NUM_BITS

WEIGHT_QUANT_PARAMS = 'weight_quant_params'
WEIGHT_INT8_BITS = 8


class InsertShiftCalibrationModulePass(BaseModuleFusionPass):
    """Wrap FP16-path target modules with ShiftCalibrationModule."""

    def __init__(self, record_file):
        """
        Parameter:
            record_file: a string, the record file to write fixed_shift_value.
        Return: None
        """
        BaseModuleFusionPass.__init__(self)
        self.conf = Configuration()
        self.record_file = record_file

    def get_quant_regime(self, name):
        """Decide the shift regime of the module, None means not matched."""
        enable_shift_value = self.conf.get_global_config(ENABLE_SHIFT_VALUE)
        if enable_shift_value is None or enable_shift_value == 'off':
            return None
        if enable_shift_value == 'all':
            return FP16_FP16
        # 'auto': respect the user's explicit layer configs
        layer_config = self.conf.get_layer_config(name)
        if layer_config is None:
            # not in quant config: skipped by user, FP16 non-quantized path
            return FP16_FP16
        if not layer_config.get('quant_enable'):
            return None
        # fp16_int8 entry: the layer's calibration_config carries a wts
        # algo but NO act algo -- the activation is carried UNQUANTIZED
        # (the same semantics as the LLM config system, where an
        # unconfigured inputs means enable_quant=false); it requires an
        # 8-bit weight. The activation-quantized paths (INT16 / INT8
        # fake quant) never join the fp16_int8 search.
        act_config = layer_config.get(ACTIVATION_QUANT_PARAMS) or {}
        if act_config.get('act_algo') is None and (
            (layer_config.get(WEIGHT_QUANT_PARAMS) or {}).get(NUM_BITS)
            == WEIGHT_INT8_BITS
        ):
            return FP16_INT8
        return None

    def match_pattern(self, module, name, graph=None):
        """
        Match the module to be wrapped for shift value calibration.
        """
        if type(module).__name__ not in FIXED_SHIFT_VALUE_TYPES:
            return False
        return self.get_quant_regime(name) is not None

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Replace the matched module with a ShiftCalibrationModule wrapper.
        """
        quant_regime = self.get_quant_regime(object_name)
        batch_num = None
        layer_config = self.conf.get_layer_config(object_name)
        if layer_config is not None:
            act_config = layer_config.get('activation_quant_params') or {}
            batch_num = act_config.get('batch_num')
        if batch_num is None:
            # skipped layers have no layer config entry any more, fall back
            # to the global batch_num
            batch_num = self.conf.get_global_config('batch_num')
        if batch_num is None:
            batch_num = 1

        shift_module = ShiftCalibrationModule(
            object_module=object_module,
            layer_name=object_name,
            record_file=self.record_file,
            batch_num=batch_num,
            quant_regime=quant_regime,
        )

        model_helper = ModuleHelper(model)
        parent_module = model_helper.get_parent_module(object_name)
        setattr(parent_module, object_name.split('.')[-1], shift_module)
        LOGGER.logd(
            "Insert ShiftCalibrationModule of {} to '{}' success!".format(
                quant_regime, object_name
            ),
            'InsertShiftCalibrationModulePass',
        )

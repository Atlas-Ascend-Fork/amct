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
"""Write fixed_shift_value attribute pass.

Write the calibrated ``fixed_shift_value`` as an ONNX INT attribute onto the
deploy model's Conv / Gemm / MatMul / ConvTranspose nodes whose record entry
carries the field. Only run in the deploy branch of ``_generate_model``; the
fakequant model never gets this attribute.
"""

from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.vars import FIXED_SHIFT_VALUE_ONNX_TYPES
from ...amct_pytorch.utils.vars import FIXED_SHIFT_VALUE


class WriteShiftValueAttrPass(BaseFusionPass):
    """Write fixed_shift_value INT attribute onto deploy target nodes."""

    def __init__(self, records):
        """
        Parameter:
            records: dict, parsed record file, keyed by layer name.
        Return: None
        """
        BaseFusionPass.__init__(self)
        self.records = {} if records is None else records

    def match_pattern(self, node):
        """
        Match the deploy node whose record carries fixed_shift_value.
        """
        if node.type not in FIXED_SHIFT_VALUE_ONNX_TYPES:
            return False
        layer_record = self.records.get(node.name)
        if not layer_record or layer_record.get(FIXED_SHIFT_VALUE) is None:
            return False
        return True

    def do_pass(self, graph, object_node):
        """
        Write the fixed_shift_value INT attribute onto the node.
        """
        value = int(self.records.get(object_node.name).get(FIXED_SHIFT_VALUE))
        attr_helper = AttributeProtoHelper(object_node.proto)
        attr_helper.set_attr_value(FIXED_SHIFT_VALUE, 'INT', value)
        LOGGER.logd(
            "Write fixed_shift_value {} to node '{}' success!".format(
                value, object_node.name
            ),
            'WriteShiftValueAttrPass',
        )

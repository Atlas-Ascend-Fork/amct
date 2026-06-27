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
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.optimizer.insert_weight_quant_pass import (
    finalize_int4_weight_for_deploy,
)
from ...amct_pytorch.optimizer.insert_weight_quant_pass import (
    finalize_int4_weight_for_fakequant,
)
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.log import LOGGER

DEPLOY_MODE = "deploy"
FAKEQUANT_MODE = "fakequant"


class FinalizeInt4WeightPass(BaseFusionPass):
    """
    Function: Finalize INT4 weight storage for INT4 layers.
        - mode="deploy": pack two INT4 nibbles into one INT8 byte.
        - mode="fakequant": store as native INT4 (ONNX nibble-packed).
    Covers both the main weight and the RNN recurrence_weight.
    APIs: match_pattern, do_pass

    This pass must run while the original LSTM/GRU nodes are still in the graph
    (i.e. BEFORE ReplaceRNNPass on the deploy path), so that QuantOpInfo can
    locate the weight tensors via the original node types.
    """

    def __init__(self, records, mode):
        BaseFusionPass.__init__(self)
        if mode not in (DEPLOY_MODE, FAKEQUANT_MODE):
            raise ValueError(
                "FinalizeInt4WeightPass mode must be '{}' or '{}', got '{}'".format(
                    DEPLOY_MODE, FAKEQUANT_MODE, mode
                )
            )
        self.records = records
        self.mode = mode
        self._finalize = (
            finalize_int4_weight_for_deploy
            if mode == DEPLOY_MODE
            else finalize_int4_weight_for_fakequant
        )

    def match_pattern(self, node):
        """Match layers configured with INT4 weight."""
        if node.name not in self.records:
            return False
        return self.records[node.name].get("wts_type") == "INT4"

    def do_pass(self, graph, object_node, model=None):
        """Finalize the INT4 weight (and recurrence_weight for RNN) of object_node."""
        weight_node = QuantOpInfo.get_weight_node(object_node)
        self._finalize_node(weight_node)
        recurrence_weight_node = QuantOpInfo.get_recurrence_weight_node(object_node)
        self._finalize_node(recurrence_weight_node)
        LOGGER.logd(
            "Finalize INT4 weight ({} mode) for layer '{}'".format(
                self.mode, object_node.name
            ),
            "FinalizeInt4WeightPass",
        )

    def _finalize_node(self, weight_node):
        """Read INT4 values from a weight tensor node and write them back in the
        target storage form. No-op when weight_node is None (e.g. non-RNN ops have
        no recurrence_weight)."""
        if weight_node is None:
            return
        helper = TensorProtoHelper(weight_node.proto, weight_node.model_path)
        int4_vals = helper.get_data()
        data, dtype_str = self._finalize(int4_vals)
        helper.clear_data()
        # dims note (deploy): set_data switches data_type to INT8 but intentionally
        # preserves the original element-count dims. The deploy model is an
        # AMCT-specific format; the Ascend runtime identifies packed INT4-in-INT8
        # weights via the op_data_type attribute, not dims x sizeof. Do NOT update
        # dims here without auditing the runtime consumer.
        helper.set_data(data, dtype_str)

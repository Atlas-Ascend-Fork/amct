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
import numpy as np

from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.utils.onnx_initializer_util import (
    TensorProtoHelper,
    pack_int4_to_int8,
)
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.log import LOGGER


class PackInt4WeightPass(BaseFusionPass):
    """
    Function: Pack INT4 weight storage for the deploy model: two INT4 nibbles
        into one INT8 byte. Applies to every layer whose weight is configured
        as INT4 (Conv/ConvTranspose/Linear/LSTM/GRU). The main weight is packed
        for all such layers; recurrence_weight is packed additionally for RNN
        ops only (non-RNN ops have no recurrence_weight and are skipped).
    APIs: match_pattern, do_pass

    For RNN layers this pass must run while the original LSTM/GRU nodes are
    still in the graph (i.e. BEFORE ReplaceRNNPass on the deploy path), so that
    QuantOpInfo can locate the recurrence_weight tensor via the original node
    type.
    """

    def __init__(self, records):
        BaseFusionPass.__init__(self)
        self.records = records

    def match_pattern(self, node):
        """Match layers configured with INT4 weight."""
        if node.name not in self.records:
            return False
        return self.records[node.name].get("wts_type") == "INT4"

    def do_pass(self, graph, object_node, model=None):
        """Pack the INT4 weight (and recurrence_weight for RNN) of object_node."""
        weight_node = QuantOpInfo.get_weight_node(object_node)
        self._pack_int4_weight(weight_node)
        recurrence_weight_node = QuantOpInfo.get_recurrence_weight_node(object_node)
        self._pack_int4_weight(recurrence_weight_node)
        LOGGER.logd(
            "Pack INT4 weight (deploy) for layer '{}'".format(object_node.name),
            "PackInt4WeightPass",
        )

    def _pack_int4_weight(self, weight_node):
        """Read INT4 values from a weight tensor node and write them back packed
        two-per-byte. No-op when weight_node is None (e.g. non-RNN ops have no
        recurrence_weight)."""
        if weight_node is None:
            return
        helper = TensorProtoHelper(weight_node.proto, weight_node.model_path)
        int4_vals = helper.get_data()
        # 两个 INT4 pack 成一个 INT8，raw_data 字节数减半，但 dims 保留原始多维
        # 形状（供部署侧按原始 shape 重建权重）。deploy 为 AMCT 专有格式，实际
        # 字节数由 op_data_type 标记为 INT4-packed，不满足标准 ONNX 的 dims*sizeof。
        packed = pack_int4_to_int8(np.asarray(int4_vals).reshape(-1))
        helper.clear_data()
        helper.set_data(packed, 'INT8')


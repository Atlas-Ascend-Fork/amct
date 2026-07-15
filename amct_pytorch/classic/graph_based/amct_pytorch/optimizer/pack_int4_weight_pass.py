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
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
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
        int4_vals = np.asarray(helper.get_data())
        orig_dims = list(helper.tensor.dims)
        # 沿最后一轴 nibble-pack：末轴每两个 INT4 合成一个 INT8，其余轴不变。
        # 末轴为奇数时逐行末尾补 0（末轴 -> ceil(last/2)）。deploy 为 AMCT 专有
        # 格式，实际字节由 op_data_type 标记为 INT4-packed。
        packed, packed_last = self._pack_last_axis(int4_vals, orig_dims)
        helper.clear_data()
        new_dims = orig_dims[:-1] + [packed_last] if len(orig_dims) > 1 else [packed_last]
        helper.set_data(packed, 'INT8', dims=new_dims)

    @staticmethod
    def _pack_last_axis(int4_vals, dims):
        """Pack every two INT4 along the last axis into one INT8, leaving other
        axes unchanged. Odd last-axis rows are zero-padded (last -> ceil(last/2)).
        Returns (flat_packed_int8, packed_last_size)."""
        last = dims[-1] if dims else int(np.prod(int4_vals.shape))
        rows = int4_vals.reshape(-1, last).astype(np.int8)
        if last % 2 == 1:
            LOGGER.logw(
                'INT4 weight last-axis length {} is odd, zero-padding each row '
                'for packing'.format(last),
                'PackInt4WeightPass',
            )
            pad = np.zeros((rows.shape[0], 1), dtype=np.int8)
            rows = np.concatenate([rows, pad], axis=1)
        low = rows[:, 0::2] & 0x0F
        high = rows[:, 1::2] & 0x0F
        packed = (low | (high << 4)).astype(np.uint8).astype(np.int8)
        return packed.reshape(-1), packed.shape[1]


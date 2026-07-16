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

    # 量化轴（per-channel scale 所在的 Cout 维）在 onnx 权重张量里的位置：
    # Conv 为轴 0；ConvTranspose/MatMul/Gemm(transB=0)/LSTM/GRU 为轴 1。
    _QUANT_AXIS = {
        'Conv': 0,
        'ConvTranspose': 1,
        'MatMul': 1,
        'Gemm': 1,
        'LSTM': 1,
        'GRU': 1,
    }

    def do_pass(self, graph, object_node, model=None):
        """Pack the INT4 weight (and recurrence_weight for RNN) of object_node."""
        axis = self._QUANT_AXIS.get(object_node.type, 0)
        weight_node = QuantOpInfo.get_weight_node(object_node)
        self._pack_int4_weight(weight_node, axis)
        # recurrence_weight 仅 RNN 有，其量化轴（4H）同样在 onnx 张量的轴 1
        recurrence_weight_node = QuantOpInfo.get_recurrence_weight_node(object_node)
        self._pack_int4_weight(recurrence_weight_node, axis)
        LOGGER.logd(
            "Pack INT4 weight (deploy) for layer '{}'".format(object_node.name),
            "PackInt4WeightPass",
        )

    def _pack_int4_weight(self, weight_node, quant_axis):
        """Read INT4 values from a weight tensor node and write them back packed
        two-per-byte along quant_axis. No-op when weight_node is None (e.g. non-RNN
        ops have no recurrence_weight)."""
        if weight_node is None:
            return
        helper = TensorProtoHelper(weight_node.proto, weight_node.model_path)
        int4_vals = np.asarray(helper.get_data())
        orig_dims = list(helper.tensor.dims)
        # 沿量化轴（per-channel scale 所在的 Cout 维）nibble-pack：该轴每两个 INT4
        # 合成一个 INT8，其余轴不变（该轴 -> ceil(axis/2)）。deploy 为 AMCT 专有
        # 格式，实际字节由 op_data_type 标记为 INT4-packed。
        packed, new_dims = self._pack_along_axis(int4_vals, orig_dims, quant_axis)
        helper.clear_data()
        helper.set_data(packed, 'INT8', dims=new_dims)

    @staticmethod
    def _pack_along_axis(int4_vals, dims, axis):
        """Pack every two INT4 into one INT8 along `axis`, leaving other axes
        unchanged. Odd axis length is zero-padded (axis -> ceil(len/2)).
        Returns (flat_packed_int8, new_dims)."""
        arr = int4_vals.reshape(dims).astype(np.int8)
        # 把量化轴移到最后，沿它两两配对，再移回原位
        moved = np.moveaxis(arr, axis, -1)
        last = moved.shape[-1]
        if last % 2 == 1:
            LOGGER.logw(
                'INT4 weight quant-axis length {} is odd, zero-padding for '
                'packing'.format(last),
                'PackInt4WeightPass',
            )
            pad = np.zeros(moved.shape[:-1] + (1,), dtype=np.int8)
            moved = np.concatenate([moved, pad], axis=-1)
        low = moved[..., 0::2] & 0x0F
        high = moved[..., 1::2] & 0x0F
        packed_moved = (low | (high << 4)).astype(np.uint8).astype(np.int8)
        packed = np.moveaxis(packed_moved, -1, axis)
        return packed.reshape(-1), list(packed.shape)


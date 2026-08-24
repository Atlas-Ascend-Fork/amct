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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import unittest
from unittest import mock

import numpy as np
from onnx import TensorProto

from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.pack_int4_weight_pass import (
    PackInt4WeightPass,
)


class TestPackInt4Weight(unittest.TestCase):
    def setUp(self):
        self.passer = PackInt4WeightPass({})

    def test_match_pattern(self):
        passer = PackInt4WeightPass(
            {'conv': {'wts_type': 'INT4'}, 'fc': {'wts_type': 'INT8'}}
        )
        node_int4 = mock.MagicMock()
        node_int4.name = 'conv'
        node_int8 = mock.MagicMock()
        node_int8.name = 'fc'
        node_absent = mock.MagicMock()
        node_absent.name = 'other'
        self.assertTrue(passer.match_pattern(node_int4))  # INT4 -> match
        self.assertFalse(passer.match_pattern(node_int8))  # INT8 -> no
        self.assertFalse(passer.match_pattern(node_absent))  # not in records -> no

    _QOI = (
        'amct_pytorch.classic.graph_based.amct_pytorch.optimizer.'
        'pack_int4_weight_pass.QuantOpInfo'
    )

    def test_do_pass_writes_native_int4_with_original_shape(self):
        node = mock.MagicMock()
        node.type = 'Conv'
        node.name = 'conv'
        weight_node = mock.MagicMock()
        weight_node.model_path = ''
        weight_node.proto = TensorProto()
        weight_node.proto.data_type = TensorProto.INT8
        weight_node.proto.dims.extend([2, 4])
        values = np.array([[-3, 5, -8, -1], [1, 2, 3, 4]], dtype=np.int8)
        weight_node.proto.raw_data = values.tobytes()

        with mock.patch(self._QOI) as m_qoi:
            m_qoi.get_weight_node.return_value = weight_node
            m_qoi.get_recurrence_weight_node.return_value = None
            self.passer.do_pass(None, node)

        self.assertEqual(weight_node.proto.data_type, TensorProto.INT4)
        self.assertEqual(list(weight_node.proto.dims), [2, 4])
        self.assertEqual(weight_node.proto.raw_data, bytes([0x5D, 0xF8, 0x21, 0x43]))

    def test_do_pass_rejects_odd_last_axis_even_when_total_size_is_even(self):
        node = mock.MagicMock()
        node.name = 'conv'
        weight_node = mock.MagicMock()
        weight_node.model_path = ''
        weight_node.proto = TensorProto()
        weight_node.proto.data_type = TensorProto.INT8
        weight_node.proto.dims.extend([2, 3])
        weight_node.proto.raw_data = np.arange(6, dtype=np.int8).tobytes()

        with mock.patch(self._QOI) as m_qoi:
            m_qoi.get_weight_node.return_value = weight_node
            m_qoi.get_recurrence_weight_node.return_value = None
            with self.assertRaisesRegex(RuntimeError, r'last axis.*odd'):
                self.passer.do_pass(None, node)

    def test_do_pass_rnn_packs_weight_and_recurrence(self):
        # RNN：主权重 + recurrence_weight 都被打包(两次 set_data)
        node = mock.MagicMock()
        node.type = 'LSTM'
        node.name = 'lstm'
        wnode = mock.MagicMock()
        wnode.model_path = ''
        helper = mock.MagicMock()
        helper.get_data.return_value = np.arange(8, dtype=np.int8).reshape(1, 4, 2)
        helper.tensor.dims = [1, 4, 2]
        with (
            mock.patch(self._QOI) as m_qoi,
            mock.patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.optimizer.'
                'pack_int4_weight_pass.TensorProtoHelper',
                return_value=helper,
            ),
        ):
            m_qoi.get_weight_node.return_value = wnode
            m_qoi.get_recurrence_weight_node.return_value = wnode
            self.passer.do_pass(None, node)
        # 主 + recurrence 各一次 set_data
        self.assertEqual(helper.set_data.call_count, 2)


if __name__ == '__main__':
    unittest.main()

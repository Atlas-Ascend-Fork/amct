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
"""Unit tests for PackInt4WeightPass that do NOT rely on onnx native INT4
support. The pack logic is pure numpy / dict manipulation, so these run on any
onnx version and keep diff coverage up even when end-to-end INT4 tests skip on
old onnx."""
import unittest
from unittest import mock

import numpy as np

from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.pack_int4_weight_pass import (
    PackInt4WeightPass,
)


class TestPackAlongAxis(unittest.TestCase):
    def setUp(self):
        self.passer = PackInt4WeightPass({})

    def test_pack_along_last_axis_even(self):
        # [2, 4] pack along axis 1 -> [2, 2]
        vals = np.arange(8, dtype=np.int8).reshape(2, 4)
        packed, new_dims = self.passer._pack_along_axis(vals, [2, 4], 1)
        self.assertEqual(new_dims, [2, 2])
        self.assertEqual(packed.size, 4)

    def test_pack_along_axis0(self):
        # Conv-like: [16, 4, 3, 3] pack along Cin axis 1 -> [16, 2, 3, 3]
        vals = (np.arange(16 * 4 * 3 * 3, dtype=np.int8) % 15 - 7)
        packed, new_dims = self.passer._pack_along_axis(vals, [16, 4, 3, 3], 1)
        self.assertEqual(new_dims, [16, 2, 3, 3])
        self.assertEqual(packed.size, 16 * 2 * 3 * 3)

    def test_pack_nibble_layout(self):
        # low nibble = first value, high nibble = second value
        vals = np.array([1, 2], dtype=np.int8)
        packed, _ = self.passer._pack_along_axis(vals, [2], 0)
        self.assertEqual(int(packed[0]) & 0x0F, 1)
        self.assertEqual((int(packed[0]) >> 4) & 0x0F, 2)

    def test_pack_odd_axis_raises(self):
        # odd pack axis should be rejected at config stage; defensive raise here
        vals = np.arange(3, dtype=np.int8)
        with self.assertRaises(RuntimeError):
            self.passer._pack_along_axis(vals, [3], 0)


class TestPackInt4Weight(unittest.TestCase):
    def setUp(self):
        self.passer = PackInt4WeightPass({})

    def test_match_pattern(self):
        passer = PackInt4WeightPass({'conv': {'wts_type': 'INT4'},
                                     'fc': {'wts_type': 'INT8'}})
        node_int4 = mock.MagicMock()
        node_int4.name = 'conv'
        node_int8 = mock.MagicMock()
        node_int8.name = 'fc'
        node_absent = mock.MagicMock()
        node_absent.name = 'other'
        self.assertTrue(passer.match_pattern(node_int4))       # INT4 -> match
        self.assertFalse(passer.match_pattern(node_int8))      # INT8 -> no
        self.assertFalse(passer.match_pattern(node_absent))    # not in records -> no

    def test_none_weight_node_is_noop(self):
        # non-RNN ops have no recurrence_weight -> weight_node is None -> no-op
        self.assertIsNone(self.passer._pack_int4_weight(None, 1))

    def test_pack_writes_int8_with_new_dims(self):
        # mock TensorProtoHelper to isolate the pack logic from onnx
        weight_node = mock.MagicMock()
        weight_node.model_path = ''
        int4_vals = np.arange(8, dtype=np.int8).reshape(2, 4)
        helper = mock.MagicMock()
        helper.get_data.return_value = int4_vals
        helper.tensor.dims = [2, 4]
        module = (
            'amct_pytorch.classic.graph_based.amct_pytorch.optimizer.'
            'pack_int4_weight_pass.TensorProtoHelper'
        )
        with mock.patch(module, return_value=helper):
            self.passer._pack_int4_weight(weight_node, 1)
        helper.clear_data.assert_called_once()
        # set_data called with dtype INT8 and Cin-halved dims
        args, kwargs = helper.set_data.call_args
        self.assertEqual(args[1], 'INT8')
        self.assertEqual(kwargs.get('dims'), [2, 2])

    def test_do_pass_packs_weight_and_recurrence(self):
        node = mock.MagicMock()
        node.type = 'LSTM'
        node.name = 'lstm'
        qoi = (
            'amct_pytorch.classic.graph_based.amct_pytorch.optimizer.'
            'pack_int4_weight_pass.QuantOpInfo'
        )
        with mock.patch(qoi) as m_qoi, \
                mock.patch.object(self.passer, '_pack_int4_weight') as m_pack:
            m_qoi.get_cin_axis.return_value = 2
            m_qoi.get_weight_node.return_value = 'w'
            m_qoi.get_recurrence_weight_node.return_value = 'r'
            self.passer.do_pass(None, node)
        # both main weight and recurrence_weight packed along the same Cin axis
        self.assertEqual(m_pack.call_count, 2)
        m_pack.assert_any_call('w', 2)
        m_pack.assert_any_call('r', 2)


if __name__ == '__main__':
    unittest.main()

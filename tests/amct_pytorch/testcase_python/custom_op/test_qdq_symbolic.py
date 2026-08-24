#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
"""Unit tests for the shared ONNX Q/DQ symbolic builder."""

import unittest
from unittest import mock

from onnx import TensorProto

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.qdq_symbolic import (
    add_qdq,
)


class TestQdqSymbolic(unittest.TestCase):
    def test_add_int4_qdq_accepts_explicit_num_bits(self):
        graph = mock.MagicMock()
        graph.op.side_effect = lambda *args, **kwargs: mock.MagicMock(name='node')

        add_qdq(
            graph,
            mock.sentinel.tensor,
            mock.sentinel.scale,
            mock.sentinel.zero_point,
            num_bits=4,
            axis=1,
        )

        quant_call, dequant_call = graph.op.call_args_list
        self.assertEqual(quant_call.args[0], 'QuantizeLinear')
        self.assertEqual(dequant_call.args[0], 'DequantizeLinear')
        self.assertEqual(quant_call.kwargs['output_dtype_i'], TensorProto.INT4)
        self.assertEqual(quant_call.kwargs['axis_i'], 1)
        self.assertEqual(dequant_call.kwargs['axis_i'], 1)

    def test_add_int8_qdq_keeps_zero_point_inputs(self):
        graph = mock.MagicMock()
        graph.op.side_effect = [mock.sentinel.quant_node, mock.sentinel.dequant_node]

        add_qdq(
            graph,
            mock.sentinel.tensor,
            mock.sentinel.scale,
            mock.sentinel.zero_point,
            num_bits=8,
        )

        quant_call, dequant_call = graph.op.call_args_list
        self.assertEqual(
            quant_call.args,
            (
                'QuantizeLinear',
                mock.sentinel.tensor,
                mock.sentinel.scale,
                mock.sentinel.zero_point,
            ),
        )
        self.assertEqual(
            dequant_call.args,
            (
                'DequantizeLinear',
                mock.sentinel.quant_node,
                mock.sentinel.scale,
                mock.sentinel.zero_point,
            ),
        )


if __name__ == '__main__':
    unittest.main()

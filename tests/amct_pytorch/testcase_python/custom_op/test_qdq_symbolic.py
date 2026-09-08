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

import torch
from onnx import TensorProto

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.qdq_symbolic import (
    add_qdq,
)
from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.qdq_symbolic import (
    add_qdq_dynamo,
    add_weight_qdq_dynamo,
    check_int4_dynamo_export,
    is_dynamo_export,
)


class TestQdqSymbolic(unittest.TestCase):
    def _patch_dynamo_ops(self):
        ops = mock.MagicMock()
        ops.symbolic.side_effect = lambda name, inputs, **kwargs: mock.Mock(
            name=name, dtype=kwargs.get('dtype'), shape=kwargs.get('shape')
        )
        return mock.patch('torch.onnx.ops', ops, create=True), ops

    def test_add_weight_qdq_dynamo_conv_transpose(self):
        patcher, ops = self._patch_dynamo_ops()
        with patcher:
            add_weight_qdq_dynamo(
                torch.randn(2, 4, 3),
                torch.ones(4),
                torch.zeros(1),
                4,
                'ConvTranspose1d',
                True,
                mock.Mock(),
            )
        self.assertEqual(ops.symbolic.call_count, 2)

    def test_add_weight_qdq_dynamo_conv_transposes_layout(self):
        patcher, ops = self._patch_dynamo_ops()
        with patcher:
            add_weight_qdq_dynamo(
                torch.randn(4, 2, 3, 3),
                torch.ones(4),
                torch.zeros(1),
                4,
                'Conv2d',
                True,
                mock.Mock(),
            )
        quant_inputs = ops.symbolic.call_args_list[0].args[1]
        self.assertEqual(tuple(quant_inputs[0].shape), (2, 4, 3, 3))

    def test_add_weight_qdq_dynamo_linear_per_channel_transposes_layout(self):
        patcher, ops = self._patch_dynamo_ops()
        module = mock.Mock(weight=torch.randn(4, 2, 3))
        with patcher:
            add_weight_qdq_dynamo(
                module.weight, torch.ones(4), torch.zeros(1), 4, 'Linear', True, module
            )
        quant_inputs = ops.symbolic.call_args_list[0].args[1]
        self.assertEqual(tuple(quant_inputs[0].shape), (2, 4, 3))

    def test_add_weight_qdq_dynamo_linear_per_tensor(self):
        patcher, ops = self._patch_dynamo_ops()
        with patcher:
            add_weight_qdq_dynamo(
                torch.randn(4, 2),
                torch.ones(1),
                torch.zeros(1),
                8,
                'Linear',
                False,
                mock.Mock(),
            )
        self.assertEqual(ops.symbolic.call_count, 2)

    def test_add_weight_qdq_dynamo_rejects_unknown_module(self):
        with self.assertRaisesRegex(RuntimeError, 'Unsupported QAT module'):
            add_weight_qdq_dynamo(
                torch.randn(2, 2),
                torch.ones(1),
                torch.zeros(1),
                8,
                'Unknown',
                False,
                mock.Mock(),
            )

    def test_is_dynamo_export_in_onnx_compilation_context(self):
        with (
            mock.patch('torch.compiler.is_compiling', return_value=True),
            mock.patch('torch.onnx.is_in_onnx_export', return_value=True),
            mock.patch('torch.onnx.ops', mock.Mock(symbolic=mock.Mock()), create=True),
        ):
            self.assertTrue(is_dynamo_export())

    def test_is_dynamo_export_requires_onnx_export_context(self):
        with (
            mock.patch('torch.compiler.is_compiling', return_value=True),
            mock.patch('torch.onnx.is_in_onnx_export', return_value=False),
            mock.patch('torch.onnx.ops', mock.Mock(symbolic=mock.Mock()), create=True),
        ):
            self.assertFalse(is_dynamo_export())

    def test_add_qdq_dynamo_uses_native_int4_dtype_without_zero_point(self):
        ops = mock.MagicMock()
        ops.symbolic.side_effect = [mock.sentinel.quant, mock.sentinel.dequant]
        tensor = mock.Mock(dtype='float32')
        with mock.patch('torch.onnx.ops', ops, create=True):
            output = add_qdq_dynamo(
                tensor,
                mock.sentinel.scale,
                mock.sentinel.zero_point,
                num_bits=4,
                axis=1,
                shape=(2, 4),
            )

        self.assertIs(output, mock.sentinel.dequant)
        quant_call, dequant_call = ops.symbolic.call_args_list
        self.assertEqual(
            quant_call.args[:2], ('QuantizeLinear', (tensor, mock.sentinel.scale))
        )
        self.assertEqual(
            quant_call.kwargs['attrs'], {'output_dtype': TensorProto.INT4, 'axis': 1}
        )
        self.assertEqual(quant_call.kwargs['dtype'], TensorProto.INT4)
        self.assertEqual(quant_call.kwargs['version'], 21)
        self.assertEqual(
            dequant_call.args[:2],
            ('DequantizeLinear', (mock.sentinel.quant, mock.sentinel.scale)),
        )
        self.assertEqual(dequant_call.kwargs['attrs'], {'axis': 1})

    def test_add_qdq_dynamo_keeps_zero_point_for_int8(self):
        ops = mock.MagicMock()
        ops.symbolic.side_effect = [mock.sentinel.quant, mock.sentinel.dequant]
        tensor = mock.Mock(dtype='float32')
        with mock.patch('torch.onnx.ops', ops, create=True):
            add_qdq_dynamo(
                tensor,
                mock.sentinel.scale,
                mock.sentinel.zero_point,
                num_bits=8,
                shape=(2, 4),
            )

        quant_call, dequant_call = ops.symbolic.call_args_list
        self.assertEqual(
            quant_call.args[1],
            (tensor, mock.sentinel.scale, mock.sentinel.zero_point),
        )
        self.assertEqual(
            dequant_call.args[1],
            (mock.sentinel.quant, mock.sentinel.scale, mock.sentinel.zero_point),
        )

    def test_add_qdq_dynamo_casts_float_zero_point_to_integer(self):
        ops = mock.MagicMock()
        ops.symbolic.side_effect = [mock.sentinel.quant, mock.sentinel.dequant]
        tensor = mock.Mock(dtype='float32')
        float_zero_point = torch.tensor([-3.0, 0.0, 7.0])
        with mock.patch('torch.onnx.ops', ops, create=True):
            add_qdq_dynamo(tensor, mock.sentinel.scale, float_zero_point, 8)

        quant_inputs = ops.symbolic.call_args_list[0].args[1]
        self.assertEqual(quant_inputs[2].dtype, torch.int8)
        self.assertTrue(
            torch.equal(quant_inputs[2], torch.tensor([-3, 0, 7], dtype=torch.int8))
        )

    def test_add_qdq_dynamo_uses_int16_for_int16_quantization(self):
        ops = mock.MagicMock()
        ops.symbolic.side_effect = [mock.sentinel.quant, mock.sentinel.dequant]
        tensor = mock.Mock(dtype=torch.float32)
        with mock.patch('torch.onnx.ops', ops, create=True):
            add_qdq_dynamo(tensor, mock.sentinel.scale, mock.sentinel.zero_point, 16)
        self.assertEqual(ops.symbolic.call_args_list[0].kwargs['dtype'], torch.int16)

    def test_add_qdq_dynamo_restores_input_device(self):
        ops = mock.MagicMock()
        ops.symbolic.side_effect = lambda *args, **kwargs: torch.zeros(
            kwargs['shape'], dtype=torch.float32
        )
        tensor = torch.empty((2, 4), device='meta')
        scale = torch.empty(1, device='meta')
        with mock.patch('torch.onnx.ops', ops, create=True):
            output = add_qdq_dynamo(tensor, scale, None, 8)
        self.assertEqual(output.device, tensor.device)

    def test_add_qdq_dynamo_requires_native_export_api(self):
        with mock.patch('torch.onnx.ops', None, create=True):
            with self.assertRaisesRegex(RuntimeError, 'PyTorch 2.10'):
                add_qdq_dynamo(mock.sentinel.tensor, mock.sentinel.scale, None, 4)

    def test_check_int4_dynamo_export_validates_per_channel_scales(self):
        module = mock.Mock(
            out_channels=2, wts_scales=mock.Mock(numel=mock.Mock(return_value=1))
        )
        with self.assertRaisesRegex(ValueError, 'scale count.*out_channels'):
            check_int4_dynamo_export({'channel_wise': True, 'module': module})

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

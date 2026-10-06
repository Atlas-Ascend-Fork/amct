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
"""Unit tests for UlqScaleRetrainFuncQAT.symbolic ONNX-lowering branches.

symbolic() only writes ONNX ops through the graph builder `g`, so it is
exercised directly with a mocked graph (a full ONNX export cannot run in the
CPU CI environment due to opset incompatibility).
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from onnx import TensorProto

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.ulq_scale_retrain.ulq_scale_retrain import (
    UlqScaleRetrainFunction,
    UlqScaleRetrainFuncQAT,
)


def _make_graph():
    g = mock.MagicMock()
    g.op.side_effect = lambda *a, **k: mock.MagicMock(name="node")
    return g


def _export_opset(version):
    return mock.patch(
        'torch.onnx._globals.GLOBALS',
        SimpleNamespace(export_onnx_opset_version=version),
    )


def _make_inputs(
    module_type,
    hidden_size=8,
    num_bits=8,
    channel_wise=False,
    out_channels=4,
    scale_count=None,
):
    module = mock.MagicMock()
    module.hidden_size = hidden_size
    module.out_channels = out_channels
    module.wts_scales = torch.ones(out_channels if scale_count is None else scale_count)
    wts_param = {
        "module_type": module_type,
        "module": module,
        "num_bits": num_bits,
        "channel_wise": channel_wise,
    }
    tensor = torch.randn(4, 4)
    scale = module.wts_scales
    zero = torch.zeros(1)
    # symbolic reads positional args at index 0, 1, 3 and 5
    return [tensor, scale, zero, wts_param, zero, zero]


class TestUlqScaleRetrainSymbolic(unittest.TestCase):
    def test_forward_dynamo_bypasses_eager_quantization(self):
        wts_param = _make_inputs('Linear', num_bits=4)[3]
        offset_deploy = torch.tensor([0], dtype=torch.int8)
        with (
            mock.patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.custom_op.ulq_scale_retrain.ulq_scale_retrain.is_dynamo_export',
                return_value=True,
            ),
            mock.patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.custom_op.ulq_scale_retrain.ulq_scale_retrain.check_int4_dynamo_export'
            ),
            mock.patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.custom_op.ulq_scale_retrain.ulq_scale_retrain.add_weight_qdq_dynamo',
                return_value=torch.randn(4, 4),
            ) as add_qdq,
        ):
            result = UlqScaleRetrainFunction.forward(
                None,
                torch.randn(4, 4),
                torch.ones(4),
                torch.zeros(4),
                wts_param,
                0,
                offset_deploy,
            )
        self.assertEqual(len(result), 3)
        add_qdq.assert_called_once()
        self.assertIs(add_qdq.call_args.args[2], offset_deploy)

    def test_symbolic_conv_transpose(self):
        self._run("ConvTranspose1d")

    def test_symbolic_conv1d(self):
        self._run("Conv1d")

    def test_symbolic_conv2d(self):
        self._run("Conv2d")

    def test_symbolic_conv3d(self):
        self._run("Conv3d")

    def test_symbolic_linear(self):
        self._run("Linear")

    def test_symbolic_lstm(self):
        self._run("LSTM")

    def test_symbolic_gru(self):
        self._run("GRU")

    def test_int4_conv2d_per_channel_qdq_contract(self):
        g = _make_graph()
        inputs = _make_inputs('Conv2d', num_bits=4, channel_wise=True)
        with _export_opset(21):
            UlqScaleRetrainFuncQAT.symbolic(g, *inputs)

        calls = g.op.call_args_list
        self.assertEqual(
            [call.args[0] for call in calls],
            ['Transpose', 'QuantizeLinear', 'DequantizeLinear', 'Transpose'],
        )
        self.assertEqual(len(calls[1].args), 3)
        self.assertEqual(len(calls[2].args), 3)
        self.assertEqual(calls[1].kwargs['output_dtype_i'], TensorProto.INT4)
        self.assertEqual(calls[1].kwargs['axis_i'], 1)
        self.assertEqual(calls[2].kwargs['axis_i'], 1)

    def test_int4_linear_per_channel_uses_inverse_transposes(self):
        g = _make_graph()
        inputs = _make_inputs('Linear', num_bits=4, channel_wise=True)
        with _export_opset(21):
            UlqScaleRetrainFuncQAT.symbolic(g, *inputs)

        calls = g.op.call_args_list
        self.assertEqual(
            [call.args[0] for call in calls],
            ['Transpose', 'QuantizeLinear', 'DequantizeLinear', 'Transpose'],
        )
        self.assertEqual(calls[0].kwargs['perm_i'], [1, 0])
        self.assertEqual(calls[3].kwargs['perm_i'], [1, 0])
        self.assertEqual(calls[1].kwargs['axis_i'], 1)
        self.assertEqual(calls[1].kwargs['output_dtype_i'], TensorProto.INT4)
        self.assertEqual(len(calls[1].args), 3)

    def test_int4_linear_per_tensor_has_no_transpose_or_zero_point(self):
        g = _make_graph()
        inputs = _make_inputs('Linear', num_bits=4, channel_wise=False, scale_count=1)
        with _export_opset(21):
            UlqScaleRetrainFuncQAT.symbolic(g, *inputs)

        calls = g.op.call_args_list
        self.assertEqual(
            [call.args[0] for call in calls],
            ['QuantizeLinear', 'DequantizeLinear'],
        )
        self.assertNotIn('axis_i', calls[0].kwargs)
        self.assertEqual(len(calls[0].args), 3)

    def test_int4_per_channel_scale_count_matches_output_channels(self):
        with _export_opset(21):
            with self.assertRaisesRegex(ValueError, r'scale count.*out_channels'):
                UlqScaleRetrainFuncQAT.symbolic(
                    _make_graph(),
                    *_make_inputs(
                        'Linear', num_bits=4, channel_wise=True, scale_count=3
                    ),
                )

    def _run(self, module_type):
        g = _make_graph()
        inputs = _make_inputs(module_type)
        out = UlqScaleRetrainFuncQAT.symbolic(g, *inputs)
        self.assertEqual(len(out), 3)
        self.assertTrue(g.op.called)


if __name__ == "__main__":
    unittest.main()

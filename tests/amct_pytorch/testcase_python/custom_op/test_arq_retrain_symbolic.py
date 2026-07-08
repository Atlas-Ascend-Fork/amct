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
"""Unit tests for ArqRetrainFuncQAT.symbolic ONNX-lowering branches.

The symbolic() method only builds ONNX ops through the graph builder `g`, so it
can be exercised directly with a mocked graph instead of a full ONNX export
(which the CPU CI environment cannot run due to opset incompatibility).
"""
import unittest
from unittest.mock import MagicMock

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.arq_retrain.arq_retrain import (
    ArqRetrainFuncQAT,
)


def _make_graph():
    """A fake ONNX graph whose op() returns a fresh sentinel node each call."""
    g = MagicMock()
    g.op.side_effect = lambda *a, **k: MagicMock(name="node")
    return g


def _make_inputs(module_type, hidden_size=8):
    module = MagicMock()
    module.hidden_size = hidden_size
    # symbolic reads positional args tensor, scale, offset, wts_param, zero_point
    wts_param = {"module_type": module_type, "module": module}
    tensor = torch.randn(4, 4)
    scale = torch.ones(1)
    offset = torch.zeros(1)
    zero_point = torch.zeros(1)
    return (tensor, scale, offset, wts_param, zero_point)


class TestArqRetrainSymbolic(unittest.TestCase):
    def test_symbolic_conv_transpose(self):
        self._run("ConvTranspose2d")

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

    def _run(self, module_type):
        g = _make_graph()
        inputs = _make_inputs(module_type)
        out = ArqRetrainFuncQAT.symbolic(g, *inputs)
        self.assertEqual(len(out), 3)
        self.assertTrue(g.op.called)
        return out


if __name__ == "__main__":
    unittest.main()

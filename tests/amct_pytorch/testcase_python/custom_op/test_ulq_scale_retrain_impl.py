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
"""Unit tests for ulq_scale_retrain_impl backward math (pure torch, no NPU/ONNX)."""
import unittest

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.ulq_scale_retrain.ulq_scale_retrain_impl import (
    calc_scale_gradient,
    ulq_scale_retrain_backward_pytorch,
)


class TestUlqScaleRetrainBackward(unittest.TestCase):
    def test_per_tensor_backward_shapes(self):
        inp = torch.randn(4, 8)
        grad_out = torch.randn(4, 8)
        scale = torch.tensor([0.1])
        grad_in, grad_scale = ulq_scale_retrain_backward_pytorch(
            inp, grad_out, scale, 8, False, 1, 0
        )
        self.assertEqual(grad_in.shape, inp.shape)
        self.assertEqual(grad_scale.shape, (1,))

    def test_per_channel_backward_shapes(self):
        inp = torch.randn(4, 8)
        grad_out = torch.randn(4, 8)
        scale = torch.tensor([0.1, 0.2, 0.1, 0.2])
        grad_in, grad_scale = ulq_scale_retrain_backward_pytorch(
            inp, grad_out, scale, 8, False, 1, 0
        )
        self.assertEqual(grad_scale.shape, (4,))

    def test_srec_flag_reciprocal_path(self):
        """srec_flag hits the reciprocal branches (lines 144/151/189)."""
        inp = torch.randn(4, 8)
        grad_out = torch.randn(4, 8)
        scale = torch.tensor([0.1, 0.2, 0.3, 0.4])
        grad_in, grad_scale = ulq_scale_retrain_backward_pytorch(
            inp, grad_out, scale, 8, True, 1, 0
        )
        self.assertEqual(grad_scale.shape, (4,))
        self.assertTrue(torch.isfinite(grad_scale).all())

    def test_group_wise_backward(self):
        """group>1 hits repeat_interleave + view-mean branch (line 154)."""
        inp = torch.randn(4, 8)
        grad_out = torch.randn(4, 8)
        scale = torch.tensor([0.1, 0.2])  # 2 groups over 4 rows
        grad_in, grad_scale = ulq_scale_retrain_backward_pytorch(
            inp, grad_out, scale, 8, False, 2, 0
        )
        self.assertEqual(grad_scale.shape, (2,))

    def test_4d_input_reshaped(self):
        """4D input is flattened then restored (lines 130-132, 156-157)."""
        inp = torch.randn(2, 3, 4, 4)
        grad_out = torch.randn(2, 3, 4, 4)
        scale = torch.tensor([0.1])
        grad_in, grad_scale = ulq_scale_retrain_backward_pytorch(
            inp, grad_out, scale, 8, False, 1, 0
        )
        self.assertEqual(grad_in.shape, (2, 3, 4, 4))

    def test_group_not_divisible_raises(self):
        inp = torch.randn(3, 8)
        grad_out = torch.randn(3, 8)
        scale = torch.tensor([0.1, 0.2])
        with self.assertRaises(ValueError):
            ulq_scale_retrain_backward_pytorch(inp, grad_out, scale, 8, False, 2, 0)

    def test_calc_scale_gradient_per_channel(self):
        inp = torch.randn(4, 8)
        grad_out = torch.randn(4, 8)
        scale_process = torch.tensor([0.1, 0.2, 0.3, 0.4])
        grad = calc_scale_gradient(inp, scale_process, grad_out, 8, False)
        self.assertEqual(grad.shape, (4,))

    def test_calc_scale_gradient_per_tensor_unsqueezes(self):
        inp = torch.randn(1, 8)
        grad_out = torch.randn(1, 8)
        scale_process = torch.tensor([0.1])
        grad = calc_scale_gradient(inp, scale_process, grad_out, 8, False)
        self.assertEqual(grad.shape, (1,))


if __name__ == "__main__":
    unittest.main()

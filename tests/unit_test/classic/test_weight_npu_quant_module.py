#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
"""Unit tests for NpuWeightQuantizedLinear (torch_npu mocked).

Regression guard for the aclnn EZ1001 bug: npu_weight_quant_batchmatmul
requires the reshaped 2D input to be contiguous. A 3D input sliced along the
last (K) dim stays non-contiguous after reshape(-1, K), so forward() must
call .contiguous() explicitly.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn


def _install_fake_torch_npu():
    """Register a fake torch_npu recording the x handed to the batchmatmul op."""
    mod = types.ModuleType("torch_npu")
    mod.hifloat8 = "hifloat8_enum"
    calls = []

    def fake_batchmatmul(x, weight, scale, **kwargs):
        calls.append(x)
        return torch.zeros(*x.shape[:-1], weight.shape[-1], dtype=x.dtype)

    mod.npu_weight_quant_batchmatmul = fake_batchmatmul
    mod.npu_weight_quant_batchmatmul._schemas = {"": MagicMock()}
    mod.npu_weight_quant_batchmatmul._schemas[""].arguments = [
        MagicMock(name="weight_dtype")
    ]
    sys.modules["torch_npu"] = mod
    return mod, calls


class TestNpuWeightQuantizedLinear(unittest.TestCase):
    """forward() must pass a contiguous 2D tensor to the npu op."""

    def setUp(self):
        self._saved_npu = getattr(torch.Tensor, "npu", None)
        self._saved_torch_npu = sys.modules.get("torch_npu")
        torch.Tensor.npu = lambda self: self
        _, self.calls = _install_fake_torch_npu()

    def tearDown(self):
        if self._saved_npu is not None:
            torch.Tensor.npu = self._saved_npu
        elif hasattr(torch.Tensor, "npu"):
            delattr(torch.Tensor, "npu")
        if self._saved_torch_npu is not None:
            sys.modules["torch_npu"] = self._saved_torch_npu
        else:
            sys.modules.pop("torch_npu", None)

    def _make_module(self, k=256, n=128, group=64):
        qm = MagicMock(spec=nn.Linear)
        linear = nn.Linear(k, n, bias=False, dtype=torch.float32)
        qm.weight = linear.weight
        qm.bias = None
        qm.wts_type = "int8"
        qm.group_size = group
        qm.scale_w = torch.rand(n, k // group, 1) * 1e-3 + 1e-4
        qm.offset_w = None
        qm.scale = None

        from amct_pytorch.classic.deploy_op.weight_npu_quant_module import (
            NpuWeightQuantizedLinear,
        )

        return NpuWeightQuantizedLinear(qm)

    def test_forward_input_to_op_is_contiguous(self):
        """Sliced 3D input stays non-contiguous after reshape; op needs contig."""
        mod = self._make_module()
        # [B, S, 2K] sliced to [..., K]: reshape(-1, K) cannot be a view here
        # and the result of reshape on this layout is still non-contiguous.
        x = torch.randn(2, 16, 512)[:, :, :256]
        self.assertFalse(
            x.reshape(-1, x.shape[-1]).is_contiguous(),
            "precondition: reshape alone must not fix contiguity",
        )

        out = mod(x)
        self.assertEqual(out.shape, (2, 16, 128))
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(
            self.calls[0].is_contiguous(),
            "npu_weight_quant_batchmatmul got a non-contiguous x (aclnn EZ1001)",
        )
        self.assertEqual(self.calls[0].shape, (32, 256))

    def test_forward_output_shape_restored(self):
        """Output is reshaped back to the original leading dims."""
        mod = self._make_module()
        out = mod(torch.randn(3, 5, 256))
        self.assertEqual(out.shape, (3, 5, 128))
        self.assertTrue(self.calls[0].is_contiguous())


if __name__ == "__main__":
    unittest.main()

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
"""UT for the public re-export (forwarding) modules.

These modules expose the documented public import paths and forward to the real
implementations under ``amct_pytorch.classic.graph_based.amct_pytorch``. The test
asserts that each documented path imports and resolves to the exact same object
as its underlying implementation.
"""
import importlib
import unittest


class TestPublicForwarding(unittest.TestCase):
    """Verify documented public import paths forward to the real implementations."""

    def test_auto_calibration_forwarding(self):
        """auto_calibration public path forwards to the real base classes."""
        base = "amct_pytorch.classic.graph_based.amct_pytorch.common.auto_calibration"
        for name in ("AutoCalibrationEvaluatorBase", "AccuracyBasedAutoCalibrationBase",
                     "AutoCalibrationStrategyBase", "SensitivityBase",
                     "BinarySearchStrategy", "CosineSimilaritySensitivity"):
            self._assert_same("amct_pytorch.common.auto_calibration", name, base, name)

    def test_qat_module_forwarding(self):
        """Each QAT submodule public path forwards to the real implementation."""
        base = "amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization"
        cases = [
            ("amct_pytorch.nn.module.quantization.conv2d", "Conv2dQAT"),
            ("amct_pytorch.nn.module.quantization.conv3d", "Conv3dQAT"),
            ("amct_pytorch.nn.module.quantization.conv_transpose_2d", "ConvTranspose2dQAT"),
            ("amct_pytorch.nn.module.quantization.linear", "LinearQAT"),
            ("amct_pytorch.nn.module.quantization.quant_calibration_op", "QuantCalibrationOp"),
        ]
        for public_mod, attr in cases:
            self._assert_same(public_mod, attr, base, attr)

    def test_qat_quantization_package_exports(self):
        """The quantization package re-exports all QAT classes."""
        mod = importlib.import_module("amct_pytorch.nn.module.quantization")
        for attr in ("Conv2dQAT", "Conv3dQAT", "ConvTranspose2dQAT",
                     "LinearQAT", "QuantCalibrationOp"):
            self.assertTrue(hasattr(mod, attr))

    def test_tensor_decompose_forwarding(self):
        """tensor_decompose public path forwards to the real functions."""
        base = "amct_pytorch.classic.graph_based.amct_pytorch.tensor_decompose"
        for name in ("auto_decomposition", "decompose_network"):
            self._assert_same("amct_pytorch.tensor_decompose", name, base, name)

    def test_auto_channel_prune_forwarding(self):
        """auto_channel_prune public submodules forward to the real base classes."""
        base = "amct_pytorch.classic.graph_based.amct_pytorch.common.auto_channel_prune"
        self._assert_same(
            "amct_pytorch.common.auto_channel_prune.sensitivity_base",
            "SensitivityBase", base, "SensitivityBase")
        self._assert_same(
            "amct_pytorch.common.auto_channel_prune.search_channel_base",
            "SearchChannelBase", base, "SearchChannelBase")

    def test_auto_channel_prune_package_exports(self):
        """The auto_channel_prune package re-exports both base classes."""
        mod = importlib.import_module("amct_pytorch.common.auto_channel_prune")
        self.assertTrue(hasattr(mod, "SensitivityBase"))
        self.assertTrue(hasattr(mod, "SearchChannelBase"))

    def _assert_same(self, public_mod, public_attr, impl_mod, impl_attr):
        """Assert the public attribute is the same object as the implementation."""
        pub = getattr(importlib.import_module(public_mod), public_attr)
        impl = getattr(importlib.import_module(impl_mod), impl_attr)
        self.assertIs(pub, impl, f"{public_mod}.{public_attr} should forward to "
                                 f"{impl_mod}.{impl_attr}")


if __name__ == '__main__':
    unittest.main()

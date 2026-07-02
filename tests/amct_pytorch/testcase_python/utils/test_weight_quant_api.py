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
import logging
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from amct_pytorch.classic.graph_based.amct_pytorch.utils.weight_quant_api import (
    adjust_axis_for_group_wise,
    adjust_conv_weight_shape,
    adjust_deconv_weight_shape,
    get_deconv_group,
)

logger = logging.getLogger(__name__)


class TestNetParams(unittest.TestCase):
    """
    UT for ParamsHelperTorch
    """

    @classmethod
    def setUpClass(cls):
        logger.info("Test ParamsHelperTorch start!")

    @classmethod
    def tearDownClass(cls):
        logger.info("Test ParamsHelperTorch end!")

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_adjust_conv_weight_shape(self):
        group = 4
        weight = np.random.randn(4, 3, 16, 16)
        adjusted_weight = adjust_conv_weight_shape(group, weight)
        self.assertEqual(adjusted_weight.shape, (12, 1, 16, 16))

        group = 1
        weight = np.random.randn(4, 3, 16)
        adjusted_weight = adjust_conv_weight_shape(group, weight)
        self.assertEqual(adjusted_weight.shape, (3, 4, 16))

        group = 2
        weight = np.random.randn(4, 3, 16)
        adjusted_weight = adjust_conv_weight_shape(group, weight)
        self.assertEqual(adjusted_weight.shape, (6, 2, 16))

        group = 2
        weight = np.random.randn(4, 3, 16, 16, 16)
        adjusted_weight = adjust_conv_weight_shape(group, weight)
        self.assertEqual(adjusted_weight.shape, (6, 2, 16, 16, 16))

        group = 2
        weight = np.random.randn(4, 3, 16)
        adjusted_weight = adjust_conv_weight_shape(group, weight)
        self.assertEqual(adjusted_weight.shape, (6, 2, 16))

    def test_adjust_axis_for_group_wise(self):
        tensor = torch.randn(1, 2, 3, 4)
        ret = adjust_axis_for_group_wise(axis=2, input_tensor=tensor)
        self.assertEqual(ret.shape, torch.Size((3, 2, 1, 4)))

    def test_adjust_deconv_weight_shape(self):
        weight_tensor = torch.randn(4, 5, 6, 7)
        group = 2
        adjusted_weight = adjust_deconv_weight_shape(group, weight_tensor)
        self.assertEqual(list(adjusted_weight.shape), [10, 2, 6, 7])

        weight_tensor = torch.randn(4, 5, 6, 7)
        group = 1
        adjusted_weight = adjust_deconv_weight_shape(group, weight_tensor)
        self.assertEqual(list(adjusted_weight.shape), [5, 4, 6, 7])

        weight_tensor = torch.randn(4, 5, 6)
        group = 2
        adjusted_weight = adjust_deconv_weight_shape(group, weight_tensor)
        self.assertEqual(list(adjusted_weight.shape), [10, 2, 6])

    def test_get_deconv_group_returns_value(self):
        node = MagicMock()
        with patch(
            "amct_pytorch.classic.graph_based.amct_pytorch.utils.weight_quant_api.AttributeProtoHelper"
        ) as mock_helper:
            inst = mock_helper.return_value
            inst.has_attr.return_value = True
            inst.get_attr_value.return_value = 4
            self.assertEqual(get_deconv_group(node), 4)

    def test_get_deconv_group_raises_when_no_group_attr(self):
        node = MagicMock()
        node.name = "deconv0"
        with patch(
            "amct_pytorch.classic.graph_based.amct_pytorch.utils.weight_quant_api.AttributeProtoHelper"
        ) as mock_helper:
            inst = mock_helper.return_value
            inst.has_attr.return_value = False
            with self.assertRaises(RuntimeError):
                get_deconv_group(node)

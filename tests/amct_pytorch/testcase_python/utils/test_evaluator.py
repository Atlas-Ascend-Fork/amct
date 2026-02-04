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
import os
import unittest
import torch
import torch.nn as nn
from unittest import mock
import numpy as np

from amct_pytorch.graph_based_compression.amct_pytorch.utils.evaluator import ModelEvaluator
import amct_pytorch.graph_based_compression.amct_pytorch.common.cmd_line_utils.data_handler as data_handler


class TestEvaluatorHelper(unittest.TestCase):
    """
    The UT for evaluator helper
    """
    @classmethod
    def setUpClass(cls):
        print("Test Evaluator Helper start!")
        input_shape = "input_name1:1,3,5,5"
        data_dir = "data/input1"
        data_types = "float32"
        cls.evaluator_helper = ModelEvaluator(
            input_shape=input_shape,
            data_dir=data_dir,
            data_types=data_types
        )

    @classmethod
    def tearDownClass(cls):
        print("Test Evaluator Helper end!")
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_calibration(self):
        modified_model = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5),
            nn.BatchNorm2d(num_features=4)
        )
        data_map = [{
            "input1": np.random.rand(1, 3, 10, 10).astype('f'),
            "input2": np.random.rand(1, 3, 20, 20).astype('f')
        }]
        with mock.patch('amct_pytorch.graph_based_compression.amct_pytorch.common.cmd_line_utils.data_handler.load_data',
            return_value=data_map):
            self.assertIsNone(self.evaluator_helper.calibration(modified_model=modified_model, batch_num=1))

    def test_evaluate(self):
        modified_model = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5),
            nn.BatchNorm2d(num_features=4)
        )
        data_map = [{
            "input1": np.random.rand(1, 3, 10, 10).astype('f'),
            "input2": np.random.rand(1, 3, 20, 20).astype('f')
        }]
        with mock.patch('amct_pytorch.graph_based_compression.amct_pytorch.common.cmd_line_utils.data_handler.load_data',
            return_value=data_map):
            self.assertIsNone(self.evaluator_helper.evaluate(modified_model=modified_model, iterations=1))

    def test_preprocess_input_shape_None(self):
        input_shape = None
        self.assertRaises(ValueError, self.evaluator_helper._preprocess_input_shape, input_shape)

    def test_preprocess_input_shape_len(self):
        input_shape = "input_name1:1:"
        self.assertRaises(ValueError, self.evaluator_helper._preprocess_input_shape, input_shape)

    def test_preprocess_input_shape(self):
        input_shape = "input_name1:1,3,5,5;input_name2:1,2,3,3"
        expect_input_dict = {"input_name1":[1,3,5,5], "input_name2": [1,2,3,3]}
        input_dict = self.evaluator_helper._preprocess_input_shape(input_shape)
        self.assertEqual(input_dict, expect_input_dict)

    def test_preprocess_data_dir_None(self):
        data_dir = None
        self.assertRaises(ValueError, self.evaluator_helper._preprocess_data_dir, data_dir)

    def test_preprocess_data_dir(self):
        data_dir = "data/input1;data/input2"
        data_paths = self.evaluator_helper._preprocess_data_dir(data_dir)
        self.assertEqual(len(data_paths), 2)

    def test_preprocess_data_types_None(self):
        data_types = None
        self.assertRaises(ValueError, self.evaluator_helper._preprocess_data_types, data_types)

    def test_preprocess_data_types(self):
        data_types = "float32;float64"
        expect_values = ["float32", "float64"]
        values = self.evaluator_helper._preprocess_data_types(data_types)
        self.assertEqual(values, expect_values)

    def test_preprocess_batch_num(self):
        batch_num = 0
        self.assertRaises(ValueError, self.evaluator_helper._preprocess_batch_num, batch_num)

    def test_preprocess_batch_num_001(self):
        batch_num = self.evaluator_helper._preprocess_batch_num(batch_num=1)
        self.assertEqual(batch_num, 1)

if __name__ == '__main__':
    unittest.main()
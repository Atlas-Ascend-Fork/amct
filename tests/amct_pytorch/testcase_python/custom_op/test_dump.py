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
import sys
import os
import unittest
from unittest import mock
from unittest.mock import patch
import json
import numpy as np
import torch
import math

from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.dump.dump import DUMP
from amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils import struct_helper
from amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils import files as files_util
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.auto_calibration_helper import AutoCalibrationHelper



CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestDumpForward(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_dump_forward')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        dump_config = struct_helper.DumpConfig(
            batch_num=-1,
            dump_dir=cls.temp_folder
        )
        layers_name = ['test_layer']
        cls.dump_module = DUMP(
            layers_name=layers_name,
            dump_config=dump_config
        )

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_001_float32(self):
        data = torch.tensor([1,2,3], dtype=torch.float32)
        ret = self.dump_module(data)
        assert 0 == ((ret != data).sum())

    def test_002_double(self):
        data = torch.tensor([1,2,3], dtype=torch.float64)
        ret = self.dump_module(data)
        assert 0 == ((ret != data).sum())

    def test_003_not_support(self):
        data = torch.tensor([1,2,3], dtype=torch.int8)
        self.assertRaises(RuntimeError, self.dump_module, data)

    def test_004_data_dump_right_nor_not(self):
        data = torch.tensor([1.2, 2.4, 3.7], dtype=torch.float32)
        self.dump_module(data)

        file_path = os.path.join(self.temp_folder, 'test_layer_activation_batch4.bin')
        read_data = files_util.parse_dump_data(file_path, with_type=True)
        print(data)
        print(torch.from_numpy(read_data))
        self.assertEqual((data - torch.from_numpy(read_data)).sum(), 0)

    def test_005_data_dump_right_nor_not(self):
        data = torch.tensor([1.123, 2.456, 3.789], dtype=torch.float64)
        self.dump_module(data)

        file_path = os.path.join(self.temp_folder, 'test_layer_activation_batch5.bin')
        read_data = files_util.parse_dump_data(file_path, with_type=True)
        print(data)
        print(torch.from_numpy(read_data))
        self.assertEqual((data - torch.from_numpy(read_data)).sum(), 0)

    def test_006_int(self):
        data = torch.tensor([1,2,3], dtype=torch.int32)
        ret = self.dump_module(data)
        assert 0 == ((ret != data).sum())

    def test_007_data_dump_right_nor_not(self):
        data = torch.tensor([1, 2, 3], dtype=torch.int32)
        self.dump_module(data)

        file_path = os.path.join(self.temp_folder, 'test_layer_activation_batch7.bin')
        read_data = files_util.parse_dump_data(file_path, with_type=True)
        print(data)
        print(torch.from_numpy(read_data))
        self.assertEqual((data - torch.from_numpy(read_data)).sum(), 0)

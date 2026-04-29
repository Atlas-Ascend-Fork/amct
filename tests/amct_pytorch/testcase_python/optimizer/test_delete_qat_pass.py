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
import json
import numpy as np
import torch
import torch.nn as nn
import copy

from .utils import models
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.model_optimizer import ModelOptimizer
from amct_pytorch.graph_based_compression.amct_pytorch.common.utils.record_file_operator import \
    ScaleOffsetRecordHelper
from amct_pytorch.graph_based_compression.amct_pytorch.proto import scale_offset_record_pb2

from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.delete_qat_pass import DeleteQatPass
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class DistillQATNet(nn.Module):
    def __init__(self):
        super(DistillQATNet, self).__init__()
        self.conv = Conv2dQAT(2, 2, kernel_size=3, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class TestDeleteQatPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_insert_qat_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.qat_model = DistillQATNet()
        cls.inputs = torch.randn((3, 2, 224, 224))
        cls.output = cls.qat_model.forward(cls.inputs)
        cls.record_helper = ScaleOffsetRecordHelper(scale_offset_record_pb2.ScaleOffsetRecord)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_delete_qat_match_pattern_success(self):
        mod = Conv2dQAT(2, 2, kernel_size=3, bias=False)
        self.assertTrue(DeleteQatPass(self.record_helper).match_pattern(mod, 'layer1'))

    def test_delete_qat_match_pattern_not_QAT(self):
        mod = torch.nn.Conv2d(1, 1, 1, padding_mode='zeros')
        self.assertFalse(DeleteQatPass(self.record_helper).match_pattern(mod, 'layer1'))

    def test_insert_qat_do_pass_success(self):
        optimizer = ModelOptimizer()
        optimizer.add_pass(DeleteQatPass(self.record_helper))
        model = copy.deepcopy(self.qat_model)
        optimizer.do_optimizer(model, None)

        self.assertTrue(isinstance(model.conv, nn.Conv2d))


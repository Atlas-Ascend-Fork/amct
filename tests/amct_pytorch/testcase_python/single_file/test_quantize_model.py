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

from .utils import models
from .utils import record_file_utils
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.proto import scale_offset_record_pb2
from google.protobuf import text_format

from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import create_quant_config
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import quantize_model
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import save_model


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestQuantizeModel(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_quantize_model')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_002 = models.Net4dMatmul().to(torch.device("cpu"))
        cls.model_002.eval()
        cls.args_shape = [(1, 4, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_quantize_model(self):
        config_file = os.path.join(CUR_DIR, 'utils/test_quantize_model/model_002.json')
        modfied_onnx_file = os.path.join(self.temp_folder, 'no_exit/model_modified.onnx')
        record_file = os.path.join(self.temp_folder, 'model_002.txt')
        print("=*"*30)
        for name, mod in self.model_002.named_modules():
            print(name, mod)
        new_model = quantize_model(config_file, modfied_onnx_file, record_file,
            self.model_002, self.args, None, None, None)

        data = self.args[0]
        for _ in range(2):
            ans_2 = new_model(data)

        self.assertTrue(os.path.exists(modfied_onnx_file))
        self.assertTrue(os.path.exists(record_file))

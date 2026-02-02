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
import unittest
import json
import os
import numpy as np
import torch

from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.distill_config import get_enable_quant_layers
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.distill_config import get_quant_layer_config

from .utils import models

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestDistillConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.distill_config = {
            'conv1' : {
                'quant_enable' : True,
                'distill_data_config' : {
                    'algo' : 'ulq_quantize',
                    'dst_type' : 'INT8'
                },
                'distill_weight_config' : {
                    'algo' : 'arq_distill',
                    'channel_wise' : True,
                    'dst_type' : 'INT8'
                }
            },
            'conv2' : {
                'quant_enable' : False
            }
        }

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_enable_quant_layers(self):
        layers = get_enable_quant_layers(self.distill_config)
        self.assertEqual(layers, ['conv1'])

    def test_get_quant_layer_config_not_quant(self):
        layer_config = get_quant_layer_config('conv2', self.distill_config)
        self.assertIsNone(layer_config)

    def test_get_quant_layer_config_success(self):
        layer_config = get_quant_layer_config('conv1', self.distill_config)
        self.assertEqual(layer_config, self.distill_config.get('conv1'))
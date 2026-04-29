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
import numpy as np

import amct_pytorch.graph_based_compression.amct_pytorch
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.fake_quant import FakeQuant

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
DEVICE = torch.device('cpu')

np.random.seed(0)


def quant_compute(inputs, scale_d, offset_d, num_bits=8):
    clamp_min = -2**(num_bits - 1)
    clamp_max = 2**(num_bits - 1) - 1
    temp_data = np.round(inputs * scale_d) + offset_d
    clamped_data = np.clip(temp_data, clamp_min, clamp_max)
    quantized_data = clamped_data - offset_d
    return quantized_data

def compare_ndarray(ndarray1, ndarray2, name):
    if not (ndarray1 - ndarray2 < 1e-5).all():
        print(ndarray1)
        print(ndarray2)
        raise ValueError('{} not equal'.format(name))


class TestFakeQuantModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_fake_quant_module')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fake_quant_module(self):
        scale_d = 1
        offset_d = -128
        inputs = torch.tensor(np.random.uniform(0, 1, (1,3,224,224)), dtype=torch.float32)
        fake_quant = FakeQuant(scale_d, offset_d)
        out = fake_quant(inputs)
        np_out = quant_compute(inputs.numpy(), scale_d, offset_d)
        compare_ndarray(np_out, out.detach().numpy(), 'fake_quant')

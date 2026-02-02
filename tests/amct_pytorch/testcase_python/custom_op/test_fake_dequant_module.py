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
import amct_pytorch.amct_pytorch_inner.amct_pytorch

from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.fake_quant import FakeDeQuant

np.random.seed(0)

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
DEVICE = torch.device('cpu')
S16_BASE = 16
S32_BASE = 32

def compare_ndarray(ndarray1, ndarray2, name):
    if not (ndarray1 - ndarray2 < 1e-5).all():
        print(ndarray1)
        print(ndarray2)
        raise ValueError('{} not equal'.format(name))

def dequant_compute(inputs, scale_d, scale_w, deq_shape):
    deq_scale = scale_d * scale_w
    deq_scale = deq_scale.reshape(deq_shape)
    dequantized_data = inputs * deq_scale
    return dequantized_data

class TestFakeDeQuantModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_fake_dequant_module')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fake_dequant_module(self):
        scale_d = 1
        scale_w = np.array([0.5, 0.5, 0.5])
        deq_shape = (-1, 1, 1, 1)
        offset_d = -128
        inputs = torch.tensor(
            np.random.randint(0, 255, (1, 3, 224, 224)), dtype=torch.float32)
        fake_dequant = FakeDeQuant(scale_d, scale_w, deq_shape)
        out = fake_dequant(inputs)
        np_out = dequant_compute(
            inputs.numpy(), scale_d, scale_w, deq_shape)
        compare_ndarray(np_out, out.detach().numpy(), 'fake_dequant')

    def test_fake_dequant_module_per_tensor(self):
        scale_d = 1
        scale_w = np.array([0.5])
        deq_shape = (1, -1)
        offset_d = -128
        inputs = torch.tensor(
            np.random.randint(0, 255, (1, 3, 224, 224)), dtype=torch.float32)
        fake_dequant = FakeDeQuant(scale_d, scale_w, deq_shape)
        out = fake_dequant(inputs)
        np_out = dequant_compute(
            inputs.numpy(), scale_d, scale_w, deq_shape)
        compare_ndarray(np_out, out.detach().numpy(), 'fake_dequant')

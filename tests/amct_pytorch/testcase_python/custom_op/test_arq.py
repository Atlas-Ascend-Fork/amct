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

from amct_pytorch.graph_based_compression.amct_pytorch.custom_op import arq_cali_pytorch
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op import arq_real_pytorch

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda:0')

class TestArqOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_arq')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_arq_cali_pytorch_channelwise_F_withoffset_F(self):
        '''channel_wise: F withoffset:F '''
        data_list = [[-1.0]*12, [0.0]*12, [1.0]*12, [-1.0, 0.0, 1.0]*4]
        input_data = torch.tensor(data_list, device=DEVICE)
        scale, offset, output_data = arq_cali_pytorch(input_data, 8, False, False)

        scale_except = torch.tensor([0.007844], device=DEVICE)
        scale_err = torch.abs(scale_except - scale).to('cpu')
        offset_except = torch.tensor([0], dtype=torch.int8, device=DEVICE)
        err = torch.abs(output_data - input_data).to('cpu')

        self.assertTrue(torch.ge(1e-4*torch.ones([1]), scale_err))
        self.assertEqual(offset_except, offset)
        self.assertTrue(torch.ge(1e-2*torch.ones([4, 12]), err).numpy().all())

    def test_arq_cali_pytorch_channelwise_T_withoffset_F(self):
        '''channel_wise: T withoffset:F '''
        data_list = [[-1.0]*12, [0.0]*12, [1.0]*12, [-1.0, 0.0, 1.0]*4]
        input_data = torch.tensor(data_list, device=DEVICE)
        scale, offset, output_data = arq_cali_pytorch(input_data, 8, True, False)

        scale_except = torch.tensor([0.007844, 1.000000, 0.007844, 0.007844], device=DEVICE)
        scale_err = torch.abs(scale_except - scale).to('cpu')
        offset_except = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=DEVICE)
        err = torch.abs(output_data - input_data).to('cpu')

        self.assertTrue(torch.ge(1e-4*torch.ones([1]), scale_err).numpy().all())
        self.assertTrue(torch.equal(offset_except, offset))
        self.assertTrue(torch.ge(1e-2*torch.ones([4, 12]), err).numpy().all())

    def test_arq_cali_pytorch_channelwise_T_withoffset_T(self):
        data_list = [[-1.0]*12, [0.0]*12, [1.0]*12, [-1.0, 0.0, 1.0]*4]
        input_data = torch.tensor(data_list, device=DEVICE)
        scale, offset, output_data = arq_cali_pytorch(input_data, 8, True, True)

        scale_except = torch.tensor([0.003923, 1.000000, 0.003923, 0.007844], device=DEVICE)
        scale_err = torch.abs(scale_except - scale).to('cpu')
        offset_except = torch.tensor([127, -128, -128, -1], dtype=torch.int32, device=DEVICE)
        err = torch.abs(output_data - input_data).to('cpu')

        self.assertTrue(torch.ge(1e-4*torch.ones([1]), scale_err).numpy().all())
        self.assertTrue(torch.equal(offset_except, offset))
        self.assertTrue(torch.ge(1e-2*torch.ones([4, 12]), err).numpy().all())

    def test_arq_real_pytorch(self):
        data_list = [[-1.0]*12, [0.0]*12, [1.0]*12, [-1.0, 0.0, 1.0]*4]
        scale_list = [0.003923, 1.000000, 0.003923, 0.007844]
        offset_list = [127, -128, -128, -1]

        input_data = torch.tensor(data_list, device=DEVICE)
        scale = torch.tensor(scale_list, device=DEVICE)
        offset = torch.tensor(offset_list, device=DEVICE, dtype=torch.int32)
        output_data = arq_real_pytorch(input_data, scale, offset, 8)

        out_list = [[-128]*12, [-128]*12, [127]*12, [-128, -1, 126]*4]
        out_except = torch.tensor(out_list, dtype=torch.int8, device=DEVICE)

        self.assertTrue(torch.equal(out_except, output_data))
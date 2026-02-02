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
import torch
from unittest import mock
from unittest.mock import patch

from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.data_utils import convert_precision, cal_shared_exponent
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.vars import MXFP4_E2M1

def fake_abs(tensor):
    return torch.ones_like(tensor).to(tensor.dtype)

def fake_pow(a, b):
    if isinstance(a, torch.Tensor):
        return torch.ones_like(a).to(a.dtype)
    elif isinstance(b, torch.Tensor):
        return torch.ones_like(b).to(b.dtype)

def fake_log2(tensor):
    return torch.ones_like(tensor).to(tensor.dtype)
 
def fake_floor(tensor):
    return torch.ones_like(tensor).to(tensor.dtype)
 
def fake_where(tensor, x, y):
    return torch.ones_like(tensor).to(torch.float32)

def fake_clamp(tensor, min=None, max=None):
    return torch.ones_like(tensor).to(tensor.dtype)

class ReturnTypes():
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices
 
def fake_max(tensor, dim=None, keepdim=None):
    if dim is not None:
        shape = list(tensor.shape)
        shape[dim] = 1
        ret_tensor = torch.ones(shape).to(tensor.dtype)
        if keepdim is not None and not keepdim:
            ret_tensor = ret_tensor.squeeze(dim)
        indices = torch.zeros_like(ret_tensor).to(torch.int64)
        return ReturnTypes(ret_tensor, indices)
    else:
        return tensor.to(torch.float32).max().to(tensor.dtype)


class TestDataUtil(unittest.TestCase):
    """
    The UT for evaluator helper
    """
    @classmethod
    def setUpClass(cls):
        print("Test Data Utils start!")

    @classmethod
    def tearDownClass(cls):
        print("Test Data Utils end!")

    def test_convert_precision_int4_data_success(self):
        tensor = torch.tensor([2.2, 1.2, 129.1, -129.3])
        converted_tensor = convert_precision(tensor, 'INT4', None)
        self.assertTrue((converted_tensor == torch.tensor([2., 1., 7., -8.])).all())

    def test_convert_precision_int8_data_success(self):
        tensor = torch.tensor([2.2, 1.2, 129.1, -129.3])
        converted_tensor = convert_precision(tensor, 'INT8', None)
        self.assertTrue((converted_tensor == torch.tensor([2., 1., 127., -128.])).all())

    def test_convert_precision_int16_data_success(self):
        tensor = torch.tensor([2.2, 1.2, 32769.1, -32769.3])
        converted_tensor = convert_precision(tensor, 'INT16', None)
        self.assertTrue((converted_tensor == torch.tensor([2., 1., 32767., -32768.])).all())

    def test_convert_precision_int32_data_success(self):
        tensor = torch.tensor([2.2, 1.2, -2147483660, 2147483680])
        converted_tensor = convert_precision(tensor, 'INT32', None)
        self.assertTrue((converted_tensor == torch.tensor([2., 1., -2147483648., 2147483647.])).all())


    def test_cal_shared_exponents_01(self):
        input_tensor = torch.tensor([[0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9], [0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9]]).to(torch.bfloat16)
        shared_exponents = cal_shared_exponent(input_tensor, MXFP4_E2M1)
        self.assertTrue(shared_exponents.shape == torch.Size([2,1]))


    def test_cal_shared_exponents_02(self):
        input_tensor = torch.tensor([[0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9, 1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9], [0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9, 1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9]]).to(torch.bfloat16)
        shared_exponents = cal_shared_exponent(input_tensor, MXFP4_E2M1)
        self.assertTrue(shared_exponents.shape == torch.Size([2,1]))

    def test_cal_shared_exponents_03(self):
        input_tensor = torch.tensor([[0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9, 1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9], [0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9, 1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9]]).to(torch.float32)
        shared_exponents = cal_shared_exponent(input_tensor, MXFP4_E2M1)
        self.assertTrue(shared_exponents.shape == torch.Size([2,1]))
        self.assertTrue((shared_exponents == torch.tensor([[1], [1]], dtype=torch.int8)).all())
        

    def test_bf16_mxfp4_success_01(self):
        input_tensor = torch.tensor([[0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9], [0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9]]).to(torch.bfloat16)
        output_tensor = convert_precision(input_tensor, MXFP4_E2M1, 'RINT')
        self.assertTrue(output_tensor.shape == input_tensor.shape)

    def test_bf16_fp4e1m2_bf16_success_01(self):
        input_tensor = torch.tensor([[-2, -1.5, -1.4, -1.25, -1.24, -1.0, -0.9, -0.75, -0.7, -0.5, -0.25, -0.22, -0.1, 0],
            [0, 0.1, 0.22, 0.25, 0.5, 0.7, 0.75, 0.9, 1.0, 1.24, 1.25, 1.4, 1.5, 2]]).to(torch.bfloat16)
        output_tensor = convert_precision(input_tensor, 'FLOAT4_E1M2', 'RINT')
        self.assertTrue((output_tensor == torch.tensor([[-1.7500, -1.5000, -1.5000, -1.2500, -1.2500, -1.0000, -1.0000, -0.7500,
         -0.7500, -0.5000, -0.2500, -0.2500, -0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.2500,  0.2500,  0.5000,  0.7500,  0.7500,  1.0000,
          1.0000,  1.2500,  1.2500,  1.5000,  1.5000,  1.7500]],
       dtype=torch.bfloat16)).all())

    def test_fp16_fp4e1m2_fp16_success_01(self):
        input_tensor = torch.tensor([[-2, -1.5, -1.4, -1.25, -1.24, -1.0, -0.9, -0.75, -0.7, -0.5, -0.25, -0.22, -0.1, 0],
            [0, 0.1, 0.22, 0.25, 0.5, 0.7, 0.75, 0.9, 1.0, 1.24, 1.25, 1.4, 1.5, 2]]).to(torch.float16)
        output_tensor = convert_precision(input_tensor, 'FLOAT4_E1M2', 'RINT')
        self.assertTrue((output_tensor == torch.tensor([[-1.7500, -1.5000, -1.5000, -1.2500, -1.2500, -1.0000, -1.0000, -0.7500,
         -0.7500, -0.5000, -0.2500, -0.2500, -0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.2500,  0.2500,  0.5000,  0.7500,  0.7500,  1.0000,
          1.0000,  1.2500,  1.2500,  1.5000,  1.5000,  1.7500]],
       dtype=torch.float16)).all())

    def test_bf16_fp4e2m1_fp16_success_01(self):
        input_tensor = torch.tensor([[0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9], [0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9]]).to(torch.bfloat16)
        output_tensor = convert_precision(input_tensor, 'FLOAT4_E2M1', 'RINT')
        self.assertTrue((output_tensor==torch.tensor([[0.0000, 0.0000, 0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.5000,
         2.0000, 2.0000, 2.0000, 2.0000, 3.0000, 4.0000, 4.0000, 4.0000, 4.0000,
         6.0000, 6.0000],
        [0.0000, 0.0000, 0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.5000,
         2.0000, 2.0000, 2.0000, 2.0000, 3.0000, 4.0000, 4.0000, 4.0000, 4.0000,
         6.0000, 6.0000]],dtype=torch.bfloat16)).all())
    
    def test_fp16_fp4e2m1_fp16_success_01(self):
        input_tensor = torch.tensor([[0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9], [0.0, 0.1, 0.25, 0.26, 0.75, 0.76, 1.24, 1.25, 
            1.26, 1.75, 1.76, 2.4, 2.5, 2.6, 3.5, 3.6, 4.9, 5.0, 5.1, 9]]).to(torch.float16)
        output_tensor = convert_precision(input_tensor, 'FLOAT4_E2M1', 'RINT')
        self.assertTrue((output_tensor==torch.tensor([[0.0000, 0.0000, 0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.5000,
         2.0000, 2.0000, 2.0000, 2.0000, 3.0000, 4.0000, 4.0000, 4.0000, 4.0000,
         6.0000, 6.0000],
        [0.0000, 0.0000, 0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.5000,
         2.0000, 2.0000, 2.0000, 2.0000, 3.0000, 4.0000, 4.0000, 4.0000, 4.0000,
         6.0000, 6.0000]],dtype=torch.float16)).all())

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
from unittest import mock
from unittest.mock import patch
import torch
import numpy as np
import torch.nn as nn
import amct_pytorch.graph_based_compression.amct_pytorch
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.fake_quant.weight_fake_quant_module import \
    FakeWeightQuantizedLinear, LutFakeWeightQuantizedLinear
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.utils import get_algo_params, get_quant_factor, apply_awq_quantize_weight
from collections import OrderedDict
CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
DEVICE = torch.device('cpu')

np.random.seed(0)

FP16 = torch.float16
BF16 = torch.bfloat16

class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        # 初始化一个线性层，输入维度为input_dim，输出维度为output_dim
        self.l1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 前向传播函数，通过线性层进行计算
        out = self.l1(x)
        return out

class ReturnTypes():
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices
 
def fake_abs(tensor):
    return torch.ones_like(tensor).to(tensor.dtype)
 
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
def fake_log2(tensor):
    return torch.ones_like(tensor).to(tensor.dtype)
 
def fake_floor(tensor):
    return torch.ones_like(tensor).to(tensor.dtype)
def fake_where(tensor, x, y):
    return torch.ones_like(tensor).to(torch.float32)
def fake_min(tensor, dim=None, keepdim=None):
    if dim is not None:
        shape = list(tensor.shape)
        shape[dim] = 1
        ret_tensor = torch.ones(shape).to(tensor.dtype)
        if keepdim is not None and not keepdim:
            ret_tensor = ret_tensor.squeeze(dim)
        indices = torch.zeros_like(ret_tensor).to(torch.int64)
        return ReturnTypes(ret_tensor, indices)
    else:
        return tensor.to(torch.float32).min().to(tensor.dtype)
def fake_pow(a, b):
    if isinstance(a, torch.Tensor):
        return torch.ones_like(a).to(a.dtype)
    elif isinstance(b, torch.Tensor):
        return torch.ones_like(b).to(b.dtype)
 
def fake_clamp(tensor, min=None, max=None):
    return torch.ones_like(tensor).to(tensor.dtype)
def fake_sqrt(tensor):
    return torch.tensor(1).to(tensor.dtype)
 
class TestFakeWeightQuantizedLinearModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_weight_fake_quant_module')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.quant_factor_path = os.path.join(cls.temp_folder, 'quant_factor.pt')

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @torch.no_grad()
    @patch.object(torch, 'where', fake_where)
    @patch.object(torch, 'floor', fake_floor)
    @patch.object(torch, 'log2', fake_log2)
    @patch.object(torch, 'sqrt', fake_sqrt)
    @patch.object(torch, 'min', fake_min)
    @patch.object(torch, 'max', fake_max)
    @patch.object(torch, 'abs', fake_abs)
    @patch.object(torch, 'clamp', fake_clamp)
    @patch.object(torch, 'pow', fake_pow)
    def test_weight_fake_quant_linear_module_noawq_success(self):
        #awq不使能
        # self.model = SimpleLinearModel(5, 10).to(FP16)
        # for name, module in self.model.named_modules():
        #     if name=='l1':
        #         self.module_name = 'l1'
        #         self.module = module
        # quant_params = OrderedDict([('wts_type', 'MXFP4_E2M1'), ('weight_compress_only', True)])
        # fake_quant = FakeWeightQuantizedLinear(self.module, quant_params, self.module_name)

        self.model = SimpleLinearModel(5, 10).to(BF16)
        for name, module in self.model.named_modules():
            if name=='l1':
                self.module_name = 'l1'
                self.module = module
        quant_params = OrderedDict([('wts_type', 'MXFP4_E2M1'), ('weight_compress_only', True)])
        fake_quant = FakeWeightQuantizedLinear(self.module, quant_params, self.module_name)
    
    @torch.no_grad()
    @patch.object(torch, 'where', fake_where)
    @patch.object(torch, 'floor', fake_floor)
    @patch.object(torch, 'log2', fake_log2)
    @patch.object(torch, 'sqrt', fake_sqrt)
    @patch.object(torch, 'min', fake_min)
    @patch.object(torch, 'max', fake_max)
    @patch.object(torch, 'abs', fake_abs)
    @patch.object(torch, 'clamp', fake_clamp)
    @patch.object(torch, 'pow', fake_pow)
    def test_weight_fake_quant_linear_module_awq_success(self):
        #awq使能，scale clipmax参数正常
        # self.model = SimpleLinearModel(5, 10).to(FP16)
        # for name, module in self.model.named_modules():
        #     if name=='l1':
        #         self.module_name = 'l1'
        #         self.module = module
        # scale = torch.tensor([[1,1,1,2,2,]], dtype=FP16)
        # clip_max = torch.randn([10,1,1], dtype=FP16)
        # torch.save({'l1':{'scale':scale,'clip_max':clip_max}}, self.awq_params_file_path)
        # quant_params = OrderedDict([('wts_type', 'MXFP4_E2M1'), ('weight_compress_only', True), ('awq_result_path', self.awq_params_file_path)])
        # fake_quant = FakeWeightQuantizedLinear(self.module, quant_params, self.module_name)

        self.model = SimpleLinearModel(5, 10).to(BF16)
        for name, module in self.model.named_modules():
            if name=='l1':
                self.module_name = 'l1'
                self.module = module
        scale_w = torch.randn([10,1,1]).to(BF16)
        scale = torch.tensor([[1,1,1,2,2,]]).to(BF16)
        clip_max = torch.randn([10,1,1]).to(BF16)
        torch.save({'l1': {'awq_quantize': {'scale':scale,'clip_max':clip_max}, 'quant_factors': {'scale_w': scale_w}}}, self.quant_factor_path)
        quant_params = OrderedDict([('wts_type', 'MXFP4_E2M1'), ('weight_compress_only', True), ('group_size', 128), ('quant_result_path', self.quant_factor_path)])
        fake_quant = FakeWeightQuantizedLinear(self.module, quant_params, self.module_name)
    
    @torch.no_grad()
    @patch.object(torch, 'where', fake_where)
    @patch.object(torch, 'floor', fake_floor)
    @patch.object(torch, 'log2', fake_log2)
    @patch.object(torch, 'sqrt', fake_sqrt)
    @patch.object(torch, 'min', fake_min)
    @patch.object(torch, 'max', fake_max)
    @patch.object(torch, 'abs', fake_abs)
    @patch.object(torch, 'clamp', fake_clamp)
    @patch.object(torch, 'pow', fake_pow)
    def test_weight_fake_quant_linear_module_awq_noclip_success(self):
        #awq使能，scale 参数正常
        self.model = SimpleLinearModel(5, 10).to(BF16)
        for name, module in self.model.named_modules():
            if name=='l1':
                self.module_name = 'l1'
                self.module = module
        scale = torch.tensor([[1,1,1,2,2,]]).to(BF16)
        scale_w = torch.tensor([1]).to(BF16)
        torch.save({'l1': {'awq_quantize': {'scale':scale}, 'quant_factors': {'scale_w': scale_w}}}, self.quant_factor_path)
        quant_params = OrderedDict([('wts_type', 'MXFP4_E2M1'), ('weight_compress_only', True), ('quant_result_path', self.quant_factor_path)])
        fake_quant = FakeWeightQuantizedLinear(self.module, quant_params, self.module_name)

    def test_get_awq_params_succ(self):
        scale = torch.tensor([[1,1,1,2,2,]]).to(BF16)
        clip_max = torch.randn([10,1,1]).to(BF16)
        module_name = 'l1'
        algo_params = {'awq_quantize':{'scale':scale,'clip_max':clip_max}}
        algo = 'awq_quantize'
        torch.save({'l1':algo_params}, self.quant_factor_path)
        quant_params = OrderedDict([('wts_type', 'MXFP4_E2M1'), ('weight_compress_only', True), ('quant_result_path', self.quant_factor_path)])
        awq = get_algo_params(quant_params, module_name, algo)
        self.assertNotEqual(awq, None)

        algo_params = {'awq_quantize':{'scale':scale,}}
        torch.save({'l1':algo_params}, self.quant_factor_path)
        quant_params = OrderedDict([('wts_type', 'MXFP4_E2M1'), ('weight_compress_only', True), ('quant_result_path', self.quant_factor_path)])
        awq = get_algo_params(quant_params, module_name, algo)
        self.assertNotEqual(awq, None)

        algo_params = {'awq_quantize':{'scale':scale,}}
        torch.save({'l1':algo_params}, self.quant_factor_path)
        quant_params = OrderedDict([('wts_type', 'MXFP4_E2M1'), ('weight_compress_only', True), ('quant_result_path', self.quant_factor_path)])
        awq = get_algo_params(quant_params, module_name, algo)
        self.assertNotEqual(awq, None)

    def test_quant_factor_path_succ(self):
        scale = torch.tensor([[0.3669]])
        module_name = 'matmul_1'
        torch.save({'matmul_1': {'quant_factors': {'scale': scale}}}, self.quant_factor_path)
        quant_params = OrderedDict([('wts_type', 'FLOAT4_E2M1'), ('weight_compress_only', True), ('group_size', 128), ('quant_result_path', self.quant_factor_path)])
        quant_factor = get_quant_factor(quant_params, module_name)
        self.assertNotEqual(quant_factor, None)

    def test_quant_factor_path_failed(self):
        scale = torch.tensor([[0.3669]])
        module_name = 'matmul_1'
        torch.save({'matmul_1': {'quant_factors': {'scale': scale}}}, self.quant_factor_path)
        quant_params = OrderedDict([('wts_type', 'FLOAT4_E2M1'), ('weight_compress_only', True), ('group_size', 128), ('quant_result_path', ' ')])
        try:
            quant_factor = get_quant_factor(quant_params, module_name)
        except BaseException as e:
            print(e)

    def test_get_awq_params_failed(self):
        scale = torch.tensor([[1,1,1,2,2,]]).to(BF16)
        clip_max = torch.randn([10,1,1]).to(BF16)
        module_name = 'l1'
        torch.save({'l1':{'awq_quantize': {'scale':scale,'clip_max':clip_max}}}, self.quant_factor_path)
        quant_params = OrderedDict([('wts_type', 'MXFP4_E2M1'), ('weight_compress_only', True), ('quant_result_path', '')])
        try:
            awq = get_algo_params(quant_params, module_name, 'awq_quantize')
        except BaseException as e:
            print(e)

        torch.save({'l2':{'awq_quantize': {'scale':scale,'clip_max':clip_max}}}, self.quant_factor_path)
        quant_params = OrderedDict([('wts_type', 'MXFP4_E2M1'), ('weight_compress_only', True), ('quant_result_path', self.quant_factor_path)])
        try:
            awq = get_algo_params(quant_params, module_name, 'awq_quantize')
        except BaseException as e:
            print(e)

        torch.save({'l1':{'awq_quantize': {'clip_max':clip_max}}}, self.quant_factor_path)
        quant_params = OrderedDict([('wts_type', 'MXFP4_E2M1'), ('weight_compress_only', True), ('quant_result_path', self.quant_factor_path)])
        try:
            awq = get_algo_params(quant_params, module_name, 'awq_quantize')
        except BaseException as e:
            print(e)

    @torch.no_grad()
    @patch.object(torch, 'where', fake_where)
    @patch.object(torch, 'floor', fake_floor)
    @patch.object(torch, 'log2', fake_log2)
    @patch.object(torch, 'sqrt', fake_sqrt)
    @patch.object(torch, 'min', fake_min)
    @patch.object(torch, 'max', fake_max)
    @patch.object(torch, 'abs', fake_abs)
    @patch.object(torch, 'clamp', fake_clamp)
    @patch.object(torch, 'pow', fake_pow)
    def test_apply_awq_quantize_weight_succ(self):
        weight = torch.ones((10,5)).to(BF16)
        print(weight)
        scale = torch.ones((1, 5)).to(BF16)
        clip_max = torch.ones((10, 1, 1)).to(BF16)
        awq_param = {'scale':scale,'clip_max':clip_max}
        weight_quant = apply_awq_quantize_weight(weight, awq_param, 128)
        print(weight)
        self.assertTrue(torch.equal(weight_quant, weight))

        awq_param = {'scale':scale}
        weight_quant = apply_awq_quantize_weight(weight, awq_param, 128)
        print(weight)
        self.assertTrue(torch.equal(weight_quant, weight))


    def test_apply_awq_quantize_weight_failed(self):
        weight = torch.randn([10,5]).to(BF16)

        scale = torch.ones((1, 3)).to(BF16)
        clip_max = torch.ones((10, 1, 1)).to(BF16)
        awq_param = {'scale':scale,'clip_max':clip_max}

        try:
            weight_quant = apply_awq_quantize_weight(weight, awq_param, 128)
        except BaseException as e:
            print(e)

        scale = torch.ones((1, 5)).to(BF16)
        clip_max = torch.ones((8, 1, 1)).to(BF16)
        awq_param = {'scale':scale,'clip_max':clip_max}

        try:
            weight_quant = apply_awq_quantize_weight(weight, awq_param, 128)
        except BaseException as e:
            print(e)
            
        scale = torch.ones((1, 5)).to(BF16)
        clip_max = torch.ones((10, 1, 1)).to(BF16)
        awq_param = {'scale':scale,'clip_max':clip_max}
        try:
            weight_quant = apply_awq_quantize_weight(weight, awq_param, 0)
        except BaseException as e:
            print(e)

class TestLutFakeWeightQuantizedLinear(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'temp_lut_weight_fake_quant')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
 
    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def testDown(self):
        pass

    def test_lut_weight_fake_quantized_linear_init_failed_wrong_dtype(self):
        mod = torch.nn.Linear(3,3)
        mod = mod.to(dtype=torch.float16)
        quant_params = {'quant_result_path': './temp_lut_weight_fake_quant/lut.pt'}
        self.assertRaises(RuntimeError,
            LutFakeWeightQuantizedLinear, mod, quant_params, 'linear_1')

    def test_lut_weight_fake_quantized_linear_init_failed_wrong_lut_length(self):
        mod = torch.nn.Linear(257, 1)
        lut = torch.randn(1, 16)
        lut_path = os.path.join(self.temp_folder, 'lut.pt')
        torch.save({'linear_1': {'lut_quantize': {'lut_table': lut}}}, lut_path)
        quant_params = {'wts_type': 'INT4', 'quant_result_path': lut_path}
        self.assertRaises(RuntimeError,
            LutFakeWeightQuantizedLinear, mod, quant_params, 'linear_1')

    def test_lut_weight_fake_quantized_linear_init_failed_no_lut_save_path(self):
        mod = torch.nn.Linear(257, 1)
        quant_params = {}
        self.assertRaises(RuntimeError,
            LutFakeWeightQuantizedLinear, mod, quant_params, 'linear_1')

    def test_lut_weight_fake_quantized_linear_infer_success_01(self):
        mod = torch.nn.Linear(256, 24)
        lut = torch.randn(24 * 1, 16)
        lut_path = os.path.join(self.temp_folder, 'lut.pt')
        torch.save({'linear_1': {'lut_quantize': {'lut_table': lut}}}, lut_path)
        quant_params = {'wts_type': 'INT4', 'quant_result_path': lut_path}
        quantized_mod = LutFakeWeightQuantizedLinear(mod, quant_params, 'linear_1')

        input_data = torch.randn(32, 256)
        out = quantized_mod(input_data)
        self.assertEqual(list(out.shape), [32, 24])

    def test_lut_weight_fake_quantized_linear_infer_success_02(self):
        mod = torch.nn.Linear(4097, 4096)
        lut = torch.randn(4096 * 17, 16)
        lut_path = os.path.join(self.temp_folder, 'lut.pt')
        torch.save({'linear_1': {'lut_quantize': {'lut_table': lut}}}, lut_path)
        quant_params = {'wts_type': 'INT4', 'quant_result_path': lut_path}
        quantized_mod = LutFakeWeightQuantizedLinear(mod, quant_params, 'linear_1')

        input_data = torch.randn(4096, 4097)
        out = quantized_mod(input_data)
        self.assertEqual(list(out.shape), [4096, 4096])

    def test_lut_weight_fake_quantized_linear_infer_success_03(self):
        # check output weight value
        mod = torch.nn.Linear(15, 1)
        mod.weight.data = torch.Tensor([[0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3],
                                       [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3]]).reshape(2, 15)
        lut = torch.Tensor([[0.5, 0.6, 0.8, 0.91, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3],
                           [0.5, 0.6, 0.8, 0.91, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3]]).reshape(2, 16)
        
        lut_path = os.path.join(self.temp_folder, 'lut.pt')
        torch.save({'linear_1': {'lut_quantize': {'lut_table': lut}}}, lut_path)
        quant_params = {'wts_type': 'INT4', 'quant_result_path': lut_path}
        quantized_mod = LutFakeWeightQuantizedLinear(mod, quant_params, 'linear_1')
        self.assertTrue((quantized_mod.quantized_weight == \
            torch.Tensor([[0.5, 0.6, 0.91, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3],
                         [0.5, 0.6, 0.91, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3]])).all())

    # def test_get_lut_tables_failed_lut_result_path_not_exists(self):
    #     self.assertRaises(RuntimeError, LutFakeWeightQuantizedLinear.get_lut_tables, {}, 'linear_1')
 
    # def test_get_lut_tables_failed_lut_result_path_wrong(self):
    #     self.assertRaises(RuntimeError, LutFakeWeightQuantizedLinear.get_lut_tables,
    #                       {'wts_type': 'INT4', 'lut_result_path': 'not_exist_path'}, 'linear_1')
 
    # def test_get_lut_tables_failed_layer_name_not_exists(self):
    #     lut_path = os.path.join(self.temp_folder, 'lut.pt')
    #     lut = torch.randn(1,1,16)
    #     torch.save({'linear_2': lut}, lut_path)
    #     self.assertRaises(RuntimeError, LutFakeWeightQuantizedLinear.get_lut_tables,
    #                       {'wts_type': 'INT4', 'lut_result_path': lut_path}, 'linear_1')

    # def test_lut_weight_fake_quantized_linear_init_failed_wrong_lut_dtype(self):
    #     mod = torch.nn.Linear(4097, 32)
    #     lut = torch.randn(32 * 17, 16).to(torch.float16)
    #     lut_path = os.path.join(self.temp_folder, 'lut.pt')
    #     torch.save({'linear_1': lut}, lut_path)
    #     quant_params = {'wts_type': 'INT4', 'lut_result_path': lut_path}
    #     self.assertRaises(RuntimeError, LutFakeWeightQuantizedLinear, mod, quant_params, 'linear_1')
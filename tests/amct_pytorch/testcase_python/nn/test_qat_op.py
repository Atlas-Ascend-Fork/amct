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
import os
import copy
import shutil
from collections import defaultdict

import torch
from torch import nn
import numpy as np
import onnx
import onnxruntime as ort

from amct_pytorch.graph_based_compression.amct_pytorch.utils.log import LOGGER
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.utils import copy_tensor
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.conv1d import Conv1dQAT
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.conv_transpose_2d import ConvTranspose2dQAT
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.conv_transpose_1d import ConvTranspose1dQAT
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.conv3d import Conv3dQAT
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.linear import LinearQAT
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.matmul import MatMulQAT

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


quant_configs = [
    {
        'retrain_weight_config': {},
        'retrain_data_config': {}
    },

    {
        'retrain_weight_config':
            {'weights_retrain_algo': 'ulq_retrain'},
        'retrain_data_config':
            {'batch_num': 3}
    },

    {
        'retrain_weight_config':
            {'channel_wise': False},
        'retrain_data_config':
            {'batch_num': 3,
             'clip_min': -1.0,
             'clip_max': 1.0}
    },

    {
        'retrain_weight_config':
            {'weights_retrain_algo': 'ulq_retrain',
             'channel_wise': False
            },
        'retrain_data_config':
            {'batch_num': 3,
             'fixed_min': True}
    },

    {
        'retrain_weight_config':
            {'weights_retrain_algo': 'ulq_retrain',
             'channel_wise': False},
        'retrain_data_config':
            {'batch_num': 3,
             'clip_min': -1.0,
             'clip_max': 1.0,
             'fixed_min': False}
    },

    {
        'retrain_weight_config':
            {'dst_type': 'INT8',
             'weights_retrain_algo': 'ulq_retrain'},
        'retrain_data_config':
            {'dst_type': 'INT8',
             'batch_num': 3,
             'clip_min': -1.0,
             'clip_max': 1.0,
             'fixed_min': False}
    }
]


def similarity(data0, data1):
    data0_nan = np.isnan(data0)
    data0[data0_nan] = 1
    data1_nan = np.isnan(data1)
    data1[data1_nan] = 1
    similarity = np.sum(np.multiply(data0, data1).astype(np.float64))\
                    /(np.sqrt(np.sum(data0.astype(np.float64)**2))\
                    *np.sqrt(np.sum(data1.astype(np.float64)**2)))*100
    if (data0 == data1).all():
        similarity = 100
    if np.isnan(similarity) or np.isinf(similarity):
        data0 = np.divide(data0,np.power(10,38))
        data1 = np.divide(data1,np.power(10,38))
        similarity = np.sum(np.multiply(data0, data1).astype(np.float64))\
                    /(np.sqrt(np.sum(data0.astype(np.float64)**2))\
                    *np.sqrt(np.sum(data1.astype(np.float64)**2)))*100
        if np.isnan(similarity) or np.isinf(similarity):
            data0 = np.divide(data0,np.power(10,38))
            data1 = np.divide(data1,np.power(10,38))
            similarity = np.sum(np.multiply(data0, data1).astype(np.float64))\
                    /(np.sqrt(np.sum(data0.astype(np.float64)**2))\
                    *np.sqrt(np.sum(data1.astype(np.float64)**2)))*100
    if np.isnan(similarity):
        similarity = 0
    return similarity

class TestQatOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        if os.path.exists('./amct_log/amct_pytorch.graph_based_compression.amct_pytorch.log'):
            os.remove('./amct_log/amct_pytorch.graph_based_compression.amct_pytorch.log')
            LOGGER.logi('amct_pytorch.graph_based_compression.amct_pytorch.log is initialized successfully.')

    def testDown(self):
        pass

    def test_qat_base_init_success(self):
        quant_conf = {
            'retrain_data_config': {
                'clip_max': 1.0,
                'clip_min': -1.0
            }
        }
        mod = Conv2dQAT(1, 1, 1, config=quant_conf)
        self.assertTrue(isinstance(mod.retrain_data_config, dict))
        self.assertTrue(isinstance(mod.retrain_weight_config, dict))
        self.assertEqual(mod.retrain_data_config.get('clip_max'), 1.0)
        self.assertEqual(mod.retrain_data_config.get('clip_min'), -1.0)
        self.assertEqual(mod.act_num_bits, 8)
        self.assertEqual(mod.wts_num_bits, 8)

    def test_qat_base_init_failed_wrong_config_data_type(self):
        wrong_quant_conf = {
            'retrain_data_config': {
                'dst_type': 1,
                'batch_num': '1',
                'fixed_min': 1,
                'clip_max': '1.0',
                'clip_min': '-1.0'
            },
            'retrain_weight_config': {
                'dst_type': 1,
                'weights_retrain_algo': 1,
                'channel_wise': 1
            }
        }
        with self.assertRaises(ValueError) as cm:
            Conv2dQAT(1, 1, 1, config=wrong_quant_conf)
            with open('./amct_log/amct_pytorch.graph_based_compression.amct_pytorch.log', 'r') as f:
                log_content = f.read()
                for item in wrong_quant_conf.get('retrain_data_config').keys():
                    self.assertIn(item, log_content)
            with open('./amct_log/amct_pytorch.graph_based_compression.amct_pytorch.log', 'r') as f:
                log_content = f.read()
                for item in wrong_quant_conf.get('retrain_weight_config').keys():
                    self.assertIn(item, log_content)

    def test_qat_base_init_failed_clip_max_min_not_both_set(self):
        wrong_quant_conf = {
            'retrain_data_config': {
                'clip_max': 1.0,
            }
        }
        Conv2dQAT(1, 1, 1, config=wrong_quant_conf)

    def test_qat_base_init_failed_wrong_config_data_scope(self):
        wrong_quant_conf = {
            'retrain_data_config': {
                'dst_type': 1,
            },
            'retrain_weight_config': {
                'dst_type': 1,
                'weights_retrain_algo': '1',
            }
        }
        with self.assertRaises(ValueError) as cm:
            Conv2dQAT(1, 1, 1, config=wrong_quant_conf)
            with open('./amct_log/amct_pytorch.graph_based_compression.amct_pytorch.log', 'r') as f:
                log_content = f.read()
                for item in wrong_quant_conf.get('retrain_data_config').keys():
                    self.assertIn(item, log_content)
            with open('./amct_log/amct_pytorch.graph_based_compression.amct_pytorch.log', 'r') as f:
                log_content = f.read()
                for item in wrong_quant_conf.get('retrain_weight_config').keys():
                    self.assertIn(item, log_content)

        wrong_quant_conf = {
            'retrain_data_config': {
                'batch_num': -1
            }
        }
        with self.assertRaises(ValueError) as cm:
            Conv2dQAT(1, 1, 1, config=wrong_quant_conf)

        wrong_quant_conf = {
            'retrain_data_config': {
                'clip_min': 1.0,
                'clip_max': 1.0
            }
        }
        with self.assertRaises(ValueError) as cm:
            Conv2dQAT(1, 1, 1, config=wrong_quant_conf)

        wrong_quant_conf = {
            'retrain_data_config': {
                'clip_min': -1.0,
                'clip_max': -1.0
            }
        }
        with self.assertRaises(ValueError) as cm:
            Conv2dQAT(1, 1, 1, config=wrong_quant_conf)

    def test_qat_base_register_qat_params(self):
        mod = Conv2dQAT(1, 1, 1)
        self.assertEqual(mod.cur_batch, torch.tensor(0))
        self.assertEqual(mod.acts_clip_max, torch.tensor(1.0))
        self.assertEqual(mod.acts_clip_min, torch.tensor(-1.0))
        self.assertTrue(np.isnan(mod.acts_clip_max_pre))
        self.assertTrue(np.isnan(mod.acts_clip_min_pre))
        self.assertTrue(np.isnan(mod.acts_scale))
        self.assertEqual(mod.acts_offset_deploy, torch.tensor([0]))
        self.assertTrue(np.isnan(mod.wts_scales))
        self.assertTrue(np.isnan(mod.wts_offsets))
        self.assertEqual(mod.wts_offsets_deploy, torch.tensor(0))
        self.assertEqual(mod.s_rec_flag, torch.tensor(False))

    def test_get_ori_op_params_redundant_params(self):
        bak_required_params = copy.deepcopy(Conv2dQAT._required_params)
        Conv2dQAT._required_params = ('err_param',)
        with self.assertRaises(RuntimeError) as cm:
            Conv2dQAT.from_float(torch.nn.Conv2d(1, 1, 1))
        Conv2dQAT._required_params = bak_required_params

    def test_check_qat_config_mismatch_dst_type(self):
        wrong_quant_conf = {
            'retrain_data_config': {
                'dst_type': 'INT16',
            },
            'retrain_weight_config': {
                'dst_type': 'INT16',
            }
        }
        with self.assertRaises(ValueError) as cm:
            Conv2dQAT(1, 1, 1, config=wrong_quant_conf)

    def test_qat_init_d16w8(self):
        quant_conf = {
            'retrain_data_config': {
                'dst_type': 'INT16',
            },
            'retrain_weight_config': {
                'dst_type': 'INT8',
            }
        }
        mod = Conv2dQAT(1, 1, 1, config=quant_conf)
        self.assertEqual(mod.act_num_bits, 16)
        self.assertEqual(mod.wts_num_bits, 8)


class TestConv2dQAT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def testDown(self):
        pass

    def test_conv2d_qat_from_float_success(self):
        mod = torch.nn.Conv2d(1, 1, 1, padding_mode='zeros')
        qat_mod = Conv2dQAT.from_float(mod)
        self.assertTrue(isinstance(qat_mod, Conv2dQAT))

    def test_conv2d_qat_from_float_failed_padding_mode_not_zeros(self):
        mod = torch.nn.Conv2d(1, 1, 1, padding_mode='reflect')
        with self.assertRaises(ValueError) as cm:
            Conv2dQAT.from_float(mod)

    def test_conv2d_qat_from_float_failed_ori_op_not_conv2d(self):
        mod = torch.nn.Conv3d(1, 1, 1)
        with self.assertRaises(TypeError) as cm:
            Conv2dQAT.from_float(mod)

    def test_conv2d_qat_forward(self):
        mod = Conv2dQAT(3, 16, 1)
        inputs = torch.randn((3, 3, 224, 224))
        output = mod.forward(inputs)
        self.assertIsNotNone(output)

    def test_conv2d_qat_forward_ulq_retrain(self):
        quant_config = {'retrain_enable': True,
                        'retrain_weight_config':
                            {'weights_retrain_algo': 'ulq_retrain'}
                        }
        mod = Conv2dQAT(3, 16, 1, config=quant_config)
        inputs = torch.randn((3, 3, 224, 224))
        output = mod.forward(inputs)
        self.assertIsNotNone(output)

    def test_conv2d_qat_forward_do_init_false(self):
        mod = Conv2dQAT(3, 16, 1)
        mod.do_init = False
        inputs = torch.randn((3, 3, 224, 224))
        output = mod.forward(inputs)
        self.assertIsNotNone(output)

    def test_conv2d_qat_forward_do_init_false_cur_batch_one(self):
        mod = Conv2dQAT(3, 16, 1)
        mod.do_init = False
        copy_tensor(mod.cur_batch, torch.tensor(1))
        inputs = torch.randn((3, 3, 224, 224))
        output = mod.forward(inputs)
        self.assertIsNotNone(output)

    def test_conv2d_qat_forward_do_init_false_cur_batch_two(self):
        mod = Conv2dQAT(3, 16, 1)
        mod.do_init = False
        copy_tensor(mod.cur_batch, torch.tensor(2))
        inputs = torch.randn((3, 3, 224, 224))
        output = mod.forward(inputs)
        self.assertIsNotNone(output)

    def test_conv2d_qat_unsupport_shape_inputs(self):
        mod = Conv2dQAT(3, 16, 1)
        with self.assertRaises(RuntimeError) as cm:
            ret = mod.forward(torch.randn((3, 224, 224)))


class TestConvTranspose2dQAT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def testDown(self):
        pass

    def test_conv_transpose_2d_qat_from_float_success(self):
        mod = torch.nn.ConvTranspose2d(1, 1, 1, padding_mode='zeros')
        qat_mod = ConvTranspose2dQAT.from_float(mod)
        self.assertTrue(isinstance(qat_mod, ConvTranspose2dQAT))

    def test_conv_transpose_2d_qat_from_float_failed_ori_op_not_conv_transpose_2d(self):
        mod = torch.nn.Conv2d(1, 1, 1)
        with self.assertRaises(TypeError) as cm:
            ConvTranspose2dQAT.from_float(mod)

    def test_conv_transpose_2d_qat_forward(self):
        mod = ConvTranspose2dQAT(3, 16, 1)
        inputs = torch.randn(3, 3, 224, 224)
        output = mod.forward(inputs)
        self.assertIsNotNone(output)

    def test_conv_transpose_2d_qat_padding_mode_not_zero(self):
        with self.assertRaises(ValueError) as cm:
            ConvTranspose2dQAT(1, 1, 1, padding_mode='reflect')

    def test_conv_transpose_2d_qat_forward_ulq_retrain(self):
        quant_config = {'retrain_enable': True,
                        'retrain_weight_config':
                            {'weights_retrain_algo': 'ulq_retrain'}
                        }
        mod = ConvTranspose2dQAT(3, 16, 1, config=quant_config)
        inputs = torch.randn((3, 3, 224, 224))
        output = mod.forward(inputs)
        self.assertIsNotNone(output)

    def test_conv_transpose_2d_qat_forward_do_init_false(self):
        mod = ConvTranspose2dQAT(3, 16, 1)
        mod.do_init = False
        inputs = torch.randn((3, 3, 224, 224))
        output = mod.forward(inputs)
        self.assertIsNotNone(output)

    def test_conv_transpose_2d_qat_forward_do_init_false_cur_batch_one(self):
        mod = ConvTranspose2dQAT(3, 16, 1)
        mod.do_init = False
        copy_tensor(mod.cur_batch, torch.tensor(1))
        inputs = torch.randn((3, 3, 224, 224))
        output = mod.forward(inputs)
        self.assertIsNotNone(output)

    def test_conv_transpose_2d_qat_forward_do_init_false_cur_batch_two(self):
        mod = ConvTranspose2dQAT(3, 16, 1)
        mod.do_init = False
        copy_tensor(mod.cur_batch, torch.tensor(2))
        inputs = torch.randn((3, 3, 224, 224))
        output = mod.forward(inputs)
        self.assertIsNotNone(output)

    def test_conv_transpose_2d_qat_unsupport_shape_inputs(self):
        mod = ConvTranspose2dQAT(3, 16, 1)
        with self.assertRaises(RuntimeError) as cm:
            output = mod.forward(torch.randn(3, 224, 224))


class TestConv3dQAT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def testDown(self):
        pass

    def test_conv3d_qat_limit_check_01(self):
        with self.assertRaises(RuntimeError):
            Conv3dQAT(1, 1, 1, padding_mode='reflect')

    def test_conv3d_qat_limit_check_02(self):
        with self.assertRaises(RuntimeError):
            Conv3dQAT(1, 1, 1, dilation=(1, 1))

    def test_conv3d_qat_limit_check_03(self):
        with self.assertRaises(RuntimeError):
            Conv3dQAT(1, 1, 1, dilation=(2, 1, 1))

    def test_conv3dqat_limit_check_04(self):
        qat_op = Conv3dQAT(1, 1, 1, dilation=(1, 1, 1))
        self.assertTrue(qat_op.do_init)

    def test_conv3d_qat_forward(self):
        qat_op = Conv3dQAT(3, 3, 4, dilation=(1, 2, 1))
        output = qat_op.forward(torch.randn(30, 3, 12, 64, 64))
        self.assertIsNotNone(output)

    def test_conv3d_qat_from_float_01(self):
        ori_op = torch.nn.Conv3d(3, 3, 4, dilation=(1, 2, 1))
        qat_op = Conv3dQAT.from_float(ori_op)
        output = qat_op.forward(torch.randn(30, 3, 12, 64, 64))
        self.assertIsNotNone(output)

    def test_conv3d_qat_from_float_02(self):
        ori_op = torch.nn.Conv2d(2, 3, 4)
        with self.assertRaises(RuntimeError) as cm:
            Conv3dQAT.from_float(ori_op)

    def test_conv3d_qat_unsupport_shape_inputs(self):
        qat_op = Conv3dQAT(3, 3, 4, dilation=(1, 2, 1))
        with self.assertRaises(RuntimeError) as cm:
            qat_op.forward(torch.randn(3, 12, 64, 64))

class TestLinearQAT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def testDown(self):
        pass

    def test_lineard_qat_limit_check_01(self):
        with self.assertRaises(RuntimeError) as cm:
            LinearQAT(1, 1,
                      config={'retrain_weight_config': {'channel_wise': True}})

    def test_lineard_qat_forward(self):
        qat_op = LinearQAT(16, 10,
                           config={'retrain_weight_config': {'channel_wise': False}})
        output = qat_op.forward(torch.rand((128, 16)))
        self.assertIsNotNone(output)

    def test_lineard_qat_from_float_01(self):
        ori_op = torch.nn.Linear(8, 10)
        qat_linear = LinearQAT.from_float(ori_op,
                             config={'retrain_weight_config': {'channel_wise': False}})
        self.assertIsNotNone(qat_linear)

    def test_lineard_qat_from_float_02(self):
        ori_op = torch.nn.Conv3d(1, 1, 1)
        with self.assertRaises(RuntimeError) as cm:
            LinearQAT.from_float(ori_op,
                                config={'retrain_weight_config': {'channel_wise': False}})

    def test_lineard_qat_channel_wise_default_false_00(self):
        config = {'retrain_weight_config': {'dst_type': 'INT8'}}
        LinearQAT(1, 1, config=config)
        self.assertEqual(config.get('retrain_weight_config'), {'dst_type': 'INT8'})

    def test_lineard_qat_channel_wise_default_false_01(self):
        config = None
        LinearQAT(1, 1, config=config)
        self.assertTrue(config is None)

def set_module(model, sub_module_name, module):
    tokens = sub_module_name.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

class NetConv1d(nn.Module):
    def __init__(self):
        super(NetConv1d, self).__init__()
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, bias=False),
            nn.BatchNorm1d(16))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        # depthwise_conv + bn
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, groups=16),
            nn.BatchNorm1d(16))
        self.layer4 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, groups=16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        # group_conv + bn
        self.layer5 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, groups=4),
            nn.BatchNorm1d(32))
        self.layer6 = nn.Sequential(
            nn.Conv1d(32, 8, kernel_size=3, groups=8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x

class NetConv1dQAT(nn.Module):
    def __init__(self):
        super(NetConv1dQAT, self).__init__()
        # conv + bn
        self.layer1 = nn.Sequential(
            Conv1dQAT(2, 16, kernel_size=3, bias=False),
            nn.BatchNorm1d(16))
        self.layer2 = nn.Sequential(
            Conv1dQAT(16, 16, kernel_size=3, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        # depthwise_conv + bn
        self.layer3 = nn.Sequential(
            Conv1dQAT(16, 16, kernel_size=3, groups=16),
            nn.BatchNorm1d(16))
        self.layer4 = nn.Sequential(
            Conv1dQAT(16, 16, kernel_size=3, groups=16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        # group_conv + bn
        self.layer5 = nn.Sequential(
            Conv1dQAT(16, 32, kernel_size=3, groups=4),
            nn.BatchNorm1d(32))
        self.layer6 = nn.Sequential(
            Conv1dQAT(32, 8, kernel_size=3, groups=8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x


class TestConv1dQAT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LOGGER.logi(f'torch version: {torch.__version__}')
        cls.temp_folder = os.path.join(CUR_DIR, 'test_quant_retrain_quant_fusion_model')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def testDown(self):
        pass

    def test_conv1d_qat_from_float_failed_ori_op_not_conv1d(self):
        mod = torch.nn.Conv3d(1, 1, 1)
        with self.assertRaises(TypeError) as cm:
            Conv1dQAT.from_float(mod)

    def test_conv1d_qat_scratch_from_zero(self):
        net_conv1d_qat = NetConv1dQAT()
        for i in range(5):
            ret = net_conv1d_qat.forward(torch.rand(1, 2, 28))
            self.assertIsNotNone(ret)

    def test_conv1_qat_convert_from_ori_op(self):
        net_conv1d = NetConv1d()
        idx = 0
        for name, module in net_conv1d.named_modules():
            if isinstance(module, nn.Conv1d):
                qat_module = Conv1dQAT.from_float(
                    module)
                set_module(net_conv1d, name, qat_module)
                idx += 1
        for i in range(5):
            ret = net_conv1d.forward(torch.rand(1, 2, 28))
            self.assertIsNotNone(ret)

    def test_conv1d_qat_unsupport_shape_inputs(self):
        net_conv1d_qat = NetConv1dQAT()
        with self.assertRaises(RuntimeError) as cm:
            ret = net_conv1d_qat.forward(torch.rand(1, 2, 28, 28))

    def test_conv1d_qat_op_not_support_padding_mode_not_zeros(self):
        self.assertRaises(ValueError, Conv1dQAT, 1, 1, 1, padding_mode='reflect')

    def test_conv1d_qat_not_support_dtype(self):
        mod = Conv1dQAT(1,1,1)
        mod = mod.to(torch.float64)
        self.assertRaises(ValueError, mod, torch.randn(1,1,1).to(torch.float64))


class NetConvTranspose1dQAT(nn.Module):
    def __init__(self):
        super(NetConvTranspose1dQAT, self).__init__()
        # conv + bn
        self.layer1 = nn.Sequential(
            ConvTranspose1dQAT(2, 16, kernel_size=3, bias=False,
                               config=quant_configs[0]),
            nn.BatchNorm1d(16))
        self.layer2 = nn.Sequential(
            ConvTranspose1dQAT(16, 16, kernel_size=3, bias=True,
                               config=quant_configs[1]),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        # depthwise_conv + bn
        self.layer3 = nn.Sequential(
            ConvTranspose1dQAT(16, 16, kernel_size=3,
                               config=quant_configs[2]),
            nn.BatchNorm1d(16))
        self.layer4 = nn.Sequential(
            ConvTranspose1dQAT(16, 16, kernel_size=3,
                               config=quant_configs[3]),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        # group_conv + bn
        self.layer5 = nn.Sequential(
            ConvTranspose1dQAT(16, 32, kernel_size=3,
                               config=quant_configs[4]),
            nn.BatchNorm1d(32))
        self.layer6 = nn.Sequential(
            ConvTranspose1dQAT(32, 8, kernel_size=3,
                               config=quant_configs[5]),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x


class NetConvTranspose1d(nn.Module):
    def __init__(self):
        super(NetConvTranspose1d, self).__init__()
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.ConvTranspose1d(2, 16, kernel_size=3, bias=False),
            nn.BatchNorm1d(16))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=3, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        # depthwise_conv + bn
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=3, groups=16),
            nn.BatchNorm1d(16))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=3, groups=16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        # group_conv + bn
        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32))
        self.layer6 = nn.Sequential(
            nn.ConvTranspose1d(32, 8, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x


class TestConvTranspose1dQAT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LOGGER.logi(f'torch version: {torch.__version__}')

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def testDown(self):
        pass

    def test_conv_transpose_1d_qat_scratch_from_zero(self):
        net_conv_transpose_1d_qat = NetConvTranspose1dQAT()
        for i in range(5):
            ret = net_conv_transpose_1d_qat.forward(torch.rand(1, 2, 28))
            self.assertIsNotNone(ret)

    def test_conv_transpose_1d_qat_convert_from_ori_op(self):
        net_conv_transpose_1d = NetConvTranspose1d()
        idx = 0
        for name, module in net_conv_transpose_1d.named_modules():
            if isinstance(module, nn.ConvTranspose1d):
                qat_module = ConvTranspose1dQAT.from_float(
                    module, config=quant_configs[idx])
                set_module(net_conv_transpose_1d, name, qat_module)
                idx += 1
        for i in range(5):
            ret = net_conv_transpose_1d.forward(torch.rand(1, 2, 28))
            self.assertIsNotNone(ret)

    def test_conv_transpose_1d_qat_unsupport_shape_inputs(self):
        net_conv_transpose_1d_qat = NetConvTranspose1dQAT()
        with self.assertRaises(RuntimeError) as cm:
            ret = net_conv_transpose_1d_qat.forward(torch.rand(2, 28))

class TestMatMulQAT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LOGGER.logi(f'torch version: {torch.__version__}')
        cls.temp_dir = os.path.join(CUR_DIR, 'tmp')
        os.makedirs(cls.temp_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        pass

    def testDown(self):
        pass
    
    def test_check_input_error_wrong_dtype(self):
        with self.assertRaises(ValueError):
            MatMulQAT.check_input(torch.randn(32,3).to(torch.float16))

    def test_check_input_error_wrong_shape(self):
        with self.assertRaises(RuntimeError):
            MatMulQAT.check_input(torch.randn(3,3,3,3,3,3,3))

        with self.assertRaises(RuntimeError):
            MatMulQAT.check_input(torch.randn(3,))

    def test_acts_quant_init_success(self):
        matmul_op = MatMulQAT(config={'retrain_data_config': {'batch_num': 2}})
        result = matmul_op.acts_quant_init(
            torch.randn(30, 30), matmul_op.acts_quant_params.get('input'), matmul_op.input_init_module)
        self.assertFalse(result)
        result = matmul_op.acts_quant_init(
            torch.randn(30, 30), matmul_op.acts_quant_params.get('input'), matmul_op.input_init_module)
        self.assertTrue(result)

        self.assertFalse(matmul_op.input_scale.isnan().all())
        self.assertFalse(matmul_op.input_clip_max.isnan().all())
        self.assertFalse(matmul_op.input_clip_min.isnan().all())
        self.assertFalse(matmul_op.input_clip_max_pre.isnan().all())
        self.assertFalse(matmul_op.input_clip_min_pre.isnan().all())

    def test_acts_quant_success(self):
        matmul_op = MatMulQAT(config={'retrain_data_config': {'batch_num': 2}})
        result = matmul_op.acts_quant(
            torch.randn(30, 30), matmul_op.acts_quant_params.get('input'))

        self.assertEqual(result.shape, (30, 30))
        self.assertFalse(matmul_op.input_scale.isnan().all())
        self.assertFalse(matmul_op.input_clip_max.isnan().all())
        self.assertFalse(matmul_op.input_clip_min.isnan().all())
        self.assertFalse(matmul_op.input_clip_max_pre.isnan().all())
        self.assertFalse(matmul_op.input_clip_min_pre.isnan().all())

    def test_matmul_qat_init_failed_wrong_device_dtype(self):
        with self.assertRaises(TypeError):
            MatMulQAT(device=3)

    def test_matmul_qat_init_failed_wrong_config_dtype(self):
        with self.assertRaises(TypeError):
            MatMulQAT(config=3)

    def test_matmul_qat_init_failed_wrong_config_item_dtype(self):
        with self.assertRaises(ValueError):
            MatMulQAT(config={'retrain_data_config': {'batch_num': '1'}})
            MatMulQAT(config={'retrain_data_config': {'dst_type': 1}})
            MatMulQAT(config={'retrain_data_config': {'fixed_min': '1'}})
            MatMulQAT(config={'retrain_data_config': {'clip_max': '1'}})
            MatMulQAT(config={'retrain_data_config': {'clip_min': '1'}})

    def test_matmul_qat_init_failed_wrong_config_item_scope(self):
        with self.assertRaises(ValueError):
            MatMulQAT(config={'retrain_data_config': {'dst_type': 'INT4'}})
            MatMulQAT(config={'retrain_data_config': {'batch_num': 0}})
            MatMulQAT(config={'retrain_data_config': {'clip_max': 0}})
            MatMulQAT(config={'retrain_data_config': {'clip_min': 0}})

    def test_matmul_qat_init_success(self):
        MatMulQAT(config={'retrain_data_config': {'dst_type': 'INT8'}})

    def test_matmul_op_infer_success(self):
        # norm data
        mod = MatMulQAT()
        for i in range(10):
            input = torch.randn(30, 30, 30)
            other = torch.randn(30, 30, 30)
            mod(input, other)
        self.onnx_path = os.path.join(self.temp_dir, 'matmul_qat.onnx')
        
        torch.onnx.export(mod, (input, other), self.onnx_path)
        ori_out = torch.matmul(input, other)
        ort_session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
        quantized_out = ort_session.run(
            None, {ort_session.get_inputs()[0].name: input.cpu().numpy(),
                   ort_session.get_inputs()[1].name: other.cpu().numpy()})
        sim = similarity(ori_out.detach().cpu().numpy(), quantized_out[0])
        self.assertTrue(sim > 99)
        
        # all zeros
        mod = MatMulQAT()
        for i in range(10):
            input = torch.zeros(30, 30, 30)
            other = torch.zeros(30, 30, 30)
            mod(input, other)
        self.onnx_path = os.path.join(self.temp_dir, 'matmul_qat.onnx')
        
        torch.onnx.export(mod, (input, other), self.onnx_path)
        ori_out = torch.matmul(input, other)
        ort_session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
        quantized_out = ort_session.run(
            None, {ort_session.get_inputs()[0].name: input.cpu().numpy(),
                   ort_session.get_inputs()[1].name: other.cpu().numpy()})
        sim = similarity(ori_out.detach().cpu().numpy(), quantized_out[0])
        self.assertTrue(sim > 99)
        
        # all ones
        mod = MatMulQAT()
        for i in range(10):
            input = torch.ones(30, 30, 30)
            other = torch.ones(30, 30, 30)
            mod(input, other)
        self.onnx_path = os.path.join(self.temp_dir, 'matmul_qat.onnx')
        
        torch.onnx.export(mod, (input, other), self.onnx_path)
        ori_out = torch.matmul(input, other)
        ort_session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
        quantized_out = ort_session.run(
            None, {ort_session.get_inputs()[0].name: input.cpu().numpy(),
                   ort_session.get_inputs()[1].name: other.cpu().numpy()})
        sim = similarity(ori_out.detach().cpu().numpy(), quantized_out[0])
        self.assertTrue(sim > 99)
        
        # uniform
        mod = MatMulQAT()
        for i in range(10):
            input = torch.rand(30, 30, 30) * 5 - 2
            other = torch.rand(30, 30, 30) * 3 + 7
            mod(input, other)
        self.onnx_path = os.path.join(self.temp_dir, 'matmul_qat.onnx')
        
        torch.onnx.export(mod, (input, other), self.onnx_path)
        ori_out = torch.matmul(input, other)
        ort_session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
        quantized_out = ort_session.run(
            None, {ort_session.get_inputs()[0].name: input.cpu().numpy(),
                   ort_session.get_inputs()[1].name: other.cpu().numpy()})
        sim = similarity(ori_out.detach().cpu().numpy(), quantized_out[0])
        self.assertTrue(sim > 99)
        
        op_type_dict = defaultdict(int)
        model = onnx.load(self.onnx_path)
        for node in model.graph.node:
            op_type_dict[node.op_type] += 1
        
        self.assertEqual(op_type_dict['QuantizeLinear'], 2)
        self.assertEqual(op_type_dict['DequantizeLinear'], 2)

    def test_matmul_qat_infer_failed_dtype_wrong(self):
        # norm data
        mod = MatMulQAT()
        input = torch.rand(30, 30, 30).to(torch.float16)
        other = torch.rand(30, 30, 30).to(torch.float16)
        with self.assertRaises(ValueError):
            mod(input, other)

    def test_matmul_qat_infer_failed_shape_wrong(self):
        # norm data
        mod = MatMulQAT()
        input = torch.rand(30)
        other = torch.rand(30)
        with self.assertRaises(RuntimeError):
            mod(input, other)

        input = torch.rand(3, 3, 3, 3, 3, 3, 3)
        other = torch.rand(3, 3, 3, 3, 3, 3, 3)

        with self.assertRaises(RuntimeError):
            mod(input, other)
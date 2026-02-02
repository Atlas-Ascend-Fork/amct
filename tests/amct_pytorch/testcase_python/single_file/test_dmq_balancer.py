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
import os,sys
import json

sys.path.append("~/amct/llt/asl/aoetools/amct")

import unittest
import argparse
import torch
import torch.nn.functional as F
import onnx
import onnxruntime
import numpy as np
from unittest.mock import patch


import amct_pytorch.amct_pytorch_inner.amct_pytorch
from amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_tool import create_quant_config
from amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_tool import quantize_preprocess
from amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_tool import quantize_model
from amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_tool import save_model
from .utils import models

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

DEVICE = 'cpu'
# DEVICE = torch.device('cuda')


class TestDMQBalancer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_dmq_balancer')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model = models.Net()
        # cls.args_shape = [(1, 2, 4, 14, 14)]
        cls.args_shape = [(16,1,28,28)]
        cls.config_file = os.path.join(cls.temp_folder, 'dmq_balancer_config.json')
        cls.record_file = os.path.join(cls.temp_folder, 'dmq_balancer_record.txt')
        cls.modfied_onnx_file = os.path.join(cls.temp_folder, 'dmq_balancer_modified.onnx')

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ptq_with_dmq_balancer(self):
        # prepare ori_model
        device = DEVICE
        model = self.model.to(device)
        test_iter = 2
        # run ori_model
        run_inference_model(model, iterations=test_iter)

        # do calibration
        modfied_onnx_file, fake_quant_onnx = do_calibration(model, self.args_shape, self.temp_folder)
        run_inference_onnx(fake_quant_onnx, iterations=test_iter)
        print('='*50, 'reesult', '='*50)
        self.assertTrue(os.path.exists(fake_quant_onnx))

    def test_create_quant_config_with_dmq_balancer(self):
        model = self.model.to(DEVICE)
        input_data = tuple([torch.randn(input_shape).to(DEVICE) for input_shape in self.args_shape])
        batch_num = 2
        cfg_def = os.path.join(CUR_DIR, 'utils/test_dmq_balancer/dmq_balancer.cfg')

        amct_pytorch.amct_pytorch_inner.amct_pytorch.create_quant_config(
            config_file=self.config_file,
            model=model,
            input_data=input_data,
            skip_layers=None,
            batch_num=batch_num,
            activation_offset=True,
            config_defination=cfg_def)
        with open(self.config_file) as f:
            quant_config = json.loads(f.read())
        for key, val in quant_config.items():
            if key == 'avg_pool': continue
            if isinstance(val, dict):
                self.assertIn('dmq_balancer_param', val)

    def test_dmq_balancer_config_not_support_layer(self):
        model = self.model.to(DEVICE)
        input_data = tuple([torch.randn(input_shape).to(DEVICE) for input_shape in self.args_shape])
        cfg_def = os.path.join(CUR_DIR, 'utils/test_dmq_balancer/dmq_balancer_not_support.cfg')

        self.assertRaises(ValueError, create_quant_config,
            self.config_file, model, input_data, config_defination=cfg_def)

    def test_dmq_balancer_config_not_support_value(self):
        model = self.model.to(DEVICE)
        input_data = tuple([torch.randn(input_shape).to(DEVICE) for input_shape in self.args_shape])
        cfg_def = os.path.join(CUR_DIR, 'utils/test_dmq_balancer/dmq_balancer_not_support.cfg')

        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.check.GraphQuerier.get_support_dmq_balancer_types',
            return_value=['avg_pool']):
            self.assertRaises(ValueError, create_quant_config,
                self.config_file, model, input_data, config_defination=cfg_def)

    def test_quantize_preprocess_genral(self):
        model = self.model.to(DEVICE)
        input_data = tuple([torch.randn(input_shape).to(DEVICE) for input_shape in self.args_shape])

        dmq_balancer_model = do_calibration_1(model, input_data)

        replaced_module_num = 0
        for name, module in dmq_balancer_model.named_modules():
            if "replaced_module" in name:
                replaced_module_num += 1

        self.assertTrue(os.path.exists(self.record_file))
        self.assertEqual(9, replaced_module_num)

    def test_quantize_preprocess_no_dmq_balancer_config(self):
        model = self.model.to(DEVICE)
        input_data = tuple([torch.randn(input_shape).to(DEVICE) for input_shape in self.args_shape])

        create_quant_config(
            config_file=self.config_file,
            model=model,
            input_data=input_data,
            skip_layers=None,
            batch_num=1,
            activation_offset=True)
        self.assertRaises(RuntimeError, quantize_preprocess, self.config_file, self.record_file, model, input_data)

    def test_quantize_model_no_balance_factor_record(self):
        model = self.model.to(DEVICE)
        input_data = tuple([torch.randn(input_shape).to(DEVICE) for input_shape in self.args_shape])

        dmq_balancer_model = do_calibration_1(model, input_data)

        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils.parse_record_file.RecordManager.get_tensor_balance_factor',
                   return_value=None):
            self.assertRaises(ValueError, quantize_model,
                self.config_file, self.modfied_onnx_file, self.record_file, model, input_data)

    def test_quantize_model_balance_factor_length_err(self):
        model = self.model.to(DEVICE)
        input_data = tuple([torch.randn(input_shape).to(DEVICE) for input_shape in self.args_shape])

        dmq_balancer_model = do_calibration_1(model, input_data)
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils.parse_record_file.RecordManager.get_tensor_balance_factor',
                   return_value=np.array([1.5], dtype=np.float32)):
            self.assertRaises(ValueError, quantize_model,
                self.config_file, self.modfied_onnx_file, self.record_file, model, input_data)

    def test_quantize_model_balance_factor_value_err(self):
        model = self.model.to(DEVICE)
        input_data = tuple([torch.randn(input_shape).to(DEVICE) for input_shape in self.args_shape])

        dmq_balancer_model = do_calibration_1(model, input_data)
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils.parse_record_file.RecordManager.get_tensor_balance_factor',
                   return_value=np.array([0.0], dtype=np.float32)):
            self.assertRaises(ValueError, quantize_model,
                self.config_file, self.modfied_onnx_file, self.record_file, model, input_data)

    def test_save_model_balance_factor_length_err(self):
        model = self.model.to(DEVICE)
        input_data = tuple([torch.randn(input_shape).to(DEVICE) for input_shape in self.args_shape])
        save_path = os.path.join(self.temp_folder, 'dmq_balancer')

        do_calibration_2(model, input_data, self.modfied_onnx_file)
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils.parse_record_file.RecordManager.get_tensor_balance_factor',
                   return_value=np.array([1.5], dtype=np.float32)):
            self.assertRaises(ValueError, save_model, self.modfied_onnx_file, self.record_file, save_path)

    def test_save_model_balance_factor_value_err(self):
        model = self.model.to(DEVICE)
        input_data = tuple([torch.randn(input_shape).to(DEVICE) for input_shape in self.args_shape])
        save_path = os.path.join(self.temp_folder, 'dmq_balancer')

        do_calibration_2(model, input_data, self.modfied_onnx_file)
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils.parse_record_file.RecordManager.get_tensor_balance_factor',
                   return_value=np.array([0.0], dtype=np.float32)):
            self.assertRaises(ValueError, save_model, self.modfied_onnx_file, self.record_file, save_path)


def do_calibration(model, args_shape, temp_folder):
    input_data = tuple([torch.randn(input_shape).to(DEVICE) for input_shape in args_shape])
    batch_num = 2
    cfg_def = os.path.join(CUR_DIR, 'utils/test_dmq_balancer/dmq_balancer.cfg')
    modfied_onnx_file = os.path.join(temp_folder, 'dmq_balancer_modified.onnx')
    fake_quant_onnx = os.path.join(temp_folder, 'dmq_balancer_fake_quant_model.onnx')

    amct_pytorch.amct_pytorch_inner.amct_pytorch.create_quant_config(
        config_file=TestDMQBalancer.config_file,
        model=model,
        input_data=input_data,
        skip_layers=None,
        batch_num=batch_num,
        activation_offset=True,
        config_defination=cfg_def)

    dmq_balancer_model = amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_preprocess(
        config_file=TestDMQBalancer.config_file,
        model=model,
        input_data=input_data,
        record_file=TestDMQBalancer.record_file)

    # run dmq_balancer_model
    run_inference_model(dmq_balancer_model, iterations=batch_num)

    print('='*50, 'calibration', '='*50)

    new_model = amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_model(
        config_file=TestDMQBalancer.config_file,
        model=model,
        input_data=input_data,
        record_file=TestDMQBalancer.record_file,
        modfied_onnx_file=TestDMQBalancer.modfied_onnx_file)

    # run model
    run_inference_model(new_model, iterations=batch_num)
    print('='*50, 'calibration', '='*50)

    # save
    amct_pytorch.amct_pytorch_inner.amct_pytorch.save_model(
        modfied_onnx_file=TestDMQBalancer.modfied_onnx_file,
        record_file=TestDMQBalancer.record_file,
        save_path=os.path.join(temp_folder, 'dmq_balancer'))

    return TestDMQBalancer.modfied_onnx_file, fake_quant_onnx

def do_calibration_1(model, input_data):
    batch_num = 2
    cfg_def = os.path.join(CUR_DIR, 'utils/test_dmq_balancer/dmq_balancer.cfg')

    amct_pytorch.amct_pytorch_inner.amct_pytorch.create_quant_config(
        config_file=TestDMQBalancer.config_file,
        model=model,
        input_data=input_data,
        skip_layers=None,
        batch_num=batch_num,
        activation_offset=True,
        config_defination=cfg_def)

    dmq_balancer_model = amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_preprocess(
        config_file=TestDMQBalancer.config_file,
        record_file=TestDMQBalancer.record_file,
        model=model,
        input_data=input_data)

    # run dmq_balancer_model
    run_inference_model(dmq_balancer_model, iterations=batch_num)
    return dmq_balancer_model

def do_calibration_2(model, input_data, modfied_onnx_file):
    batch_num = 2
    cfg_def = os.path.join(CUR_DIR, 'utils/test_dmq_balancer/dmq_balancer.cfg')
    # fake_quant_onnx = os.path.join(temp_folder, 'dmq_balancer_fake_quant_model.onnx')

    amct_pytorch.amct_pytorch_inner.amct_pytorch.create_quant_config(
        config_file=TestDMQBalancer.config_file,
        model=model,
        input_data=input_data,
        skip_layers=None,
        batch_num=batch_num,
        activation_offset=True,
        config_defination=cfg_def)

    dmq_balancer_model = amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_preprocess(
        config_file=TestDMQBalancer.config_file,
        record_file=TestDMQBalancer.record_file,
        model=model,
        input_data=input_data)

    # run dmq_balancer_model
    run_inference_model(dmq_balancer_model, iterations=batch_num)

    print('='*50, 'calibration', '='*50)

    new_model = amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_model(
        config_file=TestDMQBalancer.config_file,
        model=model,
        input_data=input_data,
        record_file=TestDMQBalancer.record_file,
        modfied_onnx_file=modfied_onnx_file)

    # run model
    run_inference_model(new_model, iterations=batch_num)
    print('='*50, 'calibration', '='*50)

    # return modfied_onnx_file

def run_inference_model(model, iterations=2):
    batch_size = 16
    torch.manual_seed(1)
    device = torch.device(DEVICE)
    kwargs = {'batch_size': batch_size}

    model.eval()
    test_loss = 0
    correct = 0
    iter_num = 0
    with torch.no_grad():
        for i in range(iterations):
            data = torch.tensor(
                np.random.uniform(0, 10, (16,1,28,28)).astype(np.float32))
                # np.random.uniform(0, 10, (1, 2, 4, 14, 14)).astype(np.float32))
            data = data.to(device)
            model = model.to(device)
            output = model(data)
            iter_num = iter_num + 1
            if iter_num == iterations:
                break

def run_inference_onnx(onnx_file, iterations=2):
    # prepare model
    print('onnx_file', onnx_file)
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    input_names = [input_onnx.name for input_onnx in ort_session.get_inputs()]
    output_names = [output_onnx.name for output_onnx in ort_session.get_outputs()]
    print('inputs:', input_names)
    print('otputs:', output_names)

    def to_numpy(tensor):
       data_numpy = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
       return data_numpy

    # prepare data
    batch_size = 16
    torch.manual_seed(1)
    device = torch.device(DEVICE)
    kwargs = {'batch_size': batch_size}

    test_loss = 0
    correct = 0
    iter_num = 0
    with torch.no_grad():
        for i in range(iterations):
            data = torch.tensor(
                np.random.uniform(0, 10, (16,1,28,28)).astype(np.float32))
                # np.random.uniform(0, 10, (1, 2, 4, 14, 14)).astype(np.float32))
            data = data.to(device)
            # run in onnxtime
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
            ort_outs = ort_session.run(output_names, ort_inputs)
            output = torch.Tensor(ort_outs[0])
            iter_num = iter_num + 1
            if iter_num == iterations:
                break

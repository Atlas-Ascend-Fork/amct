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
import unittest
import argparse
import torch
import torch.nn.functional as F
import numpy as np

import amct_pytorch.graph_based_compression.amct_pytorch
import amct_pytorch.graph_based_compression.amct_pytorch as amct
from amct_pytorch.graph_based_compression.amct_pytorch import accuracy_based_auto_calibration
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import generate_fakequant_module
from amct_pytorch.graph_based_compression.amct_pytorch.common.auto_calibration import AutoCalibrationEvaluatorBase
from . import mnist_main
from .mnist_utils import run_inference_model_auto_cali
from amct_pytorch.graph_based_compression.amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.graph_based_compression.amct_pytorch.utils.vars import QUANTIZABLE_TYPES
from amct_pytorch.graph_based_compression.amct_pytorch.utils.vars import QUANTIZABLE_ONNX_TYPES

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
DATASETS_DIR = os.path.realpath(os.path.join(CUR_DIR, '../../../../../../../../build/bin/llt/toolchain/dmct_datasets'))
CKPT_PATH = os.path.join(DATASETS_DIR, 'pytorch/model')
DATA_PATH = os.path.join(DATASETS_DIR, 'pytorch/data')

MAX_ACC_ERR = 0.5
DEVICE = 'cpu'


class TestAutoCaliFakeQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QUANTIZABLE_TYPES.extend(['ConvTranspose2d','AvgPool2d'])
        QUANTIZABLE_ONNX_TYPES.extend(['AveragePool','ConvTranspose'])
        cls.temp_folder = os.path.join(CUR_DIR, 'test_mnist')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model = mnist_main.Net()
        cls.args_shape = [(16,1,28,28)]
        cls.ckpt = os.path.join(CKPT_PATH, 'mnist_cnn.pt')

    @classmethod
    def tearDownClass(cls):
        QUANTIZABLE_TYPES.remove('ConvTranspose2d')
        QUANTIZABLE_TYPES.remove('AvgPool2d')
        QUANTIZABLE_ONNX_TYPES.remove('AveragePool')
        QUANTIZABLE_ONNX_TYPES.remove('ConvTranspose')
        os.popen('rm -r ' + cls.temp_folder)


    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_gen_fake_quant_module(self):
        # prepare ori_model
        device = DEVICE
        model = self.model.to(device)
        # model.load_state_dict(torch.load(self.ckpt, map_location=device))
        test_iter = 2
        # run ori_model
        run_inference_model_auto_cali(
            model, iterations=test_iter)

        # do calibration
        fakequant_model, fakequant_file = do_calibration(
            model, self.args_shape, self.temp_folder)
        run_inference_model_auto_cali(fakequant_model, iterations=test_iter)
        print('='*50, 'reesult', '='*50)
        # self.assertLess(abs(acc_ptq - acc_ori), MAX_ACC_ERR)
        self.assertTrue(os.path.exists(fakequant_file))


def do_calibration(model, args_shape, temp_folder):
    config_file = os.path.join(temp_folder, 'mnist_config.json')
    record_file = os.path.join(temp_folder, 'mnist_record.txt')
    modfied_onnx_file = os.path.join(temp_folder, 'mnist_modified.onnx')
    fake_quant_onnx = os.path.join(temp_folder, 'mnist_fake_quant_model.onnx')
    deploy_onnx = os.path.join(temp_folder, 'mnist_deploy_model.onnx')
    save_model_path = os.path.join(temp_folder, 'mnist')

    input_data = tuple([torch.randn(input_shape) for input_shape in args_shape])

    batch_num = 1
    amct_pytorch.graph_based_compression.amct_pytorch.create_quant_config(
        config_file=config_file,
        model=model,
        input_data=input_data,
        skip_layers=None,
        batch_num=batch_num,
        activation_offset=True,
        config_defination=None)

    new_model = amct_pytorch.graph_based_compression.amct_pytorch.quantize_model(
        config_file=config_file,
        model=model,
        input_data=input_data,
        record_file=record_file,
        modfied_onnx_file=modfied_onnx_file)

    # run model
    run_inference_model_auto_cali(new_model, iterations=batch_num)

    new_ori_model = ModuleHelper.deep_copy(model)
    fq_model = generate_fakequant_module(
        new_ori_model,
        config_file,
        record_file,
        input_data)

    # save
    amct_pytorch.graph_based_compression.amct_pytorch.save_model(
        modfied_onnx_file=modfied_onnx_file,
        record_file=record_file,
        save_path=save_model_path)

    return fq_model, fake_quant_onnx

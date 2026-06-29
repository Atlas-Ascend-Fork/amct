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
import argparse
import logging
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import amct_pytorch.classic.graph_based.amct_pytorch
import amct_pytorch.classic.graph_based.amct_pytorch as amct
from amct_pytorch.classic.graph_based.amct_pytorch import (
    accuracy_based_auto_calibration,
)
from amct_pytorch.classic.graph_based.amct_pytorch.common.auto_calibration import (
    AutoCalibrationEvaluatorBase,
)
from amct_pytorch.classic.graph_based.amct_pytorch.quantize_tool import (
    generate_fakequant_module,
    save_model,
)
from amct_pytorch.classic.graph_based.amct_pytorch.utils.model_util import (
    ModuleHelper,
)

from .utils import models

logger = logging.getLogger(__name__)

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
DATASETS_DIR = os.path.realpath(os.path.join(CUR_DIR, '../../../../../../../../build/bin/llt/toolchain/dmct_datasets'))
CKPT_PATH = os.path.join(DATASETS_DIR, 'pytorch/model')
DATA_PATH = os.path.join(DATASETS_DIR, 'pytorch/data')

DEVICE = 'cpu'


def run_inference_model_auto_cali(model, iterations=2):
    torch.manual_seed(1)
    device = torch.device(DEVICE)

    model.eval()
    iter_num = 0
    with torch.no_grad():
        for _ in range(iterations):
            data = torch.tensor(
                np.random.uniform(0, 10, (4, 2, 3, 3, 3)).astype(np.float32))
            data = data.to(device)
            model(data)
            iter_num = iter_num + 1
            if iter_num == iterations:
                break


class TestAutoCaliFakeQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'net3d')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model = models.Net3d001()
        cls.args_shape = [(4, 2, 3, 3, 3)]
        cls.ckpt = os.path.join(CKPT_PATH, 'net3d_test.pt')

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)


    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_gen_fake_quant_module(self):
        # prepare ori_model
        device = DEVICE
        model = self.model.to(device)
        test_iter = 2
        # run ori_model
        run_inference_model_auto_cali(
            model, iterations=test_iter)

        # do calibration
        fakequant_model, fakequant_file = do_calibration(
            model, self.args_shape, self.temp_folder)
        run_inference_model_auto_cali(fakequant_model, iterations=test_iter)
        logger.info('%s reesult %s', '=' * 50, '=' * 50)
        self.assertTrue(os.path.exists(fakequant_file))


def do_calibration(model, args_shape, temp_folder):
    config_file = os.path.join(temp_folder, 'net3d_config.json')
    record_file = os.path.join(temp_folder, 'net3d_record.txt')
    modfied_onnx_file = os.path.join(temp_folder, 'net3d_modified.onnx')
    fake_quant_onnx = os.path.join(temp_folder, 'net3d_fake_quant_model.onnx')
    save_model_path = os.path.join(temp_folder, 'net3d')

    input_data = tuple([torch.randn(input_shape) for input_shape in args_shape])

    batch_num = 1
    amct_pytorch.classic.graph_based.amct_pytorch.create_quant_config(
        config_file=config_file,
        model=model,
        input_data=input_data,
        skip_layers=None,
        batch_num=batch_num,
        activation_offset=True,
        config_defination=None)

    new_model = amct_pytorch.classic.graph_based.amct_pytorch.quantize_model(
        config_file=config_file,
        model=model,
        input_data=input_data,
        record_file=record_file,
        modfied_onnx_file=modfied_onnx_file)     

    # # run model
    run_inference_model_auto_cali(new_model, iterations=batch_num)

    fq_model = generate_fakequant_module(
        config_file=config_file,
        model=model,
        input_data=input_data,
        record_file=record_file)

    # save
    amct_pytorch.classic.graph_based.amct_pytorch.save_model(
        modfied_onnx_file=modfied_onnx_file,
        record_file=record_file,
        save_path=save_model_path)

    return fq_model, fake_quant_onnx
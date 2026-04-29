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
import sys
import os
import unittest

import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch

from amct_pytorch.graph_based_compression.amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.graph_based_compression.amct_pytorch.distillation_interface import create_distill_config
from amct_pytorch.graph_based_compression.amct_pytorch.distillation_interface import create_distill_model
from amct_pytorch.graph_based_compression.amct_pytorch.distillation_interface import distill
from amct_pytorch.graph_based_compression.amct_pytorch.distillation_interface import save_distill_model
from amct_pytorch.graph_based_compression.amct_pytorch.distill.distill_sample import DistillSampleBase
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.linear import LinearQAT

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class DistillNet(nn.Module):
    def __init__(self):
        super(DistillNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=3, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class DistillQATNet(nn.Module):
    def __init__(self):
        super(DistillQATNet, self).__init__()
        self.conv = Conv2dQAT(2, 2, kernel_size=3, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class DistillNetMultiInput(nn.Module):
    def __init__(self):
        super(DistillNetMultiInput, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x, y

class DistillNetMultiInputQat(nn.Module):
    def __init__(self):
        super(DistillNetMultiInputQat, self).__init__()
        self.conv = Conv2dQAT(2, 2, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x, y

class Conv2dLinear(nn.Module):
    """ not do prune"""
    def __init__(self):
        super().__init__()
        # fc
        self.layer1 = nn.Conv2d(3, 160, kernel_size=3, bias=True)
        self.layer2 = nn.BatchNorm2d(160)
        self.layer3 = nn.Linear(14, 80, bias=False)
        self.layer4 = nn.BatchNorm2d(160)
        self.layer5 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class ModelMultiTensorInput(DistillSampleBase):
    @staticmethod
    def get_model_input_data(samples):
        return samples

class ModelMultiInvalidInput(DistillSampleBase):
    @staticmethod
    def get_model_input_data(samples):
        return (1,1,1,1)

class ModelMultiInvalidInputDict(DistillSampleBase):
    @staticmethod
    def get_model_input_data(samples):
        return dict()

class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class TestDistillInterface(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_distill_interface')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.data = torch.randn(1,2,4,4)
        cls.train_loader = torch.utils.data.DataLoader(cls.data)
        cls.groups = [['conv']]
        # torch model
        cls.torch_model = DistillNet()
        cls.qat_model = DistillQATNet()
        cls.cfg_file = os.path.join(CUR_DIR, 'cfgs/distill_cfg.json')
        cls.dump_cfg_file = os.path.join(CUR_DIR, 'cfgs/distill_cfg_dump.json')
        cls.amct_log_dir = os.path.join(os.getcwd(), 'amct_log')

    @classmethod
    def tearDownClass(cls):
        os.system('rm -r ' + cls.temp_folder)
        print("[UNITTEST END test_distill_interface.py]")

    def test_create_distill_config_no_cfg(self):
        config_file = os.path.join(self.temp_folder, 'default.json')
        create_distill_config(config_file, self.torch_model, self.data)
        self.assertTrue(os.access(config_file, os.F_OK))

    def test_create_distill_config_cfg_not_exist(self):
        config_file = os.path.join(self.temp_folder, 'not_exist.json')
        config_defination = os.path.join(CUR_DIR, 'not_exist.cfg')
        with self.assertRaises(FileNotFoundError):
            create_distill_config(config_file, self.torch_model, self.data, config_defination)

    def test_create_distill_config_cfg_exist(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        config_defination = os.path.join(CUR_DIR, './cfgs/config.cfg')
        create_distill_config(config_file, self.torch_model, self.data, config_defination)
        self.assertTrue(os.access(config_file, os.F_OK))

    def test_distill_no_dump_success(self):
        distill(self.torch_model, self.qat_model, self.cfg_file, self.train_loader, epochs=1, loss=None, optimizer=None)

    def test_distill_dump_success(self):
        optimizer = torch.optim.AdamW(self.qat_model.parameters(), lr=0.1)
        distill(self.torch_model, self.qat_model, self.dump_cfg_file, self.train_loader, epochs=1, loss=None, optimizer=optimizer)

    def test_distill_invalid_epochs(self):
        optimizer = torch.optim.AdamW(self.qat_model.parameters(), lr=0.1)
        self.assertRaises(ValueError, distill, self.torch_model, self.qat_model, self.dump_cfg_file, self.train_loader, epochs=0, loss=None, optimizer=optimizer)

    def test_distill_invalid_cfg(self):
        optimizer = torch.optim.AdamW(self.qat_model.parameters(), lr=0.1)
        cfg = 'abcd.json'
        self.assertRaises(OSError, distill, self.torch_model, self.qat_model, cfg, self.train_loader, epochs=1, loss=None, optimizer=optimizer)

    def test_distill_user_define_sample(self):
        model = DistillNetMultiInput()
        qat_model = DistillNetMultiInputQat()
        sample = ModelMultiTensorInput()

        data0 = torch.randn(1,2,4,4)
        data1 = torch.randn(1,2,4,4)
        dataset = MultiDataset(data0, data1)
        train_loader = torch.utils.data.DataLoader(dataset)
        distill(model, qat_model, self.cfg_file, train_loader, epochs=1, loss=None, optimizer=None, sample_instance=sample)

    def test_distill_user_define_sample_dump(self):
        model = DistillNetMultiInput()
        qat_model = DistillNetMultiInputQat()
        sample = ModelMultiTensorInput()

        data0 = torch.randn(1,2,4,4)
        data1 = torch.randn(1,2,4,4)
        dataset = MultiDataset(data0, data1)
        train_loader = torch.utils.data.DataLoader(dataset)
        distill(model, qat_model, self.dump_cfg_file, train_loader, epochs=1, loss=None, optimizer=None, sample_instance=sample)

    def test_distill_user_define_invalid_sample(self):
        model = DistillNetMultiInput()
        qat_model = DistillNetMultiInputQat()
        sample = ModelMultiInvalidInput()

        data0 = torch.randn(1,2,4,4)
        data1 = torch.randn(1,2,4,4)
        dataset = MultiDataset(data0, data1)
        train_loader = torch.utils.data.DataLoader(dataset)
        self.assertRaises(RuntimeError, distill, model, qat_model, self.dump_cfg_file, train_loader, epochs=1, loss=None, optimizer=None, sample_instance=sample)

        sample = ModelMultiInvalidInputDict()
        self.assertRaises(RuntimeError, distill, model, qat_model, self.dump_cfg_file, train_loader, epochs=1, loss=None, optimizer=None, sample_instance=sample)


    def test_create_and_save_distill_model_with_default_record(self):
        ori_model = Conv2dLinear().to(torch.device("cpu"))
        input_data = torch.randn(4, 3, 16, 16)

        config_file = os.path.join(CUR_DIR, 'cfgs/Conv2dLinear_cfg.json')
        student_model = create_distill_model(config_file, ori_model, input_data)

        self.assertTrue(isinstance(student_model.layer1, Conv2dQAT))
        self.assertTrue(isinstance(student_model.layer3, LinearQAT))

        train_loader = torch.utils.data.DataLoader(input_data)
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=0.1)
        distill(ori_model, student_model, config_file, train_loader, epochs=1, loss=None, optimizer=None)

        save_path = os.path.join(self.temp_folder, 'save_distill')
        save_distill_model(student_model, save_path, input_data)

        fake_quant_onnx = os.path.join(self.temp_folder, 'save_distill_fake_quant_model.onnx')
        deploy_quant_onnx = os.path.join(self.temp_folder, 'save_distill_deploy_model.onnx')
        record_file = os.path.join(self.amct_log_dir, 'scale_offset_record.txt')
        self.assertTrue(os.path.exists(fake_quant_onnx))
        self.assertTrue(os.path.exists(deploy_quant_onnx))
        self.assertTrue(os.path.exists(record_file))

    def test_create_and_save_distill_model_with_define_record(self):
        ori_model = Conv2dLinear().to(torch.device("cpu"))
        input_data = torch.randn(4, 3, 16, 16)

        config_file = os.path.join(CUR_DIR, 'cfgs/Conv2dLinear_cfg.json')
        student_model = create_distill_model(config_file, ori_model, input_data)

        self.assertTrue(isinstance(student_model.layer1, Conv2dQAT))
        self.assertTrue(isinstance(student_model.layer3, LinearQAT))

        train_loader = torch.utils.data.DataLoader(input_data)
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=0.1)
        distill(ori_model, student_model, config_file, train_loader, epochs=1, loss=None, optimizer=None)

        save_path = os.path.join(self.temp_folder, 'save_distill')
        record_file = os.path.join(self.temp_folder, 'scale_offset_record.txt')
        save_distill_model(student_model, save_path, input_data, record_file = record_file)

        fake_quant_onnx = os.path.join(self.temp_folder, 'save_distill_fake_quant_model.onnx')
        deploy_quant_onnx = os.path.join(self.temp_folder, 'save_distill_deploy_model.onnx')
        self.assertTrue(os.path.exists(fake_quant_onnx))
        self.assertTrue(os.path.exists(deploy_quant_onnx))
        self.assertTrue(os.path.exists(record_file))

    @patch.object(ModuleHelper, 'deep_copy')
    def test_save_distill_model_unsupport_deep_copy_model(self, mock_deep_copy):
        mock_deep_copy.side_effect = RuntimeError()

        out = self.qat_model.forward(self.data)
        save_path = os.path.join(self.temp_folder, 'save_distill')
        save_distill_model(self.qat_model, save_path, self.data)

        fake_quant_onnx = os.path.join(self.temp_folder, 'save_distill_fake_quant_model.onnx')
        deploy_quant_onnx = os.path.join(self.temp_folder, 'save_distill_deploy_model.onnx')
        record_file = os.path.join(self.amct_log_dir, 'scale_offset_record.txt')
        self.assertTrue(os.path.exists(fake_quant_onnx))
        self.assertTrue(os.path.exists(deploy_quant_onnx))
        self.assertTrue(os.path.exists(record_file))
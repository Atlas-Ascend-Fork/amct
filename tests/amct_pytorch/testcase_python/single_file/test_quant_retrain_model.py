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
import json
import numpy as np
from io import BytesIO
import torch
from unittest.mock import patch
from unittest import mock

from .utils import models
from .utils import record_file_utils
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.proto import scale_offset_record_pb2
from google.protobuf import text_format

from amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_tool import create_quant_retrain_config
from amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_tool import create_quant_retrain_model
from amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_tool import restore_quant_retrain_model
from amct_pytorch.amct_pytorch_inner.amct_pytorch.quantize_tool import save_quant_retrain_model
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.retrain_config import RetrainConfig
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils import vars


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestQuantRetrainModel(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_quant_retrain_model')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.pth = os.path.join(cls.temp_folder, 'tmp_pth.pth')
        cls.simple_file = os.path.join(cls.temp_folder, 'net_001.cfg')
        with open(cls.simple_file, 'w') as f:
            f.write('batch_num: 1\n')
            f.write(
'''override_layer_types : {
    layer_type: "Conv2d"
    retrain_weight_quant_config: {
        arq_retrain: {
        channel_wise: true
        }
    }
}\n''')
        cls.config_file = os.path.join(cls.temp_folder, 'model_001.json')
        cls.record_file = os.path.join(cls.temp_folder, 'model_001.txt')
        cls.save_path = os.path.join(cls.temp_folder, 'save_quant')

        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.model_001.eval()
        cls.args_shape = [(10, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        create_quant_retrain_config(
            cls.config_file,
            cls.model_001,
            cls.args)

        tmp_onnx = BytesIO()
        Parser.export_onnx(cls.model_001, cls.args, tmp_onnx)
        cls.graph = Parser.parse_net_to_graph(tmp_onnx)
        cls.graph.add_model(cls.model_001)
        RetrainConfig.init_retrain(cls.config_file, cls.record_file, cls.graph)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        print("=*"*30)
        for name, mod in self.model_001.named_modules():
            print(name, mod)
        pass

    def tearDown(self):
        pass

    @unittest.expectedFailure
    def test_create_quant_retrain_config_un_init(self):
        RetrainConfig.un_init()
        retrain_config = RetrainConfig()

    def test_create_quant_retrain_config_retrain_enable_false(self):
        retrain_config = RetrainConfig()
        self.assertFalse(retrain_config.retrain_enable('inexistence'))

    def test_create_quant_retrain_config_get_layer_config_false(self):
        retrain_config = RetrainConfig()
        config = retrain_config.get_layer_config('inexistence')
        self.assertIsNone(config)

    def test_create_quant_retrain_config_get_layer_config(self):
        RetrainConfig.retrain_config = {
            'layer': {
                'retrain_enable': True,
                'retrain_weight_config': {
                    'channel_wise': True}}}
        retrain_config = RetrainConfig()
        config = retrain_config.get_layer_config('layer')
        self.assertIsNotNone(config)

    def test_create_quant_retrain_config(self):
        tmp_onnx = BytesIO()
        Parser.export_onnx(self.model_001, self.args, tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        graph.add_model(self.model_001)
        RetrainConfig.create_quant_retrain_config(
            self.config_file, graph, self.simple_file)
        with open(self.config_file) as f:
            quant_config = json.loads(f.read())
        for key, val in quant_config.items():
            if isinstance(val, dict):
                self.assertIn('retrain_data_config', val)

    def test_create_quant_retrain_model(self):
        new_model = create_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args)
        for param in new_model.parameters():
            print(param)
        data = self.args[0]
        new_model = new_model.train()
        ans_2 = new_model(data)
        torch.save({'state_dict': new_model.state_dict()}, self.pth)

        self.assertTrue(os.path.exists(self.record_file))

    def test_restore_quant_retrain_model(self):
        new_model = restore_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args,
            self.pth,
            'state_dict')

        data = self.args[0]
        new_model = new_model.eval()
        ans_2 = new_model(data)

        self.assertTrue(os.path.exists(self.record_file))

    def test_save_quant_retrain_model(self):
        new_model = restore_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args,
            self.pth,
            'state_dict')

        data = self.args[0]
        new_model = new_model.eval()
        ans_2 = new_model(data)

        save_quant_retrain_model(
            self.config_file,
            new_model,
            self.record_file,
            self.save_path,
            self.args)

        self.assertTrue(
            os.path.exists(''.join([self.save_path, '_deploy_model.onnx'])))

    def test_create_quant_retrain_config_002(self):
        with open(self.simple_file, 'w') as f:
            f.write('batch_num: 1\n')
            f.write(
'''retrain_weight_quant_config: {
    arq_retrain: {
        channel_wise: false
        dst_type: INT8
    }
}\n''')
        tmp_onnx = BytesIO()
        Parser.export_onnx(self.model_001, self.args, tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        graph.add_model(self.model_001)
        RetrainConfig.create_quant_retrain_config(
            self.config_file, graph, self.simple_file)
        with open(self.config_file) as f:
            quant_config = json.loads(f.read())
        for key, val in quant_config.items():
            if isinstance(val, dict):
                self.assertFalse(val['retrain_weight_config']['channel_wise'])

    def test_create_quant_retrain_model_002(self):
        new_model = create_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args)
        for param in new_model.parameters():
            print(param)
        data = self.args[0]
        new_model = new_model.train()
        ans_2 = new_model(data)
        torch.save({'state_dict': new_model.state_dict()}, self.pth)

        self.assertTrue(os.path.exists(self.record_file))

    def test_restore_quant_retrain_model_002(self):
        new_model = restore_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args,
            self.pth,
            'state_dict')

        data = self.args[0]
        new_model = new_model.eval()
        ans_2 = new_model(data)

        self.assertTrue(os.path.exists(self.record_file))

    def test_save_quant_retrain_model_002(self):
        new_model = restore_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args,
            self.pth,
            'state_dict')

        data = self.args[0]
        new_model = new_model.eval()
        ans_2 = new_model(data)

        save_quant_retrain_model(
            self.config_file,
            new_model,
            self.record_file,
            self.save_path,
            self.args)

        self.assertTrue(
            os.path.exists(''.join([self.save_path, '_deploy_model.onnx'])))


class TestQuantRetrainModelDeconv(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_quant_retrain_deconv_model')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.pth = os.path.join(cls.temp_folder, 'tmp_pth.pth')
        cls.simple_file = os.path.join(cls.temp_folder, 'net_conv_deconv.cfg')
        with open(cls.simple_file, 'w') as f:
            f.write('batch_num: 1\n')
            f.write(
'''override_layer_types : {
    layer_type: "ConvTranspose2d"
    retrain_weight_quant_config: {
        arq_retrain: {
        channel_wise: true
        }
    }
}\n''')
        cls.config_file = os.path.join(cls.temp_folder, 'model_001.json')
        cls.record_file = os.path.join(cls.temp_folder, 'model_001.txt')
        cls.save_path = os.path.join(cls.temp_folder, 'save_quant')

        cls.model_001 = models.NetConvDeconv().to(torch.device("cpu"))
        cls.model_001.eval()
        cls.args_shape = [(10, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        create_quant_retrain_config(
            cls.config_file,
            cls.model_001,
            cls.args)

        tmp_onnx = BytesIO()
        Parser.export_onnx(cls.model_001, cls.args, tmp_onnx)
        cls.graph = Parser.parse_net_to_graph(tmp_onnx)
        cls.graph.add_model(cls.model_001)
        RetrainConfig.init_retrain(cls.config_file, cls.record_file, cls.graph)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        print("=*"*30)
        for name, mod in self.model_001.named_modules():
            print(name, mod)

    def tearDown(self):
        pass

    def test_create_quant_retrain_model(self):
        new_model = create_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args)
        for param in new_model.parameters():
            print(param)
        data = self.args[0]
        new_model = new_model.train()
        ans_2 = new_model(data)
        torch.save({'state_dict': new_model.state_dict()}, self.pth)

        self.assertTrue(os.path.exists(self.record_file))

    def test_restore_quant_retrain_model(self):
        new_model = restore_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args,
            self.pth,
            'state_dict')

        data = self.args[0]
        new_model = new_model.eval()
        ans_2 = new_model(data)

        self.assertTrue(os.path.exists(self.record_file))

    def test_save_quant_retrain_model(self):
        new_model = restore_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args,
            self.pth,
            'state_dict')

        data = self.args[0]
        new_model = new_model.eval()
        ans_2 = new_model(data)

        save_quant_retrain_model(
            self.config_file,
            new_model,
            self.record_file,
            self.save_path,
            self.args)

        self.assertTrue(
            os.path.exists(''.join([self.save_path, '_deploy_model.onnx'])))


class TestQuantRetrainModelConvCircular(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        TORCH_VERSION = '1.4.0'
        cls.temp_folder = os.path.join(CUR_DIR, 'test_quant_retrain_conv_circular_model')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.pth = os.path.join(cls.temp_folder, 'tmp_pth.pth')
        cls.simple_file = os.path.join(cls.temp_folder, 'net_conv_conv_circular.cfg')
        with open(cls.simple_file, 'w') as f:
            f.write('batch_num: 1\n')
            f.write(
'''override_layer_types : {
    layer_type: "Conv2d"
    retrain_weight_quant_config: {
        arq_retrain: {
        channel_wise: true
        }
    }
}\n''')
        cls.config_file = os.path.join(cls.temp_folder, 'model_001.json')
        cls.record_file = os.path.join(cls.temp_folder, 'model_001.txt')
        cls.save_path = os.path.join(cls.temp_folder, 'save_quant')

        cls.model_001 = models.EltwiseConv().to(torch.device("cpu"))
        cls.model_001.eval()
        cls.args_shape = [(1, 16, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        create_quant_retrain_config(
            cls.config_file,
            cls.model_001,
            cls.args)

        tmp_onnx = BytesIO()
        Parser.export_onnx(cls.model_001, cls.args, tmp_onnx)
        cls.graph = Parser.parse_net_to_graph(tmp_onnx)
        cls.graph.add_model(cls.model_001)
        RetrainConfig.init_retrain(cls.config_file, cls.record_file, cls.graph)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        print("=*"*30)
        for name, mod in self.model_001.named_modules():
            print(name, mod)

    def tearDown(self):
        pass

    def test_create_quant_retrain_model(self):
        new_model = create_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args)
        for param in new_model.parameters():
            print(param)
        data = self.args[0]
        new_model = new_model.train()
        ans_2 = new_model(data)
        torch.save({'state_dict': new_model.state_dict()}, self.pth)

        self.assertTrue(os.path.exists(self.record_file))

    def test_create_quant_retrain_model_1_4(self):
        with patch.object(vars, 'find_torch_version', return_value='1.4.0') as mock_method:
            new_model = create_quant_retrain_model(
                self.config_file,
                self.model_001,
                self.record_file,
                self.args)
            for param in new_model.parameters():
                print(param)
            data = self.args[0]
            new_model = new_model.train()
            ans_2 = new_model(data)
            torch.save({'state_dict': new_model.state_dict()}, self.pth)

            self.assertTrue(os.path.exists(self.record_file))

    def test_restore_quant_retrain_model(self):
        new_model = restore_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args,
            self.pth,
            'state_dict')

        data = self.args[0]
        new_model = new_model.eval()
        ans_2 = new_model(data)

        self.assertTrue(os.path.exists(self.record_file))

    def test_save_quant_retrain_model(self):
        new_model = restore_quant_retrain_model(
            self.config_file,
            self.model_001,
            self.record_file,
            self.args,
            self.pth,
            'state_dict')

        data = self.args[0]
        new_model = new_model.eval()
        ans_2 = new_model(data)

        save_quant_retrain_model(
            self.config_file,
            new_model,
            self.record_file,
            self.save_path,
            self.args)

        self.assertTrue(
            os.path.exists(''.join([self.save_path, '_deploy_model.onnx'])))


class TestQuantRetrainQuantFusionModel(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_quant_retrain_quant_fusion_model')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.config_file = os.path.join(cls.temp_folder, 'quant_fusion_model.json')
        cls.record_file = os.path.join(cls.temp_folder, 'quant_fusion_model.txt')
        cls.save_path = os.path.join(cls.temp_folder, 'save_quant_fusion_')

        cls.model = models.QuantFusionNet().to(torch.device("cpu"))
        cls.model.eval()
        cls.args_shape = [(1, 16, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def test_quant_fusion_retrain_success(self):
        create_quant_retrain_config(
            self.config_file,
            self.model,
            self.args)
        new_model = create_quant_retrain_model(
            self.config_file,
            self.model,
            self.record_file,
            self.args)
        data = self.args[0]
        new_model = new_model.train()
        ans_2 = new_model(data)
        new_model = new_model.eval()
        ans_2 = new_model(data)

        save_quant_retrain_model(
            self.config_file,
            new_model,
            self.record_file,
            self.save_path,
            self.args,
            input_names=['layer1'])

        graph = Parser.parse_net_to_graph(''.join([self.save_path, '_deploy_model.onnx']))
        quant_node_num = 0
        for node in graph.nodes:
            if node.type == 'AscendQuant':
                quant_node_num += 1

        self.assertEqual(quant_node_num, 1)

    def test_conv1d_padding_mode_failed(self):
        class Net1d(torch.nn.Module):
            """ args_shape: [(1, 2, 14)]
            """
            def __init__(self):
                super(Net1d,self).__init__()
                self.args_shape = [(1, 2, 14)]
                # conv + bn
                self.layer1 = torch.nn.Sequential(
                    torch.nn.Conv1d(2, 2, kernel_size=1, bias=False, padding_mode='reflect'),
                    torch.nn.BatchNorm1d(2))

            def forward(self, x):
                x = self.layer1(x)

                return x
        model_conv1d = Net1d().to(torch.device("cpu"))
        input_data = torch.randn(1, 2, 14)
        config_file = os.path.join(self.temp_folder, 'conv1d_config.json')
        record_file = os.path.join(self.temp_folder, 'conv1d_record.txt')
        self.assertRaises(ValueError, create_quant_retrain_config, config_file, model_conv1d, input_data)

    def test_conv1d_create_quant_retrain_config_success(self):
        class Net1d(torch.nn.Module):
            """ args_shape: [(1, 2, 14)]
            """
            def __init__(self):
                super(Net1d,self).__init__()
                self.args_shape = [(1, 2, 14)]
                # conv + bn
                self.layer1 = torch.nn.Sequential(
                    torch.nn.Conv1d(2, 2, kernel_size=1, bias=False),
                    torch.nn.BatchNorm1d(2))

            def forward(self, x):
                x = self.layer1(x)
                return x
        model_conv1d = Net1d().to(torch.device("cpu"))
        input_data = torch.randn(1, 2, 14)
        config_file = os.path.join(self.temp_folder, 'conv1d_config.json')
        record_file = os.path.join(self.temp_folder, 'conv1d_record.txt')
        pth_path = os.path.join(self.temp_folder, 'conv1d_retrain.pth')
        save_path = os.path.join(self.temp_folder, 'conv1d_retrain')
        create_quant_retrain_config(config_file, model_conv1d, input_data)
        retrain_model = create_quant_retrain_model(config_file, model_conv1d, record_file, input_data)
        retrain_model = retrain_model.train()
        ans_2 = retrain_model(input_data)
        retrain_model = retrain_model.eval()
        retrain_model(input_data)
        save_quant_retrain_model(
            config_file,
            retrain_model,
            record_file,
            save_path,
            input_data)

        self.assertTrue(
            os.path.exists(''.join([save_path, '_deploy_model.onnx'])))
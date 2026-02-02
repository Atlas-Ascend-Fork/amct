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
import torch

from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.configuration import Configuration
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser

from .utils import models

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestConfigurationForTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_config')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'net_001.onnx')
        Parser.export_onnx(cls.model_001, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)


    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_from_param(self):
        ''' test create config from param for: conv, fc'''
        self.graph.add_model(self.model_001)
        config_defination = os.path.join(CUR_DIR, 'utils/precision_mode_fp16.cfg')
        Configuration.create_quant_config(
            config_file=os.path.join(self.temp_folder, 'test_create_from_param.json'),
            graph=self.graph, config_defination=config_defination)

        with open(os.path.join(self.temp_folder, 'test_create_from_param.json'), 'r') as f:
            quant_config = json.load(f)
        layers = ['avg_pool', 'fc.0', 'fc.2', 'fc.5', 'layer1.0', 'layer2.0', 'layer3.0', 'layer4.0', 'layer5.0', 'layer6.0']
        for item in layers:
            self.assertIn(item, quant_config)
        layers_name = Configuration.get_layers_name(quant_config)
        self.assertEqual(layers_name, layers)

    def test_create_from_param_skip_err_layers(self):
        ''' test create config from param for: conv, fc'''
        self.graph.add_model(self.model_001)
        # Configuration.create_quant_config(
        #     config_file=os.path.join(self.temp_folder, 'test_create_from_param_skip_error.json'),
        #     graph=self.graph,
        #     skip_modules=['layer2.2'])
        self.assertRaises(ValueError, Configuration.create_quant_config,
            os.path.join(self.temp_folder, 'test_create_from_param_skip_error.json'),
            self.graph,
            skip_modules=['layer2.2'])

    def test_create_from_cfg(self):
        ''' test create config from cfg file for: conv, fc'''
        self.graph.add_model(self.model_001)
        Configuration.create_quant_config(
            config_file=os.path.join(self.temp_folder, 'test_create_from_cfg.json'),
            graph=self.graph,
            config_defination=os.path.join(CUR_DIR, 'utils/net_001.cfg'))

        with open(os.path.join(self.temp_folder, 'test_create_from_cfg.json'), 'r') as f:
            quant_config = json.load(f)
        self.assertIsNotNone(quant_config)

    def test_create_from_cfg_and_skip_layer(self):
        ''' test create config from cfg file for: conv, fc'''
        self.graph.add_model(self.model_001)
        Configuration.create_quant_config(
            config_file=os.path.join(self.temp_folder, 'test_create_from_cfg.json'),
            graph=self.graph,
            skip_modules=['layer1.0'],
            config_defination=os.path.join(CUR_DIR, 'utils/net_001.cfg'))
        with open(os.path.join(self.temp_folder, 'test_create_from_cfg.json'), 'r') as f:
            quant_config = json.load(f)
        self.assertIsNotNone(quant_config)

    def test_config_without_init(self):
        ''' test raise error without init'''
        Configuration().uninit()
        self.assertRaises(RuntimeError, Configuration().get_quant_config)
        self.assertRaises(RuntimeError,
                          Configuration().get_layer_config, 'layer1.0')
        self.assertRaises(RuntimeError,
                          Configuration().get_global_config, 'version')
        self.assertRaises(RuntimeError, Configuration().get_fusion_switch)
        self.assertRaises(RuntimeError, Configuration().get_skip_fusion_layers)

    def test_config_with_init(self):
        ''' test run ok with init'''
        config_file = os.path.join(self.temp_folder, 'test_config_with_init.json')
        record_file = os.path.join(self.temp_folder, 'test_config_with_init.txt')
        self.graph.add_model(self.model_001)
        Configuration.create_quant_config(config_file,
                                          self.graph,
                                          activation_offset=True)
        Configuration().init(config_file, record_file, self.graph)
        quant_config = Configuration().get_quant_config()
        self.assertIsNotNone(quant_config)
        layer_name = Configuration().get_layer_config('layer1.0')
        self.assertIsNotNone(layer_name)
        global_params_name = Configuration().get_global_config('version')
        self.assertEqual(global_params_name, 1)
        do_fusion = Configuration().get_fusion_switch()
        self.assertTrue(do_fusion)
        skip_fusion_layers = Configuration().get_skip_fusion_layers()
        self.assertEqual(skip_fusion_layers, [])


class TestSharedWeightConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_shared_weight')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.net = models.TorchConv3dShareWeightModel().to(torch.device("cpu"))
        cls.args_shape = [(8, 3, 3, 128, 128)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'net_quant.onnx')
        Parser.export_onnx(cls.net, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)


    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_shared_weight_quant(self):
        self.graph.add_model(self.net)
        cfg_content ='''
    override_layer_configs : {
    layer_name : "conv1"
    calibration_config : {
        arq_quantize : {
            channel_wise : true
        }
        ifmr_quantize : {
            search_range_start : 0.8
            search_range_end : 1.2
            search_step : 0.02
            max_percentile : 0.999999
            min_percentile : 0.999999
        }
    }
}
'''
        config_file = os.path.join(self.temp_folder, 'test_shared_weight_conv1.json')
        config_defination = os.path.join(self.temp_folder, 'test_shared_weight_conv1.cfg')
        with open(config_defination,'w+') as f:
            f.write(cfg_content)

        Configuration.create_quant_config(config_file, self.graph,
                                config_defination=config_defination)

        with open(os.path.join(self.temp_folder, 'test_shared_weight_conv1.json'), 'r') as f:
            quant_config = json.load(f)
        self.assertIsNotNone(quant_config)

    def test_shared_weight_defination_error(self):
        self.graph.add_model(self.net)
        cfg_content ='''
    override_layer_configs : {
    layer_name : "conv2"
    calibration_config : {
        arq_quantize : {
            channel_wise : true
        }
        ifmr_quantize : {
            search_range_start : 0.8
            search_range_end : 1.2
            search_step : 0.02
            max_percentile : 0.999999
            min_percentile : 0.999999
        }
    }
}
'''
        config_file = os.path.join(self.temp_folder, 'test_shared_weight_conv2.json')
        config_defination = os.path.join(self.temp_folder, 'test_shared_weight_conv2.cfg')
        with open(config_defination,'w+') as f:
            f.write(cfg_content)
        try:
            Configuration.create_quant_config(config_file, self.graph,
                                    config_defination=config_defination)
        except Exception as e:
            assert 'some override_layer not in valid_layers for quant' in str(e)

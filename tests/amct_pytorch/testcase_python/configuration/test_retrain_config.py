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
import json
import os
import numpy as np
import torch

from amct_pytorch.graph_based_compression.amct_pytorch.configuration.retrain_config import RetrainConfig
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.retrain_config import CONFIGURER
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser

from .utils import models

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestRetrainConfigForPrune(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_retrain_config')
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
        cls.graph.model = cls.model_001

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    #@unittest.skip('*')
    def test_complete_cfg(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_complete.cfg")
        RetrainConfig.init(self.graph, config_defination)

        fc_config = {
            'regular_prune_enable': False,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.3,
                'ascend_optimized': True
            }
        }

        override_layer_config = {
            'regular_prune_enable': True,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.4,
                'ascend_optimized': False
            }
        }

        override_layer_types = {
            'regular_prune_enable': True,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.5,
                'ascend_optimized': True
            }
        }

        skip_override_layer_types = {
            'regular_prune_enable': False,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.5,
                'ascend_optimized': True
            }
        }

        self.assertEqual(RetrainConfig.retrain_config['fc.0'], fc_config)
        self.assertEqual(RetrainConfig.retrain_config['fc.2'], fc_config)
        self.assertEqual(RetrainConfig.retrain_config['fc.5'], fc_config)

        self.assertEqual(RetrainConfig.retrain_config['layer1.0'], skip_override_layer_types)
        self.assertEqual(RetrainConfig.retrain_config['layer2.0'], override_layer_config)

        self.assertEqual(RetrainConfig.retrain_config.get('layer3.0'), None)
        self.assertEqual(RetrainConfig.retrain_config.get('layer4.0'), None)

        self.assertEqual(RetrainConfig.retrain_config['layer5.0'], override_layer_types)
        self.assertEqual(RetrainConfig.retrain_config['layer6.0'], override_layer_types)

    #@unittest.skip('*')
    def test_no_bcp_prune_ratio(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_miss_ratio.cfg")
        # RetrainConfig.init_prune(self.graph, config_defination)
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination)

    #@unittest.skip('*')
    def test_no_bcp_algo(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_empty_001.cfg")
        # RetrainConfig.init_prune(self.graph, config_defination)
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination)

    #@unittest.skip('*')
    def test_prune_has_quant(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_has_quant.cfg")
        # RetrainConfig.init_prune(self.graph, config_defination)
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination)

    #@unittest.skip('*')
    def test_quant_has_prune(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_quant_has_prune.cfg")
        RetrainConfig.set_ability(enable_retrain=True, enable_prune=False)
        retrain_config = {}
        # CONFIGURER.create_config_from_proto(retrain_config, self.graph, config_defination)
        self.assertRaises(ValueError, CONFIGURER.create_config_from_proto, retrain_config, self.graph, config_defination)

    #@unittest.skip('*')
    def test_prune_skip_override_repeated(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_skip_override_repeated.cfg")
        RetrainConfig.init(self.graph, config_defination)

    # #@unittest.skip('*')
    def test_prune_unsupport_type(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_unsupport_type.cfg")
        # RetrainConfig.init_prune(self.graph, config_defination)
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination)

    #@unittest.skip('*')
    def test_prune_no_layer(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_no_layer.cfg")
        # RetrainConfig.init_prune(self.graph, config_defination)
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination)

    def test_compressed_complete_cfg(self):
        config_defination = os.path.join(CUR_DIR, "./utils/compressed_cfg/net_001_compressed_complete.cfg")
        RetrainConfig.init(self.graph, config_defination, True, True)

    def test_compressed_empty_cfg(self):
        config_defination = os.path.join(CUR_DIR, "./utils/compressed_cfg/net_001_compressed_empty.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)

    def test_compressed_only_prune_complete_prune_cfg(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_complete.cfg")
        RetrainConfig.init(self.graph, config_defination, True, True)
        fc_config = {
            'regular_prune_enable': False,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.3,
                'ascend_optimized': True
            }
        }

        override_layer_config = {
            'regular_prune_enable': True,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.4,
                'ascend_optimized': False
            }
        }

        override_layer_types = {
            'regular_prune_enable': True,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.5,
                'ascend_optimized': True
            }
        }

        skip_override_layer_types = {
            'regular_prune_enable': False,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.5,
                'ascend_optimized': True
            }
        }

        self.assertEqual(RetrainConfig.retrain_config['fc.0'], fc_config)
        self.assertEqual(RetrainConfig.retrain_config['fc.2'], fc_config)
        self.assertEqual(RetrainConfig.retrain_config['fc.5'], fc_config)

        self.assertEqual(RetrainConfig.retrain_config['layer1.0'], skip_override_layer_types)
        self.assertEqual(RetrainConfig.retrain_config['layer2.0'], override_layer_config)

        self.assertEqual(RetrainConfig.retrain_config.get('layer3.0'), None)
        self.assertEqual(RetrainConfig.retrain_config.get('layer4.0'), None)

        self.assertEqual(RetrainConfig.retrain_config['layer5.0'], override_layer_types)
        self.assertEqual(RetrainConfig.retrain_config['layer6.0'], override_layer_types)

    def test_compressed_only_prune_no_bcp_ration(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_miss_ratio.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)

    def test_compressed_only_prune_no_bcp_algo(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_empty_001.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)

    def test_compressed_only_prune_no_filter_prune_algo(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_empty_002.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)

    def test_compressed_only_prune_skip_override_repeated(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_skip_override_repeated.cfg")
        RetrainConfig.init(self.graph, config_defination, True, True)

    def test_compressed_only_prune_prune_unsupported_type(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_unsupport_type.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)

    def test_compressed_only_prune_prune_no_layer(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_no_layer.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)


        config_defination = os.path.join(CUR_DIR, "./utils/net_001_quant_has_prune.cfg")
        layer1_0_skip_override_types =  {
            'regular_prune_enable': False,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.5,
                'ascend_optimized': True
            }
        }

        layer2_0_override_layer_configs = {
            'regular_prune_enable': True,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.4,
                'ascend_optimized': False
            }
        }

        self.assertEqual(RetrainConfig.retrain_config['layer1.0'], layer1_0_skip_override_types)
        self.assertEqual(RetrainConfig.retrain_config['layer2.0'], layer2_0_override_layer_configs)

        self.assertEqual(RetrainConfig.retrain_config.get('layer3.0'), None)
        self.assertEqual(RetrainConfig.retrain_config.get('layer4.0'), None)

        override_layer_types =  {
            'regular_prune_enable': True,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.5,
                'ascend_optimized': True
            }
        }

        self.assertEqual(RetrainConfig.retrain_config['layer5.0'], override_layer_types)
        self.assertEqual(RetrainConfig.retrain_config['layer6.0'], override_layer_types)

        fc_config = {
            'regular_prune_enable': False,
            'regular_prune_config': {
                'prune_type': 'filter_prune',
                'algo': 'balanced_l2_norm_filter_prune',
                'prune_ratio': 0.3,
                'ascend_optimized': True
            }
        }

        self.assertEqual(RetrainConfig.retrain_config['fc.0'], fc_config)
        self.assertEqual(RetrainConfig.retrain_config['fc.2'], fc_config)
        self.assertEqual(RetrainConfig.retrain_config['fc.5'], fc_config)

    def test_compressed_only_quant_complete_quant_cfg(self):
        config_defination = os.path.join(CUR_DIR, "./utils/compressed_cfg/net_001_compressed_quant_only_complete.cfg")
        RetrainConfig.init(self.graph, config_defination, True, True)

    def test_compressed_only_quant_only_data_weight_cfg(self):
        config_defination = os.path.join(CUR_DIR, "./utils/compressed_cfg/net_001_compressed_quant_only_data_weight.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)

    def test_compressed_only_quant_only_data_weight_no_channelwise_cfg(self):
        config_defination = os.path.join(CUR_DIR, "./utils/compressed_cfg/net_001_compressed_quant_only_data_weight_nochannelwise.cfg")
        RetrainConfig.init(self.graph, config_defination, True, True)

        layer_quant = {
            'retrain_enable': True,
            'retrain_data_config': {
                'algo': 'ulq_quantize',
                'dst_type': 'INT8',
                'clip_max': 6.0,
                'clip_min': -6.0
            },
            'retrain_weight_config': {
                'algo': 'arq_retrain',
                'channel_wise': True,
                'dst_type': 'INT8'
            }
        }

        fc_quant = {
            'retrain_enable': True,
            'retrain_data_config': {
                'algo': 'ulq_quantize',
                'dst_type': 'INT8',
                'clip_max': 6.0,
                'clip_min': -6.0
            },
            'retrain_weight_config': {
                'algo': 'arq_retrain',
                'channel_wise': False,
                'dst_type': 'INT8'
            }
        }

        self.assertEqual(RetrainConfig.retrain_config['layer1.0'], layer_quant)
        self.assertEqual(RetrainConfig.retrain_config['layer2.0'], layer_quant)
        self.assertEqual(RetrainConfig.retrain_config['layer3.0'], layer_quant)
        self.assertEqual(RetrainConfig.retrain_config['layer4.0'], layer_quant)
        self.assertEqual(RetrainConfig.retrain_config['layer5.0'], layer_quant)
        self.assertEqual(RetrainConfig.retrain_config['layer6.0'], layer_quant)

        self.assertEqual(RetrainConfig.retrain_config['fc.0'], fc_quant)
        self.assertEqual(RetrainConfig.retrain_config['fc.2'], fc_quant)
        self.assertEqual(RetrainConfig.retrain_config['fc.5'], fc_quant)

    def test_compressed_only_quant_only_skip_layers(self):
        config_defination = os.path.join(CUR_DIR, "./utils/compressed_cfg/net_001_compressed_quant_only_skip_layers.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)

    def test_compressed_only_quant_only_skip_layer_types(self):
        config_defination = os.path.join(CUR_DIR, "./utils/compressed_cfg/net_001_compressed_quant_only_skip_layer_types.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)

    def test_compressed_only_override_layer_configs(self):
        config_defination = os.path.join(CUR_DIR, "./utils/compressed_cfg/net_001_compressed_quant_only_override_layer_configs.cfg")
        RetrainConfig.init(self.graph, config_defination, True, True)

        self.assertEqual(RetrainConfig.retrain_config['layer1.0'].get('regular_prune_enable'), True)
        self.assertEqual(RetrainConfig.retrain_config['layer1.0'].get('regular_prune_config'),
            dict([('prune_type', 'filter_prune'), ('algo', 'balanced_l2_norm_filter_prune'),
            ('prune_ratio', 0.5), ('ascend_optimized', True)]))

if __name__ == "__main__":
    unittest.main()
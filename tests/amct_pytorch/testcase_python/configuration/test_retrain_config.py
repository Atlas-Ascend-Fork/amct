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
import json
import os
import unittest

import numpy as np
import torch

from amct_pytorch.classic.graph_based.amct_pytorch.configuration.retrain_config import (
    CONFIGURER,
    RetrainConfig,
)
from amct_pytorch.classic.graph_based.amct_pytorch.parser.parser import Parser
from tests.amct_pytorch.testcase_python.configuration.utils import models

REGULAR_PRUNE_ENABLE = 'regular_prune_enable'
REGULAR_PRUNE_CONFIG = 'regular_prune_config'

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

ALGO = 'algo'
PRUNE_TYPE = 'prune_type'

DST_TYPE = 'dst_type'
FILTER_PRUNE = 'filter_prune'

INT8 = 'INT8'
BALANCED_L2_NORM_FILTER_PRUNE = 'balanced_l2_norm_filter_prune'

PRUNE_RATIO = 'prune_ratio'

ASCEND_OPTIMIZED = 'ascend_optimized'


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
            REGULAR_PRUNE_ENABLE: False,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.3,
                ASCEND_OPTIMIZED: True
            }
        }

        override_layer_config = {
            REGULAR_PRUNE_ENABLE: True,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.4,
                ASCEND_OPTIMIZED: False
            }
        }

        override_layer_types = {
            REGULAR_PRUNE_ENABLE: True,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.5,
                ASCEND_OPTIMIZED: True
            }
        }

        skip_override_layer_types = {
            REGULAR_PRUNE_ENABLE: False,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.5,
                ASCEND_OPTIMIZED: True
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
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination)

    #@unittest.skip('*')
    def test_no_bcp_algo(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_empty_001.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination)

    #@unittest.skip('*')
    def test_prune_has_quant(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_has_quant.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination)

    #@unittest.skip('*')
    def test_quant_has_prune(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_quant_has_prune.cfg")
        RetrainConfig.set_ability(enable_retrain=True, enable_prune=False)
        retrain_config = {}
        self.assertRaises(
            ValueError, CONFIGURER.create_config_from_proto,
            retrain_config, self.graph, config_defination)

    #@unittest.skip('*')
    def test_prune_skip_override_repeated(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_skip_override_repeated.cfg")
        RetrainConfig.init(self.graph, config_defination)

    # #@unittest.skip('*')
    def test_prune_unsupport_type(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_unsupport_type.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination)

    #@unittest.skip('*')
    def test_prune_no_layer(self):
        config_defination = os.path.join(CUR_DIR, "./utils/net_001_prune_no_layer.cfg")
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
            REGULAR_PRUNE_ENABLE: False,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.3,
                ASCEND_OPTIMIZED: True
            }
        }

        override_layer_config = {
            REGULAR_PRUNE_ENABLE: True,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.4,
                ASCEND_OPTIMIZED: False
            }
        }

        override_layer_types = {
            REGULAR_PRUNE_ENABLE: True,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.5,
                ASCEND_OPTIMIZED: True
            }
        }

        skip_override_layer_types = {
            REGULAR_PRUNE_ENABLE: False,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.5,
                ASCEND_OPTIMIZED: True
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
        layer1_0_skip_override_types = {
            REGULAR_PRUNE_ENABLE: False,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.5,
                ASCEND_OPTIMIZED: True
            }
        }

        layer2_0_override_layer_configs = {
            REGULAR_PRUNE_ENABLE: True,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.4,
                ASCEND_OPTIMIZED: False
            }
        }

        self.assertEqual(RetrainConfig.retrain_config['layer1.0'], layer1_0_skip_override_types)
        self.assertEqual(RetrainConfig.retrain_config['layer2.0'], layer2_0_override_layer_configs)

        self.assertEqual(RetrainConfig.retrain_config.get('layer3.0'), None)
        self.assertEqual(RetrainConfig.retrain_config.get('layer4.0'), None)

        override_layer_types = {
            REGULAR_PRUNE_ENABLE: True,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.5,
                ASCEND_OPTIMIZED: True
            }
        }

        self.assertEqual(RetrainConfig.retrain_config['layer5.0'], override_layer_types)
        self.assertEqual(RetrainConfig.retrain_config['layer6.0'], override_layer_types)

        fc_config = {
            REGULAR_PRUNE_ENABLE: False,
            REGULAR_PRUNE_CONFIG: {
                PRUNE_TYPE: FILTER_PRUNE,
                ALGO: BALANCED_L2_NORM_FILTER_PRUNE,
                PRUNE_RATIO: 0.3,
                ASCEND_OPTIMIZED: True
            }
        }

        self.assertEqual(RetrainConfig.retrain_config['fc.0'], fc_config)
        self.assertEqual(RetrainConfig.retrain_config['fc.2'], fc_config)
        self.assertEqual(RetrainConfig.retrain_config['fc.5'], fc_config)

    def test_compressed_only_quant_complete_quant_cfg(self):
        config_defination = os.path.join(CUR_DIR, "./utils/compressed_cfg/net_001_compressed_quant_only_complete.cfg")
        RetrainConfig.init(self.graph, config_defination, True, True)

    def test_compressed_only_quant_only_data_weight_cfg(self):
        config_defination = os.path.join(
            CUR_DIR, "./utils/compressed_cfg/net_001_compressed_quant_only_data_weight.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)

    def test_compressed_only_quant_only_data_weight_no_channelwise_cfg(self):
        config_defination = os.path.join(
            CUR_DIR,
            "./utils/compressed_cfg/net_001_compressed_quant_only_data_weight_nochannelwise.cfg")
        RetrainConfig.init(self.graph, config_defination, True, True)

        layer_quant = {
            'retrain_enable': True,
            'retrain_data_config': {
                ALGO: 'ulq_quantize',
                DST_TYPE: INT8,
                'clip_max': 6.0,
                'clip_min': -6.0
            },
            'retrain_weight_config': {
                ALGO: 'arq_retrain',
                'channel_wise': True,
                DST_TYPE: INT8
            }
        }

        fc_quant = {
            'retrain_enable': True,
            'retrain_data_config': {
                ALGO: 'ulq_quantize',
                DST_TYPE: INT8,
                'clip_max': 6.0,
                'clip_min': -6.0
            },
            'retrain_weight_config': {
                ALGO: 'arq_retrain',
                'channel_wise': False,
                DST_TYPE: INT8
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
        config_defination = os.path.join(
            CUR_DIR, "./utils/compressed_cfg/net_001_compressed_quant_only_skip_layers.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)

    def test_compressed_only_quant_only_skip_layer_types(self):
        config_defination = os.path.join(
            CUR_DIR, "./utils/compressed_cfg/net_001_compressed_quant_only_skip_layer_types.cfg")
        self.assertRaises(ValueError, RetrainConfig.init, self.graph, config_defination, True, True)

    def test_compressed_only_override_layer_configs(self):
        config_defination = os.path.join(
            CUR_DIR,
            "./utils/compressed_cfg/net_001_compressed_quant_only_override_layer_configs.cfg")
        RetrainConfig.init(self.graph, config_defination, True, True)

        self.assertTrue(RetrainConfig.retrain_config['layer1.0'].get(REGULAR_PRUNE_ENABLE))
        self.assertEqual(RetrainConfig.retrain_config['layer1.0'].get(REGULAR_PRUNE_CONFIG),
            dict([(PRUNE_TYPE, FILTER_PRUNE), (ALGO, BALANCED_L2_NORM_FILTER_PRUNE),
            (PRUNE_RATIO, 0.5), (ASCEND_OPTIMIZED, True)]))

if __name__ == "__main__":
    unittest.main()


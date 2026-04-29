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

from .utils import models
from .utils import record_file_utils

from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.ifmr.ifmr import IFMR
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.recorder.recorder import Recorder

from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import create_quant_config
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import quantize_model
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import save_model
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import _check_config_consistency

from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import add_dump_operations
from amct_pytorch.graph_based_compression.amct_pytorch.common.utils import struct_helper


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestQuantizeTool(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_quantize_tool')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.model_001.eval()
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_quant_config(self):
        ''' test create ok for: conv, fc'''
        args = torch.randn((1, 2, 28, 28))
        config_file = os.path.join(self.temp_folder, 'model_001.json')
        create_quant_config(
            config_file=config_file,
            model=self.model_001,
            input_data=args,
            skip_layers=None,
            batch_num=2,
            activation_offset=True,
            config_defination=None)

        self.assertTrue(os.path.exists(config_file))

    def test_create_quant_config_ifmr(self):
        ''' test create ok for: conv, fc'''
        mod_conv = torch.nn.Conv2d(2, 4, kernel_size=2)
        record_file = os.path.join(CUR_DIR, 'utils/conv_model.txt')
        record_module = Recorder(record_file)
        model = IFMR(mod_conv, record_module, ["conv"])
        model.to(torch.device("cpu"))
        # model.forward(self.args[0])

        args = torch.randn((1, 2, 28, 28))
        config_file = os.path.join(self.temp_folder, 'ifmr.json')
        # create_quant_config(onfig_file, model, args)

        self.assertRaises(RuntimeError, create_quant_config, config_file, model, args)


    def test_quantize_model(self):
        config_file = os.path.join(CUR_DIR, 'utils/test_quantize_tool/model_001.json')
        modfied_onnx_file = os.path.join(self.temp_folder, 'no_exit/model_modified.onnx')
        record_file = os.path.join(self.temp_folder, 'model_001.txt')

        new_model = quantize_model(config_file, modfied_onnx_file, record_file,
            self.model_001, self.args, None, None, None)

        data = self.args[0]
        for _ in range(2):
            ans_2 = new_model(data)

        self.assertTrue(os.path.exists(modfied_onnx_file))
        self.assertTrue(os.path.exists(record_file))

    def test_quantize_model_ifmr(self):
        ''' test create ok for: conv, fc'''
        mod_conv = torch.nn.Conv2d(2, 4, kernel_size=2)
        record_file = os.path.join(CUR_DIR, 'utils/conv_model.txt')
        record_module = Recorder(record_file)
        model = IFMR(mod_conv, record_module, ["conv"])
        model.to(torch.device("cpu"))
        # model.forward(self.args[0])

        args = torch.randn((1, 2, 28, 28))
        config_file = os.path.join(self.temp_folder, 'ifmr.json')
        modfied_onnx_file = os.path.join(self.temp_folder, 'no_exit/ifmr_modified.onnx')
        record_file = os.path.join(self.temp_folder, 'ifmr.txt')
        # quantize_model(config_file, modfied_onnx_file, record_file, model, args)

        self.assertRaises(RuntimeError, quantize_model, config_file, modfied_onnx_file, record_file, model, args)


    def test_check_config_consistency_true_001(self):
        ''' '''
        retrain_config = {
            "version":1,
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize"
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        single_instance_config = {
            "version":1,
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize"
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        consistency = _check_config_consistency(retrain_config, single_instance_config)

        self.assertTrue(consistency)

    def test_check_config_consistency_true_002(self):
        ''' '''
        retrain_config = {
            "version":1,
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize"
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        single_instance_config = {
            "version":1,
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize",
                    "ifmr_init": True
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        consistency = _check_config_consistency(retrain_config, single_instance_config)

        self.assertTrue(consistency)

    def test_check_config_consistency_false_001(self):
        ''' '''
        retrain_config = {
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize"
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        single_instance_config = {
            "version":1,
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize"
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        consistency = _check_config_consistency(retrain_config, single_instance_config)

        self.assertFalse(consistency)

    def test_check_config_consistency_false_002(self):
        ''' '''
        retrain_config = {
            "version":1,
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize"
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        single_instance_config = {
            "version":1,
            "batch_num":1
        }
        consistency = _check_config_consistency(retrain_config, single_instance_config)

        self.assertFalse(consistency)

    def test_check_config_consistency_false_003(self):
        ''' '''
        retrain_config = {
            "version":1,
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize",
                    "ifmr_init": False
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        single_instance_config = {
            "version":1,
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize",
                    "ifmr_init": True
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        consistency = _check_config_consistency(retrain_config, single_instance_config)

        self.assertFalse(consistency)

    def test_check_config_consistency_false_004(self):
        ''' '''
        retrain_config = {
            "version":1,
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize",
                    "ifmr_init": False
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        single_instance_config = {
            "version":1,
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize"
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        consistency = _check_config_consistency(retrain_config, single_instance_config)

        self.assertFalse(consistency)

    def test_check_config_consistency_false_005(self):
        ''' '''
        retrain_config = {
            "version":1,
            "batch_num":1,
            "conv1":{
                "retrain_enable":True,
                "retrain_data_config":{
                    "algo":"ulq_quantize",
                    "ifmr_init": False
                },
                "retrain_weight_config":{
                    "algo":"arq_retrain",
                    "channel_wise":True
                }
            }
        }
        single_instance_config = {
            "version":1,
            "batch_num":1,
            "conv1": 1,
        }
        consistency = _check_config_consistency(retrain_config, single_instance_config)

        self.assertFalse(consistency)
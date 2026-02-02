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
from io import BytesIO
import stat

import numpy as np
import torch

from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.save import generate_onnx_file_name
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.save import split_dir_prefix
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.save import _write_node_info, delete_customized_attr
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestSave(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_save')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_onnx_name(self):
        name = generate_onnx_file_name('amct', 'model', 'Deploy')
        self.assertEqual(name, 'amct/model_deploy_model.onnx')

        name = generate_onnx_file_name('amct', 'model', 'Fakequant')
        self.assertEqual(name, 'amct/model_fake_quant_model.onnx')

        name = generate_onnx_file_name('amct', '', 'Deploy')
        self.assertEqual(name, 'amct/deploy_model.onnx')

        name = generate_onnx_file_name('amct', '', 'Fakequant')
        self.assertEqual(name, 'amct/fake_quant_model.onnx')

    def test_split_dir_prefix(self):
        save_dir, save_prefix = split_dir_prefix('')
        self.assertEqual(save_dir, os.path.realpath(''))
        self.assertEqual(save_prefix, '')

        save_dir, save_prefix = split_dir_prefix('./')
        self.assertEqual(save_dir, os.path.realpath('./'))
        self.assertEqual(save_prefix, '')

        save_dir, save_prefix = split_dir_prefix('./model')
        self.assertEqual(save_dir, os.path.realpath('./'))
        self.assertEqual(save_prefix, 'model')

    def test_write_node_info(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                    torch.nn.BatchNorm2d(3, 3))
        model.eval()
        tmp_onnx = BytesIO()
        onnx_file = os.path.join(self.temp_folder, "tmp.onnx")
        torch.onnx.export(model, torch.randn(1, 3, 19, 19), onnx_file)
        customized_attr = {
            "Conv_0": [{"attr_name": "op_data_type", "attr_type": "STRING", "attr_val": bytes("float16", encoding='utf-8')},],
            "BatchNormalization_1": [{"attr_name": "op_data_type", "attr_type": "STRING", "attr_val": bytes("float16", encoding='utf-8')},]
        }
        graph = Parser.parse_net_to_graph(onnx_file)
        dump_model = graph.dump_proto()
        dump_model = _write_node_info(dump_model, customized_attr)
        file_realpath = os.path.join(self.temp_folder, 'temp.onnx')
        with open(file_realpath, 'wb') as fid:
            fid.write(dump_model.SerializeToString())
        # set file's permission 640
        os.chmod(file_realpath, stat.S_IRUSR + stat.S_IWUSR + stat.S_IRGRP)
        graph = Parser.parse_net_to_graph(file_realpath)
        for node in graph.nodes:
            find_flag = False
            if node.proto.name in ['Conv_0', 'BatchNormalization_1']:
                for attr in node.proto.attribute:
                    if attr.name == "op_data_type":
                        find_flag = True
                self.assertTrue(find_flag)

    def test_delete_customized_attr(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                    torch.nn.BatchNorm2d(3, 3))
        model.eval()
        tmp_onnx = BytesIO()
        onnx_file = os.path.join(self.temp_folder, "tmp.onnx")
        torch.onnx.export(model, torch.randn(1, 3, 19, 19), onnx_file)
        customized_attr = {
            "Conv_0": [{"attr_name": "op_data_type", "attr_type": "STRING", "attr_val": bytes("float16", encoding='utf-8')},],
            "BatchNormalization_1": [{"attr_name": "op_data_type", "attr_type": "STRING", "attr_val": bytes("float16", encoding='utf-8')},]
        }
        graph = Parser.parse_net_to_graph(onnx_file)
        dump_model = graph.dump_proto()
        dump_model = _write_node_info(dump_model, customized_attr)
        delete_customized_attr(dump_model, ["op_data_type"])
        file_realpath = os.path.join(self.temp_folder, 'temp.onnx')
        with open(file_realpath, 'wb') as fid:
            fid.write(dump_model.SerializeToString())
        # set file's permission 640
        os.chmod(file_realpath, stat.S_IRUSR + stat.S_IWUSR + stat.S_IRGRP)
        graph = Parser.parse_net_to_graph(file_realpath)
        for node in graph.nodes:
            find_flag = False
            if node.proto.name in ['Conv_0', 'BatchNormalization_1']:
                for attr in node.proto.attribute:
                    if attr.name == "op_data_type":
                        find_flag = True
                self.assertFalse(find_flag)

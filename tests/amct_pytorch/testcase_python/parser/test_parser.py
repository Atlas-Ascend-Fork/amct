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

import os
from io import BytesIO
import unittest
import torch
from unittest.mock import patch
import stat

from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils.util import version_higher_than
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import _export_to_onnx
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.save import _write_node_info, delete_customized_attr
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.model_util import ModuleHelper

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_parser')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_parser_success(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.model = torch.nn.BatchNorm2d(2).to(torch.device("cpu"))

            def forward(self, x):
                return self.model(x)

        model = TestModel()
        tmp_onnx = BytesIO()
        torch_out = Parser.export_onnx(model, self.args, tmp_onnx)
        self.assertIsNone(torch_out)

    @patch('torch.onnx.export')
    def test_parse_unsupport_bn(self, mock_export):
        mock_export.side_effect = RuntimeError()
        class TestModel(torch.nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.bn = torch.nn.BatchNorm2d(2, track_running_stats=False).to(torch.device("cpu"))

            def forward(self, x):
                return self.bn(x)
        model = TestModel()
        tmp_onnx = BytesIO()
        self.assertRaises(
            RuntimeError,
            Parser.export_onnx,
            model,
            self.args,
            tmp_onnx)

    @patch.object(ModuleHelper, 'deep_copy')
    def test_parse_export_unsupport_deep_copy_model(self, mock_deep_copy):
        mock_deep_copy.side_effect = RuntimeError()
        class TestModel(torch.nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.bn = torch.nn.BatchNorm2d(2).to(torch.device("cpu"))

            def forward(self, x):
                return self.bn(x)

        model = TestModel()
        tmp_onnx = BytesIO()
        torch_out = Parser.export_onnx(model, self.args, tmp_onnx)
        self.assertIsNone(torch_out)


    def test_export_large(self):
        class ConvBNConvSerial(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
                        padding=0, dilation=1, groups=1, bias=True, \
                        padding_mode='zeros', affine=True, track_running_stats=True):
                super(ConvBNConvSerial, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels, out_channels, \
                                    kernel_size = kernel_size, stride=stride,\
                                    padding=padding, dilation = dilation, \
                                    groups = groups, bias = bias, \
                                    padding_mode = padding_mode)
                self.bn1 = torch.nn.BatchNorm2d(out_channels,affine=affine, track_running_stats=track_running_stats)
                self.conv2 = torch.nn.Conv2d(out_channels, out_channels, \
                                    kernel_size = kernel_size, stride=stride,\
                                    padding=padding, dilation = dilation, \
                                    groups = groups, bias = bias, \
                                    padding_mode = padding_mode)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.conv2(x)
                return x

        model = ConvBNConvSerial(3,10000,3).to('cpu')

        args_shape = [(4,3,16,16)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape).to('cpu'))
        args = tuple(args)

        tmp_onnx = BytesIO()
        # Parser.export_onnx(model, args, tmp_onnx)
        if not version_higher_than(torch.__version__, '1.10.0'):
            self.assertRaises(RuntimeError, Parser.export_onnx, model, args, tmp_onnx)

        if not version_higher_than(torch.__version__, '1.10.0'):
            tmp_onnx = os.path.join(self.temp_folder, 'ConvBNConvSerial.onnx')
            # Parser.export_onnx(model, args, tmp_onnx)
            self.assertRaises(RuntimeError, Parser.export_onnx, model, args, tmp_onnx)

    def test_write_node_attrs_extracted_from_onnx(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                    torch.nn.BatchNorm2d(3, 3))
        model.eval()
        tmp_onnx = BytesIO()
        onnx_file = os.path.join(self.temp_folder, "tmp.onnx")
        torch.onnx.export(model, torch.randn(1, 3, 19, 19), onnx_file)

        graph = Parser.parse_net_to_graph(onnx_file)
        conv_node = None
        for node in graph.nodes:
            if node.type == 'Conv':
                conv_node = node

        customized_attr = {
            conv_node.name: [{"attr_name": "op_data_type", "attr_type": "STRING", "attr_val": bytes("float16", encoding='utf-8')},],
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
        Parser.write_node_attrs_extracted_from_onnx(graph, file_realpath, ['op_data_type'])
        conv_node = None
        for node in graph.nodes:
            if node.type == 'Conv':
                conv_node = node
                break
        self.assertTrue(conv_node.has_attr("op_data_type"))
        self.assertEqual(conv_node.get_attr("op_data_type"), "float16")

    @patch('torch.onnx.export')
    def test_parse_export_return_model(self, mock_torch_onnx_export):
        mock_torch_onnx_export.return_value = 0
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), torch.nn.BatchNorm2d(3, 3))
        model.eval()
        tmp_onnx = BytesIO()
        export_setting = {}
        self.assertRaises(RuntimeError, _export_to_onnx, model, self.args, tmp_onnx, export_setting)

    def test_validate_export_setting_invalid_input_names(self):
        export_setting = {'input_names': [1]}
        self.assertRaises(RuntimeError, Parser.validate_export_setting, export_setting)

    def test_validate_export_setting_invalid_output_names(self):
        export_setting = {'output_names': [1]}
        self.assertRaises(RuntimeError, Parser.validate_export_setting, export_setting)

    def test_validate_export_setting_invalid_dynamic_axes_1(self):
        export_setting = {'dynamic_axes': {0: "inputs"}}
        self.assertRaises(RuntimeError, Parser.validate_export_setting, export_setting)

    def test_validate_export_setting_invalid_dynamic_axes_2(self):
        export_setting = {'dynamic_axes': {"inputs": (0,2,3)}}
        self.assertRaises(RuntimeError, Parser.validate_export_setting, export_setting)

    def test_validate_export_setting_invalid_dynamic_axes_3(self):
        export_setting = {'dynamic_axes': {"inputs": {'0': '32'}}}
        self.assertRaises(RuntimeError, Parser.validate_export_setting, export_setting)

    def test_validate_export_setting_invalid_dynamic_axes_4(self):
        export_setting = {'dynamic_axes': {"inputs": {0: 32}}}
        self.assertRaises(RuntimeError, Parser.validate_export_setting, export_setting)

    def test_validate_export_setting_invalid_dynamic_axes_5(self):
        export_setting = {'dynamic_axes': {"inputs": ['0']}}
        self.assertRaises(RuntimeError, Parser.validate_export_setting, export_setting)

    def test_validate_export_setting_invalid_dynamic_axes_6(self):
        export_setting = {'dynamic_axes': {"inputs": [-4]}}
        self.assertRaises(RuntimeError, Parser.validate_export_setting, export_setting)

    def test_validate_export_setting_invalid_dynamic_axes_7(self):
        export_setting = {'dynamic_axes': {"inputs": {-4: '32'}}}
        self.assertRaises(RuntimeError, Parser.validate_export_setting, export_setting)
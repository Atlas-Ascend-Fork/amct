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
import unittest
import torch

from .utils import models

from amct_pytorch.amct_pytorch_inner.amct_pytorch.prune_interface import create_compressed_retrain_model
from amct_pytorch.amct_pytorch_inner.amct_pytorch.prune_interface import restore_compressed_retrain_model
from amct_pytorch.amct_pytorch_inner.amct_pytorch.prune_interface import save_compressed_retrain_model
from amct_pytorch.amct_pytorch_inner.amct_pytorch.prune_interface import create_prune_retrain_model
from amct_pytorch.amct_pytorch_inner.amct_pytorch.prune_interface import restore_prune_retrain_model
from amct_pytorch.amct_pytorch_inner.amct_pytorch.prune_interface import save_prune_retrain_model


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestPruneInterface(unittest.TestCase):
    """
    The ST for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_prune_interface')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_prune_model_001(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_001.txt')
        pth_file = os.path.join(self.temp_folder, 'pth_file.pth')

        ori_model = models.Net001().to(torch.device("cpu"))
        args_shape = [(2, 2, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)
        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        new_model.eval()
        new_model(args[0])

        torch.save({'state_dict': new_model.state_dict()}, pth_file)

        new_model2 = restore_prune_retrain_model(ori_model, args, record_file, config_defination, pth_file, 'state_dict')
        new_output2 = new_model2.forward(args[0])

        self.assertEqual(ori_output.shape, new_output.shape)
        self.assertEqual((new_output==new_output2).all(), torch.tensor(True))

    def test_prune_model_002(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_002_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_002.txt')

        ori_model = models.EltwiseConv().to(torch.device("cpu"))
        args_shape = [(1, 16, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)
        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        self.assertEqual(new_model.layer1[0].out_channels, 120)


    def test_prune_model_003(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_003.txt')

        ori_model = models.LinearIn2().to(torch.device("cpu"))
        args_shape = [(4, 16)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])
        self.assertEqual(ori_output.shape, new_output.shape)

        self.assertEqual(new_model.layer1.out_features, 512)
        self.assertEqual(new_model.layer3.out_features, 128)

    def test_prune_model_004(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_004.txt')

        ori_model = models.LinearIn3().to(torch.device("cpu"))
        args_shape = [(4, 16, 16)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])
        self.assertEqual(ori_output.shape, new_output.shape)

        self.assertEqual(new_model.layer1.out_features, 1024)
        self.assertEqual(new_model.layer3.out_features, 128)

    def test_prune_model_005(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_005.txt')

        ori_model = models.LinearIn4().to(torch.device("cpu"))
        args_shape = [(4, 3, 16, 16)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])
        print('new_model', new_model)
        self.assertEqual(ori_output.shape, new_output.shape)

        if '1.10' in torch.__version__:
            self.assertEqual(new_model.layer1.out_features, 1024)
        else:
            self.assertEqual(new_model.layer1.out_features, 512)

        self.assertEqual(new_model.layer3.out_features, 64)

    def test_prune_model_006(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_006.txt')

        ori_model = models.Conv2dLinear().to(torch.device("cpu"))
        args_shape = [(4, 3, 16, 16)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])
        print('new_model', new_model)
        self.assertEqual(ori_output.shape, new_output.shape)

        self.assertEqual(new_model.layer1.out_channels, 160)
        self.assertEqual(new_model.layer3.out_features, 80)

    def test_prune_model_007(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_007.txt')

        ori_model = models.LinearConv2d().to(torch.device("cpu"))
        args_shape = [(4, 3, 16, 16)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])
        print('new_model', new_model)
        self.assertEqual(ori_output.shape, new_output.shape)

        self.assertEqual(new_model.layer1.out_features, 80)
        self.assertEqual(new_model.layer3.out_channels, 160)

    def test_prune_model_008(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_008.txt')

        ori_model = models.LinearAddConv2d().to(torch.device("cpu"))
        args_shape = [(4, 32, 64, 64)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        self.assertEqual(ori_output.shape, new_output.shape)

        self.assertEqual(new_model.layer1.out_features, 64)
        self.assertEqual(new_model.layer3.out_channels, 32)

    def test_prune_model_009(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_009.txt')

        ori_model = models.LinearConcatConv2d().to(torch.device("cpu"))
        args_shape = [(4, 32, 64, 64)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        self.assertEqual(ori_output.shape, new_output.shape)

        self.assertEqual(new_model.layer1.out_features, 64)
        self.assertEqual(new_model.layer3.out_channels, 32)

    def test_prune_model_010(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_010.txt')

        ori_model = models.ConcatDim0Conv().to(torch.device("cpu"))
        args_shape = [(1, 16, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        self.assertEqual(ori_output.shape, new_output.shape)

        self.assertEqual(new_model.layer1[0].out_channels, 160)
        self.assertEqual(new_model.layer2[0].out_channels, 160)

    def test_prune_model_011(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_011.txt')

        ori_model = models.GroupConv().to(torch.device("cpu"))
        args_shape = [(1, 16, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        self.assertEqual(ori_output.shape, new_output.shape)

        self.assertEqual(new_model.layer1[0].out_channels, 32)
        self.assertEqual(new_model.layer2[0].out_channels, 32)

    def test_prune_model_012(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_002_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_012.txt')
        pth_file = os.path.join(self.temp_folder, 'pth_file.pth')

        ori_model = models.NetConvDeconv().to(torch.device("cpu"))
        args_shape = [(1, 2, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        new_model.eval()
        new_model(args[0])

        torch.save({'state_dict': new_model.state_dict()}, pth_file)

        new_model2 = restore_prune_retrain_model(ori_model, args, record_file, config_defination, pth_file, 'state_dict')
        new_output2 = new_model2.forward(args[0])

        self.assertEqual(ori_output.shape, new_output2.shape)

        self.assertEqual(new_model2.layer1[0].out_channels, 16)
        self.assertEqual(new_model2.layer2[0].in_channels, 16)

    def test_mix_prune_retrain_model(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_003_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_003.txt')
        pth_file = os.path.join(self.temp_folder, 'pth_file.pth')

        ori_model = models.Net001().to(torch.device("cpu"))
        args_shape = [(2, 2, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)
        ori_output = ori_model.forward(args[0])
        new_model = create_prune_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        torch.save({'state_dict': new_model.state_dict()}, pth_file)
        new_model2 = restore_prune_retrain_model(ori_model, args, record_file, config_defination, pth_file, 'state_dict')

        new_model2.eval()
        new_model2(args[0])
        save_path = os.path.join(self.temp_folder, 'save_model')
        save_prune_retrain_model(new_model2, save_path, args)
        exists_deploy_path = os.path.join(self.temp_folder, 'save_model_deploy_model.onnx')
        exists_fake_quant_path = os.path.join(self.temp_folder, 'save_model_fake_quant_model.onnx')
        self.assertEqual(new_model2.layer1[0].out_channels, 8)
        self.assertTrue(os.path.exists(exists_deploy_path))
        self.assertTrue(os.path.exists(exists_fake_quant_path))

    def test_compressed_model_001(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/net_001_compressed_complete.cfg')
        record_file = os.path.join(self.temp_folder, 'model_001.txt')
        pth_file = os.path.join(self.temp_folder, 'pth_file.pth')
        save_path = os.path.join(self.temp_folder, 'save_compressed')

        ori_model = models.Net001().to(torch.device("cpu"))
        args_shape = [(2, 2, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_compressed_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        self.assertEqual(ori_output.shape, new_output.shape)

        new_model.eval()
        new_model(args[0])

        torch.save({'state_dict': new_model.state_dict()}, pth_file)
        restore_model = restore_compressed_retrain_model(ori_model, args, config_defination, record_file, pth_file, 'state_dict')

        save_compressed_retrain_model(new_model, record_file, save_path, args)

        exists_deploy_path = os.path.join(self.temp_folder, 'save_compressed_deploy_model.onnx')
        exists_fake_quant_path = os.path.join(self.temp_folder, 'save_compressed_fake_quant_model.onnx')
        self.assertTrue(exists_deploy_path)
        self.assertTrue(exists_fake_quant_path)

    def test_compressed_model_002(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_prune.cfg')
        record_file = os.path.join(self.temp_folder, 'model_001.txt')
        pth_file = os.path.join(self.temp_folder, 'pth_file.pth')
        save_path = os.path.join(self.temp_folder, 'save_compressed')

        ori_model = models.Net001().to(torch.device("cpu"))
        args_shape = [(2, 2, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_compressed_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        self.assertEqual(ori_output.shape, new_output.shape)

        new_model.eval()
        new_model(args[0])

        torch.save({'state_dict': new_model.state_dict()}, pth_file)
        restore_model = restore_compressed_retrain_model(ori_model, args, config_defination, record_file, pth_file, 'state_dict')

        save_compressed_retrain_model(new_model, record_file, save_path, args)

        exists_prune_deploy_path = os.path.join(self.temp_folder, 'save_compressed_deploy_model.onnx')
        exists_prune_fake_quant_path = os.path.join(self.temp_folder, 'save_compressed_fake_quant_model.onnx')
        self.assertTrue(exists_prune_deploy_path)
        self.assertTrue(exists_prune_fake_quant_path)

    def test_compressed_model_003(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_001_compressed.cfg')
        record_file = os.path.join(self.temp_folder, 'model_comp_001.txt')
        pth_file = os.path.join(self.temp_folder, 'pth_file.pth')
        save_path = os.path.join(self.temp_folder, 'save_compressed')

        ori_model = models.Net001().to(torch.device("cpu"))
        args_shape = [(2, 2, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_compressed_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        self.assertEqual(ori_output.shape, new_output.shape)

        new_model.eval()
        new_model(args[0])

        torch.save({'state_dict': new_model.state_dict()}, pth_file)
        restore_model = restore_compressed_retrain_model(ori_model, args, config_defination, record_file, pth_file, 'state_dict')

        save_compressed_retrain_model(new_model, record_file, save_path, args)

        exists_prune_deploy_path = os.path.join(self.temp_folder, 'save_compressed_deploy_model.onnx')
        exists_prune_fake_quant_path = os.path.join(self.temp_folder, 'save_compressed_fake_quant_model.onnx')
        self.assertTrue(exists_prune_deploy_path)
        self.assertTrue(exists_prune_fake_quant_path)

    def test_compressed_model_004(self):
        config_defination = os.path.join(CUR_DIR, 'utils/test_prune_interface/model_002_compressed.cfg')
        record_file = os.path.join(self.temp_folder, 'model_comp_002.txt')
        pth_file = os.path.join(self.temp_folder, 'pth_file.pth')
        save_path = os.path.join(self.temp_folder, 'save_compressed')

        ori_model = models.DefaultNet().to(torch.device("cpu"))
        args_shape = [(16, 6, 32, 32)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = ori_model.forward(args[0])

        new_model = create_compressed_retrain_model(ori_model, args, config_defination, record_file)
        new_output = new_model.forward(args[0])

        new_model.eval()
        new_model(args[0])

        torch.save({'state_dict': new_model.state_dict()}, pth_file)
        restore_model = restore_compressed_retrain_model(ori_model, args, config_defination, record_file, pth_file, 'state_dict')

        save_compressed_retrain_model(new_model, record_file, save_path, args)

        exists_prune_deploy_path = os.path.join(self.temp_folder, 'save_compressed_deploy_model.onnx')
        exists_prune_fake_quant_path = os.path.join(self.temp_folder, 'save_compressed_fake_quant_model.onnx')
        self.assertTrue(exists_prune_deploy_path)
        self.assertTrue(exists_prune_fake_quant_path)
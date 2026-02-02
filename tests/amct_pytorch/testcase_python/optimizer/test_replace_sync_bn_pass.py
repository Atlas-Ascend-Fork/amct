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
import torch.nn as nn

from .utils import models

from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.recorder.recorder import Recorder
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.model_optimizer import ModelOptimizer
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.retrain_config import RetrainConfig

from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.replace_sync_bn_pass import \
    RepalceSyncBNPass

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestReplaceSyncBNPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_replace_sync_bn_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_003 = models.Net003().to(torch.device("cpu"))

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_replace_sync_bn_pass(self):
        optimizer = ModelOptimizer()
        optimizer.add_pass(RepalceSyncBNPass())
        optimizer.do_optimizer(self.model_003, None)

        named_module_dict = {name: mod for name, mod in self.model_003.named_modules()}
        print('named_module_dict', named_module_dict)

        self.assertEqual(isinstance(named_module_dict['bn'], torch.nn.modules.batchnorm.BatchNorm2d), True)


if __name__ == '__main__':
    unittest.main()

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
from unittest import mock
import json
import numpy as np
import torch
import math

from .utils import models
from .utils import record_utils

DEVICE = 'cuda:0'
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.prune.filter_prune_helper import create_filter_prune_helper
from amct_pytorch.graph_based_compression.amct_pytorch.proto import scale_offset_record_pytorch_pb2
from amct_pytorch.graph_based_compression.amct_pytorch.common.prune.prune_recorder_helper import PruneRecordHelper
from amct_pytorch.graph_based_compression.amct_pytorch.configuration import retrain_config

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestPruneIndexHelper(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_filter_prune_helper')
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



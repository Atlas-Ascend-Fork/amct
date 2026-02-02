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
import importlib.util
import sys
import pkg_resources
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.vars import PLATFORM
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.vars import OP_PY


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


def __bootstrap():
    name = 'amct_pytorch_op_dump'
    lib_name = '../../../../../{}.cpython-{}-{}-linux-gnu.so'.format(name, OP_PY, PLATFORM)
    files = pkg_resources.resource_filename(__name__, lib_name)
    spec = importlib.util.spec_from_file_location(__name__, files)
    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)

__bootstrap()


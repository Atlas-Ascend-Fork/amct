#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

load libquant.so and libamct_pytorch_ops.so

"""
import os
import importlib.util
import sys
import pkg_resources
from amct_pytorch.utils.vars import PLATFORM
from amct_pytorch.utils.vars import OP_PY


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


def __bootstrap():
    name = 'amct_pytorch_op_dump'
    lib_name = '../../../{}.cpython-{}-{}-linux-gnu.so'.format(name, OP_PY, PLATFORM)
    files = pkg_resources.resource_filename(__name__, \
        lib_name)
    spec = importlib.util.spec_from_file_location(__name__, files)
    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)

__bootstrap()


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
from collections import OrderedDict

import numpy as np
import torch

from ...amct_pytorch.common.utils.util import version_higher_than
from ...amct_pytorch.common.utils.parse_record_file import RecordFileParserBase
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.vars import ALLOWED_ROUND_MODE_MAP
from ...amct_pytorch.utils.vars import MXFP4_E2M1, MXFP8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2, INT4, INT8


def get_layer_quant_params(records, layer_name):
    """
    Function: get single layer quant params from quant_result_path in records
    Inputs:
    records: record
    layer_name: str, name of check layer
    return: dict, include current layer params
    """
    if records.get('quant_result_path') is None:
        raise RuntimeError("quant_result_path not exists in record!".format(layer_name))

    quant_result_path = records.get('quant_result_path')
    if not os.path.exists(quant_result_path):
        raise RuntimeError("quant_result_path {} not exists. Please check your record file.".format(quant_result_path))
    if version_higher_than(torch.__version__, '2.1.0'):
        load_kwargs = {'mmap': True, 'weights_only': False}
    else:
        load_kwargs = {}
    try:
        quant_params = torch.load(quant_result_path, **load_kwargs)
    except Exception as e:
        raise RuntimeError("obtain quant_params params from file failed!") from e
    if quant_params.get(layer_name) is None:
        raise RuntimeError("obtain quant_params param for layer {} failed!".format(layer_name))
    return quant_params.get(layer_name)

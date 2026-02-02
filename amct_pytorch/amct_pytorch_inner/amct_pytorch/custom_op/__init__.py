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
__all__ = [
    'bcp',
    'dump_forward',
    'selective_mask_gen',
    'dmq_balancer_forward',
    'float_to_hifp8',
    'hifp8_to_float',
    'float_to_fp8e4m3fn',
    'fp8e4m3fn_to_float',
    'float_to_fp4e2m1',
    'float_to_fp4e1m2',
    'fp4e2m1_to_float',
    'fp4e1m2_to_float',
    'ifmr_forward_pytorch',
    'ifmr_backward_pytorch',
    'arq_retrain_forward_pytorch',
    'arq_retrain_backward_pytorch',
    'ulq_retrain_forward_pytorch',
    'ulq_retrain_backward_pytorch',
    'ulq_scale_retrain_forward_pytorch',
    'ulq_scale_retrain_backward_pytorch',
    'arq_cali_pytorch',
    'arq_real_pytorch',
    'hfmg_arq_pytorch',
    'hfmg_merge_pytorch',
    'hfmg_forward_pytorch',
    'hfmg_backward_pytorch']

import os
import ctypes
import pkg_resources
import torch

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


def __load_quant_lib():
    lib_name = './libquant_lib.so'
    lib_name = os.path.join(CUR_DIR, lib_name)
    ctypes.cdll.LoadLibrary(lib_name)


__load_quant_lib()

from ...amct_pytorch.custom_op.ifmr.ifmr_impl import ifmr_forward_pytorch
from ...amct_pytorch.custom_op.ifmr.ifmr_impl import ifmr_backward_pytorch
from ...amct_pytorch.custom_op.arq_retrain.arq_retrain_impl import arq_retrain_forward_pytorch
from ...amct_pytorch.custom_op.arq_retrain.arq_retrain_impl import arq_retrain_backward_pytorch
from ...amct_pytorch.custom_op.ulq_retrain.ulq_retrain_impl import ulq_retrain_forward_pytorch
from ...amct_pytorch.custom_op.ulq_retrain.ulq_retrain_impl import ulq_retrain_backward_pytorch
from ...amct_pytorch.custom_op.ulq_scale_retrain.ulq_scale_retrain_impl import ulq_scale_retrain_forward_pytorch
from ...amct_pytorch.custom_op.ulq_scale_retrain.ulq_scale_retrain_impl import ulq_scale_retrain_backward_pytorch
from ...amct_pytorch.custom_op.arq.arq_impl import arq_cali_pytorch
from ...amct_pytorch.custom_op.arq.arq_impl import arq_real_pytorch
from ...amct_pytorch.custom_op.hfmg.hfmg_impl import hfmg_arq_pytorch
from ...amct_pytorch.custom_op.hfmg.hfmg_impl import hfmg_merge_pytorch
from ...amct_pytorch.custom_op.hfmg.hfmg_impl import hfmg_forward_pytorch
from ...amct_pytorch.custom_op.hfmg.hfmg_impl import hfmg_backward_pytorch

from .bcp import bcp
from .selective_mask_gen import selective_mask_gen
from .dmq_balancer.dmq_balancer_func import dmq_balancer_forward
from .dump.amct_pytorch_op_dump import dump_forward
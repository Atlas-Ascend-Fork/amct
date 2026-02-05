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
import numpy as np
import torch
from torch import tensor
import torch.nn as nn
from torch.nn.parameter import Parameter

from .....amct_pytorch.common.utils.check_params import check_params
from .....amct_pytorch.nn.module.quantization.qat_base import QATBase, ULQRetrainParams
from .....amct_pytorch.custom_op.utils import get_distribute_config
from .....amct_pytorch.custom_op.utils import copy_tensor
from .....amct_pytorch.common.utils.vars_util import INT8
from .....amct_pytorch.utils.vars import CLIP_MAX, CLIP_MIN, FIXED_MIN, DST_TYPE, BATCH_NUM, NUM_BITS_MAP
from .....amct_pytorch.custom_op.ifmr.ifmr import IFMR

TYPE = 'type'
SCOPE = 'scope'
DEFAULT_VALUE = 'default'


class MatMulQAT(nn.Module):
    @check_params(device=(str, type(None)),
                  config=(dict, type(None)))
    def __init__(self, device=None, config=None):
        super().__init__()
        self.device = device
        if config is None:
            config = dict()
        # 1.check quant configuration
        self.retrain_enable = config.get('retrain_enable', True)

        self.retrain_data_config = config.get('retrain_data_config', dict())
        self._check_qat_config()
        self.retrain_data_config['asymmetric'] = False

        act_num_bits = NUM_BITS_MAP.get(
            self.retrain_data_config.get('dst_type', INT8))
        self.distribute_config = get_distribute_config()
        # 2.register parameters required for training
        self._register_qat_params()
        
        # 3.init ifmr using in do calibration
        self.input_init_module = IFMR(layers_name='input_ifmr', num_bits=act_num_bits, 
            batch_num=self.retrain_data_config.get(BATCH_NUM, 1))
        self.other_init_module = IFMR(layers_name='other_ifmr', num_bits=act_num_bits, 
            batch_num=self.retrain_data_config.get(BATCH_NUM, 1))

        if self.retrain_data_config.get(CLIP_MAX) is None:
            self.do_init = True
        else:
            self.do_init = False

    @staticmethod
    def check_input(input_data):
        """
        Check the validity of the input data
        
        Parameters:
        input_data: input data for model to be checked
        """
        if input_data.dtype is not torch.float32:
            raise ValueError('Only support dtype torch.float32, but your input dtype is {}'.format(input_data.dtype))
        
        if len(input_data.shape) < 2 or len(input_data.shape) > 6:
            raise RuntimeError('MatMul dual input quantize only support input dim 2-6,'
                ' but your input dim is {}'.format(len(input_data.shape)))

    def acts_quant(self, data, quant_params):
        """
        Quantize the input data.

        Parameters:
        data: The input data to be quantized.
        quant_params: Quantization parameters

        Returns:
        outputs: The output data after quantization.
        """
        ulq_retrain_params = ULQRetrainParams(quant_params.get('scale'), quant_params.get('offset_deploy'),
                                              quant_params.get('clip_max'), quant_params.get('clip_min'),
                                              quant_params.get('clip_max_pre'), quant_params.get('clip_min_pre'))
        outputs, ulq_retrain_params_update = QATBase.do_ulq_retrain(
            data, ulq_retrain_params, self.retrain_data_config, self.distribute_config, self.training)

        with torch.no_grad():
            copy_tensor(quant_params.get('scale'), ulq_retrain_params_update.scale)
            copy_tensor(quant_params.get('offset_deploy'), ulq_retrain_params_update.offset.to(torch.int8))
            copy_tensor(quant_params.get('clip_max'), ulq_retrain_params_update.clip_max)
            copy_tensor(quant_params.get('clip_min'), ulq_retrain_params_update.clip_min)
            copy_tensor(quant_params.get('clip_max_pre'), ulq_retrain_params_update.clip_max)
            copy_tensor(quant_params.get('clip_min_pre'), ulq_retrain_params_update.clip_min)
        return outputs
    
    def acts_quant_init(self, data, quant_params, ifmr_module):
        """
        Initialize the quantization parameters

        Parameters:
        data: Input data
        quant_params: Quantization parameters
        ifmr_module: Module used for initialization

        Returns:
        is_init: Whether the initialization is finished
        data: Input data
        """
        is_init, ulq_retrain_params = QATBase.do_ifmr(
            data, ifmr_module, self.distribute_config)
        with torch.no_grad():
            copy_tensor(quant_params.get('scale'), ulq_retrain_params.scale)
            copy_tensor(quant_params.get('offset_deploy'), ulq_retrain_params.offset.to(torch.int8))
            copy_tensor(quant_params.get('clip_max'), ulq_retrain_params.clip_max)
            copy_tensor(quant_params.get('clip_min'), ulq_retrain_params.clip_min)
            copy_tensor(quant_params.get('clip_max_pre'), ulq_retrain_params.clip_max)
            copy_tensor(quant_params.get('clip_min_pre'), ulq_retrain_params.clip_min)

        return is_init

    def forward(self, input, other):
        """
        Forward propagation function. Depending on whether retraining and initialization are needed, 
        the input data is processed with quantization and then multiplied matrix-wise.

        Parameters:
        input: Input data
        other: Another input data

        Returns:
        out: Result of matrix multiplication
        """
        self.check_input(input)
        self.check_input(other)

        if self.retrain_enable:
            if self.do_init:
                input_is_init = self.acts_quant_init(
                    input, self.acts_quant_params.get('input'), self.input_init_module)
                other_is_init = self.acts_quant_init(
                    other, self.acts_quant_params.get('other'), self.other_init_module)
                self.do_init = not input_is_init and not other_is_init
            else:
                input = self.acts_quant(input, self.acts_quant_params.get('input'))
                other = self.acts_quant(other, self.acts_quant_params.get('other'))

        out = torch.matmul(input, other)
        return out

    def _check_qat_config(self):
        """
        Check if the QAT configuration meets the requirements.
        """
        if not self.retrain_data_config.get(DST_TYPE, INT8) in [INT8]:
            raise ValueError("dst_type for activation should be in range ['INT8'], "
                             "but your input is {}".format(self.retrain_data_config.get(DST_TYPE, INT8)))
        
        batch_num = self.retrain_data_config.get(BATCH_NUM, 1)
        if not isinstance(batch_num, int) or batch_num <= 0:
            raise ValueError("batch_num should be a int bigger than 0, but your input is {}".format(batch_num))
        
        if not isinstance(self.retrain_data_config.get(FIXED_MIN, False), bool):
            raise ValueError("fixed_min should be a bool value, but your input is {}".format(
                type(self.retrain_data_config.get(FIXED_MIN))))

        clip_min = self.retrain_data_config.get(CLIP_MIN, -1.0)
        clip_max = self.retrain_data_config.get(CLIP_MAX, 1.0)

        if not isinstance(clip_min, float) or clip_min >= 0.0:
            raise ValueError('clip_min should be a float smaller than zero, but your input is {}.'.format(clip_min))
        if not isinstance(clip_max, float) or clip_max <= 0.0:
            raise ValueError('clip_max should a float be bigger than zero, but your input is {}.'.format(clip_max))

    def _register_qat_params(self):
        """
        Register the parameters for Quantization-Aware Training
        """
        # register buf and params for input
        self.register_parameter('input_clip_max', Parameter(tensor(self.retrain_data_config.get(CLIP_MAX, 1.0),
            device=self.device), requires_grad=True))
        self.register_parameter('input_clip_min', Parameter(tensor(self.retrain_data_config.get(CLIP_MIN, -1.0),
            device=self.device), requires_grad=True))
        self.register_buffer('input_clip_max_pre', tensor(np.nan, device=self.device))
        self.register_buffer('input_clip_min_pre', tensor(np.nan, device=self.device))
        self.register_buffer('input_scale', tensor([np.nan], device=self.device))
        self.register_buffer('input_offset_deploy', tensor([0], dtype=torch.int8, device=self.device))

        # register buf and params for other
        self.register_parameter('other_clip_max', Parameter(tensor(self.retrain_data_config.get(CLIP_MAX, 1.0),
            device=self.device), requires_grad=True))
        self.register_parameter('other_clip_min', Parameter(tensor(self.retrain_data_config.get(CLIP_MIN, -1.0),
            device=self.device), requires_grad=True))
        self.register_buffer('other_clip_max_pre', tensor(np.nan, device=self.device))
        self.register_buffer('other_clip_min_pre', tensor(np.nan, device=self.device))
        self.register_buffer('other_scale', tensor([np.nan], device=self.device))
        self.register_buffer('other_offset_deploy', tensor([0], dtype=torch.int8, device=self.device))

        self.acts_quant_params = {
            'input': {
                'clip_min': self.input_clip_min,
                'clip_max': self.input_clip_max,
                'clip_min_pre': self.input_clip_min_pre,
                'clip_max_pre': self.input_clip_max_pre,
                'scale': self.input_scale,
                'offset_deploy': self.input_offset_deploy
            },
            'other': {
                'clip_min': self.other_clip_min,
                'clip_max': self.other_clip_max,
                'clip_min_pre': self.other_clip_min_pre,
                'clip_max_pre': self.other_clip_max_pre,
                'scale': self.other_scale,
                'offset_deploy': self.other_offset_deploy
            }
        }
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
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from ....amct_pytorch.common.utils.vars_util import RNN_TENSOR_NUM
from ....amct_pytorch.custom_op.arq_retrain.arq_retrain import ArqRetrainFunction
from ....amct_pytorch.custom_op.ulq_retrain.ulq_retrain import UlqRetrainFunction
from ....amct_pytorch.custom_op.ulq_scale_retrain.ulq_scale_retrain import UlqScaleRetrainFunction
from ....amct_pytorch.custom_op.utils import copy_tensor
from ....amct_pytorch.custom_op.utils import tensor
from ....amct_pytorch.utils.vars import NUM_BITS, FIXED_MIN, H_FIXED_MIN, FLT_EPSILON
from ....amct_pytorch.utils.log import LOGGER


COMP_ALG_QUANT = 'quant'
S_REC_FLAG = 's_rec_flag'


class CompModuleRNN(nn.Module):
    """
    Function: module of quantized retrain for RNN.
    APIs: __init__, forward
    """
    def __init__(self, module,
                 act_config=None,
                 wts_config=None,
                 common_config=None,
                 acts_comp_reuse=None):
        super().__init__()
        self.replaced_module = module
        self.replaced_module_type = module._get_name()
        self.act_config = act_config
        self.wts_config = wts_config
        self.common_config = common_config
        self.acts_comp_reuse = acts_comp_reuse
        self.comp_algs = []
        self.comp_module = deepcopy(module)
        if self.wts_config.get('channel_wise'):
            self.num_scales = self.replaced_module.weight_ih_l0.size(0)
        else:
            self.num_scales = RNN_TENSOR_NUM.get(self.replaced_module_type)
        self._init_output()

    def forward(self, inputs, hx=None):
        """
        Define the computation performed at every call.
        """
        if inputs.abs().max() <= FLT_EPSILON:
            LOGGER.logw('The input tensor is all zeros')
        compressed_inputs, compressed_weights = self._comp_act_wts(inputs, hx)

        with torch.enable_grad():
            self.comp_module.weight_ih_l0.data = compressed_weights[0].data
            self.comp_module.weight_hh_l0.data = compressed_weights[1].data
            if self.replaced_module_type == 'LSTM':
                output = self.comp_module(compressed_inputs[0], (compressed_inputs[1], hx[1]))
            else:
                output = self.comp_module(compressed_inputs[0], compressed_inputs[1])

        return output

    def _comp_act_wts(self, inputs, hx):
        """
        compress activations and weights
        """
        process_group = torch.distributed.group.WORLD
        try:
            world_size = torch.distributed.get_world_size(process_group)
        except (AttributeError, AssertionError, RuntimeError, ValueError):
            process_group = None
            world_size = 1

        need_sync = world_size > 1
        self.common_config['need_sync'] = need_sync
        self.common_config['process_group'] = process_group
        self.common_config['world_size'] = world_size

        # Compress activations.
        acts_comp_func = self.acts_comp_reuse._acts_comp_quant if self.acts_comp_reuse else self._acts_comp_quant
        compressed_inputs = acts_comp_func(inputs, hx)

        # Compress weights.
        compressed_weights = self._wts_quant(self.replaced_module.weight_ih_l0, self.replaced_module.weight_hh_l0)

        if self.cur_batch < 100:
            self.cur_batch += 1
        return compressed_inputs, compressed_weights

    def _init_output(self):
        device = self.common_config.get('device')
        # Register quantitative parameters.
        self.register_buffer('cur_batch', tensor(0, device=device))
        if self.act_config:
            if self.act_config.get('ifmr_init'):
                self.register_parameter('acts_clip_max', Parameter(tensor(1.0, requires_grad=True, device=device)))
                self.register_parameter('acts_clip_min', Parameter(tensor(-1.0, requires_grad=True, device=device)))
                self.register_parameter('acts_h_clip_max', Parameter(tensor(1.0, requires_grad=True, device=device)))
                self.register_parameter('acts_h_clip_min', Parameter(tensor(-1.0, requires_grad=True, device=device)))
            else:
                self.register_parameter('acts_clip_max',
                    Parameter(tensor(self.act_config.get('clip_max'), requires_grad=True, device=device)))
                self.register_parameter('acts_clip_min',
                    Parameter(tensor(self.act_config.get('clip_min'), requires_grad=True, device=device)))
                self.register_parameter('acts_h_clip_max',
                    Parameter(tensor(self.act_config.get('clip_max'), requires_grad=True, device=device)))
                self.register_parameter('acts_h_clip_min',
                    Parameter(tensor(self.act_config.get('clip_min'), requires_grad=True, device=device)))
            self.register_buffer('acts_clip_max_pre', tensor(np.nan, device=device))
            self.register_buffer('acts_clip_min_pre', tensor(np.nan, device=device))
            self.register_buffer('acts_h_clip_max_pre', tensor(np.nan, device=device))
            self.register_buffer('acts_h_clip_min_pre', tensor(np.nan, device=device))
            self.register_buffer('acts_scale', tensor(np.nan, device=device))
            self.register_buffer('acts_offset', tensor(np.nan, device=device))
            self.register_buffer('acts_h_scale', tensor(np.nan, device=device))
            self.register_buffer('acts_h_offset', tensor(np.nan, device=device))

        self.register_parameter('wts_scales',
                                Parameter(tensor([np.nan] * self.num_scales, requires_grad=True, device=device)))
        self.register_parameter('wts_offsets',
                                Parameter(tensor([np.nan] * self.num_scales, requires_grad=True, device=device)))
        self.register_parameter('rec_wts_scales',
                                Parameter(tensor([np.nan] * self.num_scales, requires_grad=True, device=device)))
        self.register_parameter('rec_wts_offsets',
                                Parameter(tensor([np.nan] * self.num_scales, requires_grad=True, device=device)))
        self.register_buffer(S_REC_FLAG,
            tensor(False if self.wts_config.get(S_REC_FLAG) is None else self.wts_config.get(S_REC_FLAG),
            device=device))

    def _acts_comp_quant(self, inputs, hx):
        """
        Function: activation quantization function.
        Inputs:
            inputs: data used for calibration in torch.tensor.
        """
        act_qat_param = {
            NUM_BITS: self.act_config.get(NUM_BITS),
            FIXED_MIN: self.act_config.get(FIXED_MIN),
        }
        # forward with fake-quantized activations
        quant_inputs, scale, offset, clip_max, clip_min = \
            UlqRetrainFunction.apply(
                inputs, self.acts_clip_max, self.acts_clip_min,
                self.acts_clip_max_pre, self.acts_clip_min_pre, act_qat_param,
                self.cur_batch,
                self.common_config.get('need_sync'), self.common_config.get('process_group'),
                self.common_config.get('world_size'))

        act_h_qat_param = {
            NUM_BITS: self.act_config.get(NUM_BITS),
            FIXED_MIN: self.act_config.get(H_FIXED_MIN),
        }
        # forward with fake-quantized initial_h
        if self.replaced_module_type == 'LSTM':
            initial_h = hx[0]
        else:
            initial_h = hx
        quant_initial_h, scale_h, offset_h, clip_max_h, clip_min_h = \
            UlqRetrainFunction.apply(
                initial_h, self.acts_h_clip_max, self.acts_h_clip_min,
                self.acts_h_clip_max_pre, self.acts_h_clip_min_pre, act_h_qat_param,
                self.cur_batch,
                self.common_config.get('need_sync'), self.common_config.get('process_group'),
                self.common_config.get('world_size'))

        # Update quantization related parameters
        with torch.no_grad():
            copy_tensor(self.acts_scale, scale)
            copy_tensor(self.acts_offset, offset)
            copy_tensor(self.acts_clip_max, clip_max)
            copy_tensor(self.acts_clip_min, clip_min)
            copy_tensor(self.acts_clip_max_pre, clip_max)
            copy_tensor(self.acts_clip_min_pre, clip_min)
            copy_tensor(self.acts_h_scale, scale_h)
            copy_tensor(self.acts_h_offset, offset_h)
            copy_tensor(self.acts_h_clip_max, clip_max_h)
            copy_tensor(self.acts_h_clip_min, clip_min_h)
            copy_tensor(self.acts_h_clip_max_pre, clip_max_h)
            copy_tensor(self.acts_h_clip_min_pre, clip_min_h)
        return quant_inputs, quant_initial_h

    def _wts_quant(self, weights, rec_weights):
        """
        Function: weights quantization function.
        Inputs:
            weights: weights used for calibration in torch.tensor.
        """
        wts_quant_dict = {
            'arq_retrain': self._wts_quant_arq,
            'ulq_retrain': self._wts_quant_ulq,
        }

        quant_algo = self.wts_config.get('algo')
        # Forward with fake-quantized weights
        quant_weights, scales, offsets = wts_quant_dict.get(quant_algo)(
            weights, self.wts_scales, self.wts_offsets)
        # Forward with fake-quantized rec_weights
        quant_rec_weights, scales_r, offsets_r = wts_quant_dict.get(quant_algo)(
            rec_weights, self.rec_wts_scales, self.rec_wts_offsets)

        # Update quantization related parameters
        with torch.no_grad():
            copy_tensor(self.wts_scales, scales)
            copy_tensor(self.wts_offsets, offsets)
            copy_tensor(self.rec_wts_scales, scales_r)
            copy_tensor(self.rec_wts_offsets, offsets_r)
        return quant_weights, quant_rec_weights

    def _wts_quant_arq(self, weights, wts_scales, wts_offsets):
        """Call the core method of arq_retrain."""
        wts_param = {
            'num_bits': self.wts_config.get('num_bits'),
            'channel_wise': self.wts_config.get('channel_wise'),
            'with_offset': False,
            'module_type': self.replaced_module_type,
            'module': self.replaced_module,
        }
        wts_group = 1 if self.wts_config.get('channel_wise') else self.num_scales
        quantized_weight, scale, offset = ArqRetrainFunction.apply(
            weights, wts_scales, wts_offsets, wts_param, None, wts_group)
        return quantized_weight, scale, offset

    def _wts_quant_ulq(self, weights, wts_scales, wts_offsets):
        """Call the core method of ulq_retrain."""
        wts_param = {
            'num_bits': self.wts_config.get('num_bits'),
            'channel_wise': self.wts_config.get('channel_wise'),
            'with_offset': False,
            'arq_init': True,
            's_rec_flag': self.s_rec_flag,
            'module_type': self.replaced_module_type,
            'module': self.replaced_module,
        }
        wts_group = 1 if self.wts_config.get('channel_wise') else self.num_scales
        quantized_weight, scale, offset = UlqScaleRetrainFunction.apply(
            weights, wts_scales, wts_offsets, wts_param, self.cur_batch, None, wts_group)
        return quantized_weight, scale, offset

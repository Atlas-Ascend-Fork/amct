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

from collections import namedtuple
import numpy as np
import torch # pylint: disable=E0401
import torch.nn as nn # pylint: disable=E0401
from torch.nn.parameter import Parameter # pylint: disable=E0401

from ....amct_pytorch.custom_op.arq_retrain.arq_retrain import ArqRetrainFunction
from ....amct_pytorch.custom_op.ulq_retrain.ulq_retrain import UlqRetrainFunction
from ....amct_pytorch.custom_op.ulq_scale_retrain.ulq_scale_retrain import UlqScaleRetrainFunction
from ....amct_pytorch.custom_op.selective_prune.selective_prune import SelectivePruneFunction
from ....amct_pytorch.custom_op.selective_prune.selective_prune import add_mask_to_record, get_mask_from_record
from ....amct_pytorch.custom_op.utils import copy_tensor
from ....amct_pytorch.custom_op.utils import tensor
from ....amct_pytorch.utils.vars import FLT_EPSILON
from ....amct_pytorch.utils.log import LOGGER

COMP_ALG_PRUNE = 'prune'
COMP_ALG_QUANT = 'quant'
MASK_REFRESH = 'mask_refresh'
DEVICE = 'device'
S_REC_FLAG = 's_rec_flag'

ActsQuantInfo = namedtuple('ActsQuantInfo', ['outputs', 'scale', 'offset', 'clip_max', 'clip_min'])


class CompModuleBase(nn.Module):
    """
    Function: Base class module for quantized retrain.
    APIs: __init__, _init_output, forward
    """
    def __init__(self, module,  # pylint: disable=R0913
                 act_config=None,
                 wts_config=None,
                 common_config=None,
                 acts_comp_reuse=None):
        super(CompModuleBase, self).__init__()
        self.replaced_module = module
        self.replaced_module_type = module._get_name()
        self.act_config = act_config
        self.wts_config = wts_config
        self.common_config = common_config
        self.acts_comp_reuse = acts_comp_reuse
        self.comp_algs = []

    def forward(self, inputs):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        if inputs.abs().max() <= FLT_EPSILON:
            LOGGER.logw('The input tensor is all zeros')
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
        if self.acts_comp_reuse:
            compressed_inputs = self.acts_comp_reuse.acts_comp(
                inputs, self.act_config, self.common_config)
        else:
            compressed_inputs = self.acts_comp(
                inputs, self.act_config, self.common_config)

        # Compress weights.
        compressed_weights = self.wts_comp(
            self.replaced_module.weight, self.wts_config, self.common_config)

        if self.cur_batch < 100:
            self.cur_batch += 1
        return compressed_inputs, compressed_weights

    def acts_comp(self, inputs, act_config, common_config):
        """
        Function: activation function.
        Inputs:
            inputs: data used for calibration in torch.tensor.
            act_config: activation configuration parameters.
            common_config: quantized public configuration parameters.
        """
        acts_comp_types = {
            COMP_ALG_QUANT: self._acts_comp_quant,
            COMP_ALG_PRUNE: self._acts_comp_prune,
        }
        compressed_inputs = None
        if COMP_ALG_PRUNE in self.comp_algs:
            compressed_inputs = acts_comp_types.get(COMP_ALG_PRUNE)(inputs, act_config, common_config)
            inputs = compressed_inputs
        if COMP_ALG_QUANT in self.comp_algs:
            compressed_inputs = acts_comp_types.get(COMP_ALG_QUANT)(inputs, act_config, common_config)
        return compressed_inputs

    def wts_comp(self, weights, wts_config, common_config):
        """
        Function: weights quantization function.
        Inputs:
            weights: weights used for calibration in torch.tensor.
            wts_config: weights configuration parameters.
            common_config: quantized public configuration parameters.
        """
        wts_comp_types = {
            COMP_ALG_QUANT: self._wts_quant,
            COMP_ALG_PRUNE: self._wts_prune,
        }

        compressed_weights = None
        if COMP_ALG_PRUNE in self.comp_algs:
            compressed_weights = wts_comp_types.get(COMP_ALG_PRUNE)(weights, wts_config, common_config)
            weights = compressed_weights
        if COMP_ALG_QUANT in self.comp_algs:
            compressed_weights = wts_comp_types.get(COMP_ALG_QUANT)(weights, wts_config, common_config)
        return compressed_weights

    def _init_output(self):
        # Register quantitative parameters.
        self.register_buffer('cur_batch', tensor(0, device=self.common_config.get(DEVICE)))
        if self.act_config:
            if self.act_config.get('ifmr_init'):
                self.register_parameter(
                    'acts_clip_max',
                    Parameter(tensor(1.0, requires_grad=True, device=self.common_config.get(DEVICE))))
                self.register_parameter(
                    'acts_clip_min',
                    Parameter(tensor(1.0, requires_grad=True, device=self.common_config.get(DEVICE))))
            else:
                self.register_parameter(
                    'acts_clip_max',
                    Parameter(tensor(self.act_config.get('clip_max'),
                                    requires_grad=True, device=self.common_config.get(DEVICE))))
                self.register_parameter(
                    'acts_clip_min',
                    Parameter(tensor(self.act_config.get('clip_min'),
                                    requires_grad=True, device=self.common_config.get(DEVICE))))
            self.register_buffer('acts_clip_max_pre', tensor(np.nan, device=self.common_config.get(DEVICE)))
            self.register_buffer('acts_clip_min_pre', tensor(np.nan, device=self.common_config.get(DEVICE)))
            self.register_buffer('acts_scale', tensor(np.nan, device=self.common_config.get(DEVICE)))
            self.register_buffer('acts_offset', tensor(np.nan, device=self.common_config.get(DEVICE)))

        self.register_buffer('prune_cur_batch', tensor(0, device=self.common_config.get(DEVICE)))
        self.register_buffer('wts_mask', torch.ones_like(self.replaced_module.weight).mul(np.nan))

        self.register_parameter('wts_scales',
                                Parameter(tensor([np.nan] * self.num_scales,
                                          requires_grad=True, device=self.common_config.get(DEVICE))))
        self.register_parameter('wts_offsets',
                                Parameter(tensor([np.nan] * self.num_scales,
                                          requires_grad=True, device=self.common_config.get(DEVICE))))
        self.register_buffer(S_REC_FLAG,
            tensor(False if self.wts_config.get(S_REC_FLAG) is None else self.wts_config.get(S_REC_FLAG),
            device=self.common_config.get(DEVICE)))

    def _acts_comp_quant(self, inputs, act_config, common_config):
        """
        Function: activation quantization function.
        Inputs:
            inputs: data used for calibration in torch.tensor.
            act_config: activation configuration parameters.
            common_config: quantized public configuration parameters.
        """
        acts_quant_dict = {
            'ulq_quantize': self._acts_quant_ulq,
        }

        # Forward with fake-quantized activations.
        act_algo = act_config.get('algo')
        acts_quant_info = acts_quant_dict.get(act_algo)(inputs, act_config, common_config)
        quant_inputs = acts_quant_info.outputs

        # Update quantization related parameters.
        with torch.no_grad():
            copy_tensor(self.acts_scale, acts_quant_info.scale)
            copy_tensor(self.acts_offset, acts_quant_info.offset)
            copy_tensor(self.acts_clip_max, acts_quant_info.clip_max)
            copy_tensor(self.acts_clip_min, acts_quant_info.clip_min)
            copy_tensor(self.acts_clip_max_pre, acts_quant_info.clip_max)
            copy_tensor(self.acts_clip_min_pre, acts_quant_info.clip_min)
        return quant_inputs

    def _acts_comp_prune(self, inputs, act_config, common_config):
        return inputs

    def _acts_quant_ulq(self, inputs, act_config, common_config):
        """Call the core method of ulq_retrain."""
        act_qat_param = {
            'num_bits': act_config.get('num_bits'),
            'fixed_min': act_config.get('fixed_min'),
        }
        outputs, scale, offset, clip_max, clip_min = \
            UlqRetrainFunction.apply(
                inputs, self.acts_clip_max, self.acts_clip_min,
                self.acts_clip_max_pre, self.acts_clip_min_pre, act_qat_param,
                self.cur_batch,
                common_config.get('need_sync'), common_config.get('process_group'),
                common_config.get('world_size'))
        return ActsQuantInfo._make([outputs, scale, offset, clip_max, clip_min])

    def _wts_quant(self, weights, wts_config, common_config):
        """
        Function: weights quantization function.
        Inputs:
            weights: weights used for calibration in torch.tensor.
            wts_config: weights configuration parameters.
            common_config: quantized public configuration parameters.
        """
        wts_quant_dict = {
            'arq_retrain': self._wts_quant_arq,
            'ulq_retrain': self._wts_quant_ulq,
        }

        # Forward with fake-quantized weights.
        quant_algo = wts_config.get('algo')
        quant_weights, scales, offsets = \
            wts_quant_dict.get(quant_algo)(weights, wts_config, common_config)

        # Update quantization related parameters.
        with torch.no_grad():
            copy_tensor(self.wts_scales, scales)
            copy_tensor(self.wts_offsets, offsets)
        return quant_weights

    def _wts_quant_arq(self, weights, wts_config, common_config):
        """Call the core method of arq_retrain."""
        wts_param = {
            'num_bits': wts_config.get('num_bits'),
            'channel_wise': wts_config.get('channel_wise'),
            'with_offset': False,
            'module_type': self.replaced_module_type,
            'module': self.replaced_module,
        }
        quantized_weight, scale, offset = ArqRetrainFunction.apply(
            weights,
            self.wts_scales,
            self.wts_offsets,
            wts_param)
        return quantized_weight, scale, offset

    def _wts_quant_ulq(self, weights, wts_config, common_config):
        """Call the core method of ulq_retrain."""
        wts_param = {
            'num_bits': wts_config.get('num_bits'),
            'channel_wise': wts_config.get('channel_wise'),
            'with_offset': False,
            'arq_init': True,
            's_rec_flag': self.s_rec_flag,
            'module_type': self.replaced_module_type,
            'module': self.replaced_module,
        }
        quantized_weight, scale, offset = UlqScaleRetrainFunction.apply(
            weights,
            self.wts_scales, self.wts_offsets, wts_param, self.cur_batch)
        return quantized_weight, scale, offset

    def _wts_prune(self, weights, wts_config, common_config):
        wts_prune_dict = {
            'l1_selective_prune': self._wts_select_prune,
        }

        wts_config[MASK_REFRESH] = False
        if wts_config.get('update_freq') == 0:
            if self.prune_cur_batch == 0:
                wts_config[MASK_REFRESH] = True
        else:
            if self.prune_cur_batch % wts_config.get('update_freq') == 0:
                wts_config[MASK_REFRESH] = True

        self.prune_cur_batch += 1

        if wts_config.get(MASK_REFRESH):
            prune_algo = wts_config.get('prune_algo')
            updated_mask = wts_prune_dict.get(prune_algo)(weights, wts_config, common_config)
            add_mask_to_record(wts_config.get('layer_name'), updated_mask)

        with torch.no_grad():
            copy_tensor(self.wts_mask, get_mask_from_record(wts_config.get('layer_name')))
        pruned_weights = weights.mul(self.wts_mask)

        return pruned_weights

    def _wts_select_prune(self, weights, wts_config, common_config):
        updated_mask = SelectivePruneFunction.apply(
            weights,
            wts_config.get('n_out_of_m_type'),
            wts_config.get('prune_axis'))

        return updated_mask
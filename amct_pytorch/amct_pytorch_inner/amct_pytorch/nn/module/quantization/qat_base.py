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
from abc import ABCMeta, abstractmethod
import copy
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.nn.parameter import Parameter

from .....amct_pytorch.utils.log import LOGGER
from .....amct_pytorch.common.utils.check_params import check_params
from .....amct_pytorch.common.utils.vars_util import INT8, INT16
from .....amct_pytorch.common.utils.vars_util import RNN_TENSOR_NUM
from .....amct_pytorch.custom_op.ifmr.ifmr import IFMR
from .....amct_pytorch.custom_op.utils import copy_tensor
from .....amct_pytorch.custom_op.utils import get_distribute_config
from .....amct_pytorch.custom_op.ulq_retrain.ulq_retrain import UlqRetrainFuncQAT
from .....amct_pytorch.custom_op.arq_retrain.arq_retrain import ArqRetrainFuncQAT
from .....amct_pytorch.custom_op.ulq_scale_retrain.ulq_scale_retrain import UlqScaleRetrainFuncQAT
from .....amct_pytorch.utils.vars import CLIP_MAX, CLIP_MIN, FIXED_MIN, DST_TYPE, BATCH_NUM, NUM_BITS_MAP

TYPE = 'type'
SCOPE = 'scope'
DEFAULT_VALUE = 'default'
BIAS = 'bias'
SUPPORT_RETRAIN_ACT_WTS_TYPE_MAP = {
    INT8: INT8,
    INT16: INT8
}

ULQRetrainParams = namedtuple('ULQRetrainParams',
                              ['scale', 'offset', 'clip_max', 'clip_min', 'clip_max_pre', 'clip_min_pre'])
WtsRetrainParams = namedtuple('WtsRetrainParam', ['scale', 'offset', 'offset_deploy', 'wts_group_num'])


class QATBase(metaclass=ABCMeta):
    """
    Function: Base class module for qat op.
    """
    _float_module = None
    _required_params = list()

    @check_params(layer_type=str,
                  device=(str, type(None)),
                  config=(dict, type(None)))
    def __init__(self, layer_type, device, config=None):
        self.layer_type = layer_type
        self.quant_device = device
        self.cur = 0
        if config is None:
            config = dict()
        if not isinstance(config, dict):
            raise TypeError(f'config should be none or dict, but your input is {type(config)}')
        self.retrain_enable = config.get('retrain_enable', True)
        self.retrain_data_config = config.get('retrain_data_config', dict())
        if not isinstance(self.retrain_data_config, dict):
            raise TypeError(
                f'retrain_data_config should be none or dict, but your input is {type(config)}')
        self.retrain_weight_config = config.get(
            'retrain_weight_config', dict())
        if not isinstance(self.retrain_weight_config, dict):
            raise TypeError(
                f'retrain_weight_config should be none or dict, but your input is {type(config)}')
        self.distribute_config = get_distribute_config()
        
        # config has been checked in parsing if distill
        if not config.get('distill'):
            self._check_qat_config()

        if self.retrain_data_config.get(CLIP_MAX) is None:
            self.do_init = True
        else:
            self.do_init = False

        self.act_num_bits = NUM_BITS_MAP.get(self.retrain_data_config.get('dst_type', INT8))
        self.offset_dtype = torch.int16 if self.act_num_bits == 16 else torch.int8
        self.wts_num_bits = NUM_BITS_MAP.get(self.retrain_weight_config.get('dst_type', INT8))
        if not self.retrain_weight_config.get('channel_wise', True):
            self.wts_group_num = RNN_TENSOR_NUM.get(self.layer_type, 1)
        else:
            self.wts_group_num = 1
        self.check_quantifiable()

        self._register_qat_params(self)
        self.init_module = IFMR(layers_name=self.layer_type,
                                num_bits=self.act_num_bits,
                                batch_num=self.retrain_data_config.get(BATCH_NUM, 1))

    @staticmethod
    def do_ifmr(inputs, ifmr_module, distribute_config):
        is_init, scale_list, offset_list, clip_max, clip_min = ifmr_module.forward(inputs)
        scale = scale_list[0]
        offset = offset_list[0]
        if is_init:
            # sync in ddp mode
            world_size = distribute_config.get('world_size')
            process_group = distribute_config.get('process_group')
            if distribute_config.get('need_sync'):
                clip_max_all = torch.empty(
                    world_size, 1, dtype=clip_max.dtype, device=clip_max.device)
                clip_min_all = torch.empty(
                    world_size, 1, dtype=clip_min.dtype, device=clip_min.device)

                clip_max_l = list(clip_max_all.unbind(0))
                clip_min_l = list(clip_min_all.unbind(0))

                clip_max_all_reduce = torch.distributed.all_gather(
                    clip_max_l, clip_max, process_group, async_op=True)
                clip_min_all_reduce = torch.distributed.all_gather(
                    clip_min_l, clip_min, process_group, async_op=True)

                # wait on the async communication to finish
                clip_max_all_reduce.wait()
                clip_min_all_reduce.wait()
                clip_max_tmp = clip_max_all.mean()
                clip_min_tmp = clip_min_all.mean()
                clip_max.data.copy_(clip_max_tmp.data)
                clip_min.data.copy_(clip_min_tmp.data)
        else:
            clip_min = inputs.min()
            clip_max = inputs.max()

        return is_init, ULQRetrainParams(scale, offset, clip_max, clip_min, None, None)

    @staticmethod
    def do_ulq_retrain(inputs, ulq_retrain_params, retrain_data_config, distribute_config, training):
        """
        do ulq retrain process for activation
        """
        num_bits = NUM_BITS_MAP.get(retrain_data_config.get('dst_type', INT8))
        need_sync = distribute_config.get('need_sync') and training
        act_param = {
            'num_bits': num_bits,
            'ifmr_init': False,
            FIXED_MIN: retrain_data_config.get(FIXED_MIN, False),
            'acts_scale': ulq_retrain_params.scale,
            'acts_offset': ulq_retrain_params.offset,
            'asymmetric': retrain_data_config.get('asymmetric', True),
        }
        outputs, scale, offset, clip_max, clip_min = \
            UlqRetrainFuncQAT.apply(
                inputs, ulq_retrain_params.clip_max, ulq_retrain_params.clip_min,
                ulq_retrain_params.clip_max_pre, ulq_retrain_params.clip_min_pre,
                act_param, None, need_sync, distribute_config.get('process_group'),
                distribute_config.get('world_size')
            )
        return outputs, ULQRetrainParams(scale, offset, clip_max, clip_min, None, None)
    
    @staticmethod
    def _register_qat_params(module):
        """Register quantitative parameters
        """
        module.register_buffer('cur_batch', tensor(0))
        module.register_parameter('acts_clip_max',
                                  Parameter(tensor(module.retrain_data_config.get(CLIP_MAX, 1.0),
                                                   device=module.quant_device),
                                            requires_grad=True))
        module.register_parameter('acts_clip_min',
                                  Parameter(tensor(module.retrain_data_config.get(CLIP_MIN, -1.0),
                                                   device=module.quant_device),
                                            requires_grad=True))
        module.register_buffer('acts_clip_max_pre',
                               tensor(np.nan, device=module.quant_device))
        module.register_buffer('acts_clip_min_pre',
                               tensor(np.nan, device=module.quant_device))
        module.register_buffer('acts_scale',
                               tensor([np.nan], device=module.quant_device))
        module.register_buffer('acts_offset_deploy',
                               tensor([0], dtype=module.offset_dtype, device=module.quant_device))
        if module.retrain_weight_config.get('channel_wise', True):
            num_channel = module.out_channels
        else:
            num_channel = module.wts_group_num
        module.register_parameter('wts_scales',
                                  Parameter(tensor([np.nan] * num_channel,
                                                   device=module.quant_device),
                                            requires_grad=False))
        module.register_parameter('wts_offsets',
                                  Parameter(tensor([np.nan] * num_channel,
                                                   device=module.quant_device),
                                            requires_grad=False))
        module.register_buffer('wts_offsets_deploy',
                               tensor([0] * num_channel, dtype=torch.int8,
                                      device=module.quant_device))
        module.register_buffer('s_rec_flag',
                               tensor(module.retrain_weight_config.get('s_rec_flag', False),
                                      device=module.quant_device))

    @classmethod
    def from_float(cls,
                   mod,
                   config=None):
        """turn a torch.nn.Module to custom qat operator

        Args:
            mod (torch.nn.module): a float module, either produced by torch.quantization utilities or directly from user
            config (dict, optional): config used in wts quant and act quant. Defaults to DEFAULT_QAT_CONF.

        Returns:
            torch.nn.Module: qat operator constructed based on mod input
        """
        if not isinstance(mod, cls._float_module):
            raise TypeError(
                f'{cls.__name__}.from_float can only works for '
                f'{cls._float_module.__name__}')
        ori_op_params = cls._get_ori_op_params(mod)
        if BIAS in ori_op_params:
            ori_op_params[BIAS] = ori_op_params.get(BIAS) is not None
        qat_op = cls(config=config, **ori_op_params)
        setattr(qat_op, 'weight', mod.weight)
        setattr(qat_op, BIAS, mod.bias)
        qat_op.to(mod.weight.device)
        LOGGER.logi(
            f'Convert {cls._float_module.__name__} to QAT op successfully.')
        return qat_op
    
    @classmethod
    def _get_ori_op_params(cls, mod):
        """
        extract parameters from torch origin op.
        Args:
            mod (torch.nn.Module): a float module, either produced by torch.quantization utilities or directly from user
        """
        ori_op_params = {}
        err_params = list()
        for param in cls._required_params:
            if not hasattr(mod, param):
                err_params.append(param)
                continue
            ori_op_params[param] = getattr(mod, param)

        if err_params:
            raise RuntimeError(f'The following parameters are not found in the torch operator. {err_params}'\
                ' Check the _required_params parameter of the QAT operator.')
        return ori_op_params

    @abstractmethod
    def check_quantifiable(self):
        """Check whether the ori op support quant on its params
        """
        pass

    @abstractmethod
    def forward(self):
        pass

    def forward_qat(self, inputs):
        if inputs.dtype is not torch.float32:
            raise ValueError('Only support dtype torch.float32, but your input dtype is {}'.format(inputs.dtype))
        if self.retrain_enable:
            if self.do_init:
                self.acts_quant_init(inputs)
                quantized_acts = inputs
            else:
                quantized_acts = self.acts_quant(inputs)
            quantized_weights = self.wts_quant()
            if self.cur < 100:
                self.cur += 1
                self.cur_batch += 1
        else:
            quantized_acts, quantized_weights = inputs, self.weight
        return quantized_acts, quantized_weights

    def acts_quant_init(self, inputs):
        """do activations quant in the first batch
        """
        is_init, ulq_retrain_params = self.do_ifmr(
            inputs, self.init_module, self.distribute_config)
        with torch.no_grad():
            copy_tensor(self.acts_scale, ulq_retrain_params.scale)
            copy_tensor(self.acts_offset_deploy, ulq_retrain_params.offset.to(self.offset_dtype))
            copy_tensor(self.acts_clip_max, ulq_retrain_params.clip_max)
            copy_tensor(self.acts_clip_min, ulq_retrain_params.clip_min)
            copy_tensor(self.acts_clip_max_pre, ulq_retrain_params.clip_max)
            copy_tensor(self.acts_clip_min_pre, ulq_retrain_params.clip_min)

        self.do_init = not is_init
        if not self.do_init:
            del self.init_module

    def acts_quant(self, inputs):
        """
        Call the core method of ulq_retrain.
        """
        ulq_retrain_params = ULQRetrainParams(self.acts_scale, self.acts_offset_deploy,
                                              self.acts_clip_max, self.acts_clip_min,
                                              self.acts_clip_max_pre, self.acts_clip_min_pre)
        outputs, ulq_retrain_params_update = self.do_ulq_retrain(
            inputs, ulq_retrain_params, self.retrain_data_config, self.distribute_config, self.training)
        # Update quantization related parameters.
        with torch.no_grad():
            copy_tensor(self.acts_scale, ulq_retrain_params_update.scale)
            copy_tensor(self.acts_offset_deploy, ulq_retrain_params_update.offset.to(self.offset_dtype))
            copy_tensor(self.acts_clip_max, ulq_retrain_params_update.clip_max)
            copy_tensor(self.acts_clip_min, ulq_retrain_params_update.clip_min)
            copy_tensor(self.acts_clip_max_pre, ulq_retrain_params_update.clip_max)
            copy_tensor(self.acts_clip_min_pre, ulq_retrain_params_update.clip_min)
        return outputs

    def wts_quant(self):
        """do weight quant using arq or ulq algo based on config
        """
        wts_quant_dict = {
            'arq_retrain': self.wts_quant_arq,
            'ulq_retrain': self.wts_quant_ulq
        }
        weights_retrain_algo = self.retrain_weight_config.get('weights_retrain_algo', 'arq_retrain')
        wts_quant_params = WtsRetrainParams(self.wts_scales,
                                            self.wts_offsets,
                                            self.wts_offsets_deploy,
                                            self.wts_group_num)
        quantized_weight, scales, offsets = \
            wts_quant_dict.get(weights_retrain_algo)(self.weight, wts_quant_params)

        with torch.no_grad():
            copy_tensor(self.weight, quantized_weight)
            copy_tensor(self.wts_scales, scales)
            copy_tensor(self.wts_offsets, offsets)
            copy_tensor(self.wts_offsets_deploy, offsets.to(torch.int8))
        return quantized_weight

    def wts_quant_arq(self, wts, wts_retrain_params):
        """do weight quant using arq algo
        """
        wts_param = {
            'num_bits': self.wts_num_bits,
            'channel_wise': self.retrain_weight_config.get('channel_wise', True),
            'with_offset': False,
            'module_type': self.layer_type,
            'module': self
        }
        quantized_weight, scale, offset = ArqRetrainFuncQAT.apply(wts,
                                                                  wts_retrain_params.scale,
                                                                  wts_retrain_params.offset,
                                                                  wts_param,
                                                                  wts_retrain_params.offset_deploy,)
        return quantized_weight, scale, offset

    def wts_quant_ulq(self, wts, wts_retrain_params):
        """do weight quant using ulq algo
        """
        wts_param = {
            'num_bits': self.wts_num_bits,
            'channel_wise': self.retrain_weight_config.get('channel_wise', True),
            'with_offset': False,
            'module_type': self.layer_type,
            's_rec_flag': self.retrain_weight_config.get('s_rec_flag', False),
            'arq_init': True,
            'module': self
        }
        quantized_weight, scale, offset = UlqScaleRetrainFuncQAT.apply(wts,
                                                                       wts_retrain_params.scale,
                                                                       wts_retrain_params.offset,
                                                                       wts_param,
                                                                       self.cur_batch,
                                                                       wts_retrain_params.offset_deploy,
                                                                       wts_retrain_params.wts_group_num)
        return quantized_weight, scale, offset

    def _check_qat_config(self):
        """
        Check if the QAT configuration meets the requirements.
        """
        # check params for activation
        if not self.retrain_data_config.get(DST_TYPE, INT8) in [INT8, INT16]:
            raise ValueError("dst_type for activation should be in range ['INT8', 'INT16'], "
                             "but your input is {}".format(self.retrain_data_config.get(DST_TYPE)))
        
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

        # check params for weights
        if not self.retrain_weight_config.get(DST_TYPE, INT8) in [INT8]:
            raise ValueError("dst_type for weight should be in range ['INT8'], "
                             "but your input is {}".format(self.retrain_weight_config.get(DST_TYPE)))

        if not self.retrain_weight_config.get('weight_retrain_algo', 'arq_retrain') in ['arq_retrain', 'ulq_retrain']:
            raise ValueError("weight_retrain_algo should be in range ['arq_retrain', 'ulq_retrain'], "
                             "but your input is {}".format(self.retrain_weight_config.get(DST_TYPE)))

        if not isinstance(self.retrain_weight_config.get('channel_wise', True), bool):
            raise ValueError("channel_wise should be a bool value, but your input is {}".format(
                type(self.retrain_weight_config.get('channel_wise'))))
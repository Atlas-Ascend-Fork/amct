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
from torch import nn

from ...amct_pytorch.custom_op.comp_module.comp_module_base import CompModuleBase
from ...amct_pytorch.custom_op.comp_module.comp_module_rnn import CompModuleRNN
from ...amct_pytorch.custom_op.ifmr.ifmr import IFMR
from ...amct_pytorch.custom_op.utils import copy_tensor
from ...amct_pytorch.custom_op.utils import tensor
from ...amct_pytorch.custom_op.utils import process_scale
from ...amct_pytorch.utils.vars import BATCH_NUM
from ...amct_pytorch.common.utils.vars_util import DEFAULT, FORCE_FP16_QUANT

LAYERS_NAME = 'layers_name'
DEVICE = 'device'
SCALE_D = 'scale_d'
SCALE_W = 'scale_w'
DATA_NUM_BITS = 'data_num_bits'
WTS_NUM_BITS = 'wts_num_bits'
NUM_BITS = 'num_bits'


class RetrainQuant(nn.Module):
    """
    record quantization factor.
    """
    def __init__(self, quant_module, record_module, bn_module=None, bn_module_name=None):
        """
        Function: init function
        Inputs:
            quant_module: a torch.nn.Module inculdes conv/linear, ulq_retrain,
                arq_retrain.
            record_module: a torch.nn.Module recording quantification factor.
            bn_module: a torch.nn.Module record batchnorm.
            bn_module_name: a string of the batchnorm's name.
        Returns: None
        """
        super(RetrainQuant, self).__init__()
        self.common_config = quant_module.common_config
        self.quant_module = quant_module
        self.do_init = quant_module.act_config.get('ifmr_init')
        self.init_module = None
        if self.do_init:
            self.init_module = IFMR(
                num_bits=self.quant_module.act_config.get(NUM_BITS),
                layers_name=self.common_config.get(LAYERS_NAME),
                batch_num=self.common_config.get(BATCH_NUM))
        self.record_module = record_module
        self.bn_module = bn_module
        self.bn_module_name = bn_module_name
        self.cur_batch = 0
        self.type = 'RetrainQuant'
        # if channel-wise F, num_scales should keep same with bn features.
        if self.bn_module and self.quant_module.num_scales == 1:
            self.quant_module.num_scales *= self.bn_module.num_features

        self.scale_w_old = \
            tensor([0.0], device=self.common_config.get(DEVICE))
        self.scale_d_old = \
            tensor(0.0, device=self.common_config.get(DEVICE))
        self.scale_d = 1.0
        # register buffer space FIRST instead of reference in update_quant_factor.
        self.register_buffer(
            SCALE_W,
            tensor([1.0] * self.quant_module.num_scales)
        )
        self.offset_d = 0.0
        # register buffer space FIRST instead of reference in update_quant_factor.
        self.register_buffer(
            'offset_w',
            tensor([0.0] * self.quant_module.num_scales)
        )
        self.write_done_flag = False

    def update_quant_factor(self):
        """
        Function: update quant factor function.
        Inputs: None
        Returns: None
        """
        if self.quant_module.acts_comp_reuse:
            quant_module = self.quant_module.acts_comp_reuse
        else:
            quant_module = self.quant_module
        self.scale_d = quant_module.acts_scale
        self.offset_d = quant_module.acts_offset
        # assignment instead of reference
        # if, channel-wise False, has bn, quant_module.num_scales has expand.
        # else, channel-wise True.
        for idx in range(self.quant_module.num_scales):
            scale_id = 0
            if self.quant_module.wts_scales.numel() != 1:
                scale_id = idx
            wts_scale = self.quant_module.wts_scales.data[scale_id]
            self.scale_w.data[idx] = 1 / wts_scale if self.quant_module.s_rec_flag else wts_scale

            offset_id = 0
            if self.quant_module.wts_scales.numel() != 1:
                offset_id = idx
            self.offset_w.data[idx] = self.quant_module.wts_offsets.data[offset_id]

        if self.bn_module:
            bn_var_rsqrt = torch.rsqrt(self.bn_module.running_var + self.bn_module.eps)
            scale = self.bn_module.weight * bn_var_rsqrt
            # if channel-wise F & BN. scale_w and offset_w expand.
            self.scale_w = self.scale_w.to(scale.device).abs() * abs(scale).detach()

        self.scale_d, self.offset_d = process_scale(
            self.scale_d, self.offset_d, True,
            self.quant_module.act_config.get(NUM_BITS))
        self.scale_w, self.offset_w = process_scale(
            self.scale_w, self.offset_w, False,
            self.quant_module.wts_config.get(NUM_BITS))

    def acts_quant_init(self, inputs):
        """
        Function: use ifmr to init act_clip_max/act_clip_min of quant_module.
        Inputs:
            inputs: a torch.tensor, activations.
        Returns: None
        """
        quant_info = self.init_module.forward(inputs)
        is_init = quant_info.flag
        scale = quant_info.scale[0]
        offset = quant_info.offset[0]
        clip_max = quant_info.clip_max
        clip_min = quant_info.clip_min
        if is_init:
            # sync in ddp mode
            process_group = torch.distributed.group.WORLD
            try:
                world_size = torch.distributed.get_world_size(process_group)
            except (AttributeError, AssertionError, RuntimeError, ValueError):
                process_group = None
                world_size = 1

            need_sync = world_size > 1
            if need_sync:
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
            # delete after init
            del self.init_module
            self.init_module = None
        else:
            clip_min = inputs.min()
            clip_max = inputs.max()

        with torch.no_grad():
            if not self.quant_module.acts_comp_reuse:
                copy_tensor(self.quant_module.acts_scale, scale)
                copy_tensor(self.quant_module.acts_offset, offset)
                copy_tensor(self.quant_module.acts_clip_max, clip_max)
                copy_tensor(self.quant_module.acts_clip_min, clip_min)
                copy_tensor(self.quant_module.acts_clip_max_pre, clip_max)
                copy_tensor(self.quant_module.acts_clip_min_pre, clip_min)

        self.do_init = not is_init

    def forward(self, inputs): # pylint: disable=W0221
        """
        Function: forward function
        Inputs:
            inputs: a torch.tensor, activations.
        Returns:
            outputs: a torch.tensor, quantized output data.
        """
        if self.do_init:
            self.acts_quant_init(inputs)
        layers_name = self.common_config.get(LAYERS_NAME)
        outputs = self.quant_module(inputs)
        if self.bn_module:
            outputs = self.bn_module(outputs)

        if not self.training and not self.do_init:
            self.cur_batch += 1
            if self.cur_batch == 1:
                self.update_quant_factor()

            if self.common_config.get('fakequant_precision_mode') == FORCE_FP16_QUANT:
                self.record_module.fakquant_precision_mode = FORCE_FP16_QUANT

            if not self.write_done_flag:
                self.record_module(
                    layers_name, 'ifmr',
                    {SCALE_D: self.scale_d.cpu().tolist(),
                     'offset_d': int(self.offset_d.cpu().tolist()),
                     'num_bits': self.quant_module.act_config.get(NUM_BITS)})
                self.record_module(
                    layers_name, 'arq',
                    {SCALE_W: self.scale_w.cpu().tolist(),
                     'offset_w': list(
                         map(int, self.offset_w.cpu().tolist())),
                     'num_bits': self.quant_module.wts_config.get(NUM_BITS)})
                self.write_done_flag = True
        else:
            self.cur_batch = 0
            self.record_module.record_quant_layer(layers_name)

        return outputs


def get_quant_type(module):
    """get quantized module's type from  module"""
    if isinstance(module, CompModuleBase) or isinstance(module, CompModuleRNN):
        return type(module.replaced_module).__name__
    return type(module).__name__


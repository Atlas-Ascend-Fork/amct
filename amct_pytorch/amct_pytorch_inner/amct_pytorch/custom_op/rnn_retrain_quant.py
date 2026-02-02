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
import torch
from torch import nn

from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE
from ...amct_pytorch.custom_op.ifmr.ifmr import IFMR
from ...amct_pytorch.custom_op.utils import copy_tensor
from ...amct_pytorch.custom_op.utils import tensor
from ...amct_pytorch.custom_op.utils import process_scale
from ...amct_pytorch.custom_op.retrain_quant import get_quant_type
from ...amct_pytorch.utils.vars import BATCH_NUM

DATA_NUM_BITS = 'data_num_bits'
WTS_NUM_BITS = 'wts_num_bits'
NUM_BITS = 'num_bits'
LAYERS_NAME = 'layers_name'
RNN_TENSOR_SEQUENCE = {
    'LSTM': [0, 3, 1, 2],
    'GRU': [1, 0, 2]
}
ActRetrainParams = namedtuple('ActRetrainParams',
                              ['scale', 'offset', 'clip_max', 'clip_min', 'clip_max_pre', 'clip_min_pre'])


class RNNRetrainQuant(nn.Module):
    """
    record quantization factor.
    """
    def __init__(self, quant_module, record_module):
        """
        Function: init function
        Inputs:
            quant_module: a torch.nn.Module includes lstm/gru, ulq_retrain, arq_retrain.
            record_module: a torch.nn.Module recording quantification factor.
        Returns: None
        """
        super().__init__()
        self.common_config = quant_module.common_config
        self.quant_module = quant_module
        self.quant_module_type = get_quant_type(quant_module)
        self.do_init = quant_module.act_config.get('ifmr_init')
        self.init_module = None
        self.init_module_h = None
        if self.do_init:
            self.init_module = IFMR(
                num_bits=self.quant_module.act_config.get(NUM_BITS),
                layers_name=self.common_config.get(LAYERS_NAME),
                batch_num=self.common_config.get(BATCH_NUM))
            self.init_module_h = IFMR(
                num_bits=self.quant_module.act_config.get(NUM_BITS),
                layers_name=self.common_config.get(LAYERS_NAME),
                batch_num=self.common_config.get(BATCH_NUM))
        self.record_module = record_module
        self.cur_batch = 0
        self.type = 'RNNRetrainQuant'

        self.scale_d = 1.0
        self.scale_h = 1.0
        self.offset_d = 0.0
        self.offset_h = 0.0
        # register buffer space FIRST instead of reference in update_quant_factor.
        self.register_buffer('scale_w', tensor([1.0] * self.quant_module.num_scales))
        self.register_buffer('scale_r', tensor([1.0] * self.quant_module.num_scales))
        self.register_buffer('offset_w', tensor([0.0] * self.quant_module.num_scales))
        self.register_buffer('offset_r', tensor([0.0] * self.quant_module.num_scales))
        self.write_done_flag = False

    @staticmethod
    def _do_ifmr(inputs, init_module):
        """
        Function: use ifmr to init act_clip_max/act_clip_min of quant_module.
        Inputs:
            inputs: a torch.tensor, activations.
            init_module: IFMR module
        Returns: None
        """
        quant_info = init_module.forward(inputs)
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
        else:
            clip_min = inputs[0].min()
            clip_max = inputs[0].max()

        return is_init, ActRetrainParams._make([scale, offset, clip_max, clip_min, clip_max, clip_min])

    @staticmethod
    def _reorganize_rnn_quant_factor(quant_factor, module_name, module_type):
        '''
        Reorganize the rnn weight quant factor from torch order to onnx order.
        '''
        length = len(quant_factor)
        tensor_sequence = RNN_TENSOR_SEQUENCE.get(module_type)
        if length % len(tensor_sequence) != 0:
            raise RuntimeError(
                'Layer\'s quant factor length {} is not suitable mutiple of {}'.format(length, module_name))
        splited_quant_factor = np.split(quant_factor, len(tensor_sequence))
        temp_list = list()
        for idx in tensor_sequence:
            temp_list.append(splited_quant_factor[idx])
        reorganized_quant_factor = np.concatenate(temp_list, axis=0)
        return reorganized_quant_factor.tolist()

    def forward(self, inputs, hx=None):
        """
        Function: forward function
        Inputs:
            inputs: a torch.tensor, activations.
            hx: a torch.tensor, initial_h
        Returns:
            outputs: a torch.tensor, quantized output data.
        """
        layers_name = self.common_config.get(LAYERS_NAME)
        if not isinstance(inputs, torch.Tensor):
            raise ValueError("Only support input type \'torch.Tensor\'.")
        if len(inputs.shape) != 3:
            raise ValueError("Layer {} input data only support 3-D shape.".format(layers_name))
        if self.quant_module_type in RNN_LAYER_TYPE:
            if self.quant_module.replaced_module.batch_first:
                sequence_length = inputs.shape[1]
            else:
                sequence_length = inputs.shape[0]
            if sequence_length != 1:
                raise ValueError("Layer {} sequence length only support 1, actually is {}.".format(
                    layers_name, sequence_length))
        if hx is None:
            raise ValueError("Layer {} except second input, bu got None.".format(layers_name))

        if self.do_init:
            self._acts_quant_init(inputs, hx)
        outputs = self.quant_module(inputs, hx)

        if not self.training and not self.do_init:
            self.cur_batch += 1
            if self.cur_batch == 1:
                self._update_quant_factor()

            if not self.write_done_flag:
                self.record_module(
                    layers_name, 'ifmr',
                    {'scale_d': self.scale_d.cpu().tolist(),
                     'scale_h': self.scale_h.cpu().tolist(),
                     'offset_d': int(self.offset_d.cpu().tolist()),
                     'offset_h': int(self.offset_h.cpu().tolist()),
                     'num_bits': self.quant_module.act_config.get(NUM_BITS)})
                self.record_module(
                    layers_name, 'arq',
                    {'scale_w': self._reorganize_rnn_quant_factor(
                        self.scale_w.cpu().numpy(), layers_name[0], self.quant_module_type),
                     'scale_r': self._reorganize_rnn_quant_factor(
                        self.scale_r.cpu().numpy(), layers_name[0], self.quant_module_type),
                     'offset_w': self._reorganize_rnn_quant_factor(
                        self.offset_w.cpu().numpy().astype(np.int32), layers_name[0], self.quant_module_type),
                     'offset_r': self._reorganize_rnn_quant_factor(
                        self.offset_r.cpu().numpy().astype(np.int32), layers_name[0], self.quant_module_type),
                     'num_bits': self.quant_module.wts_config.get(NUM_BITS)})
                self.write_done_flag = True
        else:
            self.cur_batch = 0
            self.record_module.record_quant_layer(layers_name)

        return outputs

    def _update_quant_factor(self):
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
        self.scale_h = quant_module.acts_h_scale
        self.offset_h = quant_module.acts_h_offset

        for idx in range(self.quant_module.num_scales):
            scale_id = 0
            if self.quant_module.wts_scales.numel() != 1:
                scale_id = idx
            wts_scale = self.quant_module.wts_scales.data[scale_id]
            rec_wts_scale = self.quant_module.rec_wts_scales.data[scale_id]
            self.scale_w.data[idx] = 1 / wts_scale if self.quant_module.s_rec_flag else wts_scale
            self.scale_r.data[idx] = 1 / rec_wts_scale if self.quant_module.s_rec_flag else rec_wts_scale

            offset_id = 0
            if self.quant_module.wts_scales.numel() != 1:
                offset_id = idx
            self.offset_w.data[idx] = self.quant_module.wts_offsets.data[offset_id]
            self.offset_r.data[idx] = self.quant_module.rec_wts_offsets.data[offset_id]

        self.scale_d, self.offset_d = process_scale(
            self.scale_d, self.offset_d, True,
            self.quant_module.act_config.get(NUM_BITS))
        self.scale_h, self.offset_h = process_scale(
            self.scale_h, self.offset_h, True,
            self.quant_module.act_config.get(NUM_BITS))
        self.scale_w, self.offset_w = process_scale(
            self.scale_w, self.offset_w, False,
            self.quant_module.wts_config.get(NUM_BITS))
        self.scale_r, self.offset_r = process_scale(
            self.scale_r, self.offset_r, False,
            self.quant_module.wts_config.get(NUM_BITS))

    def _acts_quant_init(self, inputs, hx):
        """
        Function: do ifmr for activations and initial_h
        Inputs:
            inputs: a torch.tensor, activations.
            hx: a torch.tensor, initial_h
        Returns: None
        """
        is_init, act_retrain_params = self._do_ifmr(inputs, self.init_module)
        if is_init:
            del self.init_module
            self.init_module = None

        if self.quant_module_type == 'LSTM':
            initial_h = hx[0]
        else:
            initial_h = hx
        is_init_h, act_h_retrain_params = self._do_ifmr(initial_h, self.init_module_h)
        if is_init_h:
            del self.init_module_h
            self.init_module_h = None

        with torch.no_grad():
            if not self.quant_module.acts_comp_reuse:
                copy_tensor(self.quant_module.acts_scale, act_retrain_params.scale)
                copy_tensor(self.quant_module.acts_offset, act_retrain_params.offset)
                copy_tensor(self.quant_module.acts_clip_max, act_retrain_params.clip_max)
                copy_tensor(self.quant_module.acts_clip_min, act_retrain_params.clip_min)
                copy_tensor(self.quant_module.acts_clip_max_pre, act_retrain_params.clip_max_pre)
                copy_tensor(self.quant_module.acts_clip_min_pre, act_retrain_params.clip_min_pre)
                copy_tensor(self.quant_module.acts_h_scale, act_h_retrain_params.scale)
                copy_tensor(self.quant_module.acts_h_offset, act_h_retrain_params.offset)
                copy_tensor(self.quant_module.acts_h_clip_max, act_h_retrain_params.clip_max)
                copy_tensor(self.quant_module.acts_h_clip_min, act_h_retrain_params.clip_min)
                copy_tensor(self.quant_module.acts_h_clip_max_pre, act_h_retrain_params.clip_max_pre)
                copy_tensor(self.quant_module.acts_h_clip_min_pre, act_h_retrain_params.clip_min_pre)

        self.do_init = not is_init
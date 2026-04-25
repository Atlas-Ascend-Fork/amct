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
import torch
import numpy as np
from torch import tensor
from torch.nn.parameter import Parameter
 
from .....amct_pytorch.nn.module.quantization.qat_base import QATBase, ULQRetrainParams, WtsRetrainParams
from .....amct_pytorch.custom_op.ifmr.ifmr import IFMR
from .....amct_pytorch.utils.vars import CLIP_MAX, CLIP_MIN, BATCH_NUM
from .....amct_pytorch.custom_op.utils import copy_tensor
from .....amct_pytorch.common.utils.util import quant_dequant_tensor
 
RNN_TENSOR_SEQUENCE = {
    'LSTM': ([0, 3, 1, 2], [0, 2, 3, 1]),
    'GRU': ([1, 0, 2], [1, 0, 2])
}
RNN_INPUT_DIM = 3
 
 
class RnnQatBase(QATBase):
    def __init__(self, layer_type, device, config):
        super().__init__(layer_type, device, config)
        self._register_rnn_qat_params()
        if self.retrain_data_config.get(CLIP_MAX) is None:
            self.h_do_init = True
            self.h_init_module = IFMR(layers_name=self.layer_type,
                                      num_bits=self.act_num_bits,
                                      batch_num=self.retrain_data_config.get(BATCH_NUM, 1))
        else:
            self.h_do_init = False

        if not config.get('from_float'):
            self.weight = Parameter(RnnQatBase.reorganize_tensor(
                self.weight_ih_l0, self.layer_type).to(device=device))
            self.recurrence_weight = Parameter(RnnQatBase.reorganize_tensor(
                self.weight_hh_l0, self.layer_type).to(device=device))
            if hasattr(self, 'bias_hh_l0'):
                self.bias_value = torch.cat([RnnQatBase.reorganize_tensor(self.bias_ih_l0.detach(), self.layer_type),
                                       RnnQatBase.reorganize_tensor(
                                        self.bias_hh_l0.detach(), self.layer_type)]).reshape(1, -1)
            else:
                self.bias_value = None
 
    @staticmethod
    def reorganize_tensor(data, layer_type, back=False):
        '''
        convert format from ifgo to icof if back is false else convert it backward
        '''
        def concat_tensor_on_sequence(splited_tensors, sequence):
            tensor_list = list()
            for idx in sequence:
                tensor_list.append(splited_tensors[idx])
            return torch.cat(tensor_list)
        tensor_sequence = RNN_TENSOR_SEQUENCE.get(layer_type)
        hidden_size = data.shape[0] // len(tensor_sequence[0])
        splited_tensor = torch.split(data, int(hidden_size), dim=0)
        reoriganized_tensor = concat_tensor_on_sequence(
            splited_tensor, tensor_sequence[int(back)])
        return reoriganized_tensor
 
    @staticmethod
    def set_rnn_op_trained_params(mod, qat_op, layer_type):
        """
        set weight, recurrence_weight for training and bias for exporting onnx model
        """
        device = mod.weight_ih_l0.device
        weight = torch.nn.Parameter(RnnQatBase.reorganize_tensor(mod.weight_ih_l0, layer_type).to(device=device))
        recurrence_weight = torch.nn.Parameter(
            RnnQatBase.reorganize_tensor(mod.weight_hh_l0, layer_type).to(device=device))
        setattr(qat_op, 'weight', weight)
        setattr(qat_op, 'recurrence_weight', recurrence_weight)
        
        if mod.bias:
            bias = torch.nn.Parameter(mod.bias_ih_l0.to(device=device))
            setattr(qat_op, 'bias_ih_l0', bias)
            recurrence_bias = torch.nn.Parameter(mod.bias_hh_l0.to(device=device))
            setattr(qat_op, 'bias_hh_l0', recurrence_bias)
            setattr(qat_op, 'bias_value',
                    torch.cat([RnnQatBase.reorganize_tensor(bias.detach(), layer_type),
                               RnnQatBase.reorganize_tensor(recurrence_bias.detach(), layer_type)]).reshape(1, -1))
        else:
            setattr(qat_op, 'bias_value', None)

    def check_quantifiable(self):
        """check qat config for rnn qat op"""
        if self.num_layers != 1:
            raise RuntimeError('Do not support {} with num_layers {}'.format(self.layer_type, self.num_layers))
        if self.bidirectional:
            raise RuntimeError('Do not support {} with bidirectional {}'.format(self.layer_type, self.bidirectional))
        if self.dropout != 0:
            raise RuntimeError('Do not support {} with dropout {}'.format(self.layer_type, self.dropout))
        return True

    def check_input_data(self, inputs, hx):
        """check type and shape of input data"""
        if hx is None:
            raise RuntimeError('Not support no hx input')
        if not isinstance(inputs, torch.Tensor):
            raise RuntimeError('{}QAT only support tensor input, but your input type is {}'.format(
                self.layer_type, type(input)))

        if not isinstance(inputs, torch.Tensor):
            raise RuntimeError('{}QAT op only support tensor input, but your input type is {}'.format(
                self.layer_type, type(input)))

        if len(inputs.shape) != RNN_INPUT_DIM:
            raise RuntimeError("{} quantize only support input dim 3,"\
                               " but your input dim is {}".format(self.layer_type, len(inputs.shape)))

    def for_loop_rnncell_forward(self, inputs, hx, module_type, params):
        h_next = hx
        c_next = None
        seq_len, w, r, bw, br = params
        if module_type == "GRU":
            h_next = h_next.squeeze(0) if h_next.dim() == 3 else h_next
            rnn_cell = torch.nn.GRUCell(inputs.shape[-1], h_next.shape[-1])
        else:
            h_next, c_next = hx
            h_next = h_next.squeeze(0) if h_next.dim() == 3 else h_next
            c_next = c_next.squeeze(0) if c_next.dim() == 3 else c_next
            rnn_cell = torch.nn.LSTMCell(inputs.shape[-1], h_next.shape[-1])
        rnn_cell.weight_ih = w
        rnn_cell.weight_hh = r
        rnn_cell.bias_ih = bw
        rnn_cell.bias_hh = br
        output_cell = []
        current_state = None
        for i in range(0, seq_len):
            h_next = quant_dequant_tensor(h_next, self.h_scale, self.h_offset_deploy, self.act_num_bits)
            xi = inputs[:, i, :]
            if module_type == "GRU":
                h_next = rnn_cell(xi, h_next)
                output_cell.append(h_next)
                current_state = h_next
            else:
                h_next, c_next = rnn_cell(xi, (h_next, c_next))
                output_cell.append(h_next)
                current_state = (h_next, c_next)

        full_output = torch.stack(output_cell, dim=1)
        return full_output, current_state 
    
    def forward_qat(self, inputs, hx, seq_len):
        if self.layer_type == "LSTM":
            initial_h, _ = hx
        else:
            initial_h = hx # 1, B, H
        if self.retrain_enable:
            quantized_weights = self.wts_quant()
            quantized_recurrence_weights = self._recurrence_wts_quant()
            if self.h_do_init:
                self.acts_quant_init(inputs)
                self._h_quant_init(initial_h)
                quantized_acts = inputs
                quantized_h = initial_h
            else:
                quantized_acts = self.acts_quant(inputs)
                if seq_len > 1:
                    params = (seq_len, self.weight, self.recurrence_weight,
                              self.bias_ih_l0, self.bias_hh_l0)
                    initial_h, _ = self.for_loop_rnncell_forward(inputs, hx, 
                                                                 self.layer_type, params)
                quantized_h = self._h_quant(initial_h)
                if seq_len > 1:
                    quantized_h = quantized_h[:, 0:1, :]
            if self.cur < 100:
                self.cur += 1
                self.cur_batch += 1
        else:
            quantized_acts, quantized_weights = inputs, self.weight
            quantized_h, quantized_recurrence_weights = initial_h, self.recurrence_weight
        ret = (quantized_acts, quantized_h, quantized_weights, quantized_recurrence_weights)
        return ret
    
    def _h_quant_init(self, initial_h):
        """do activations quant in the first batch
        """
        is_init, ulq_retrain_params = self.do_ifmr(
            initial_h, self.h_init_module, self.distribute_config)
 
        with torch.no_grad():
            copy_tensor(self.h_scale, ulq_retrain_params.scale)
            copy_tensor(self.h_offset_deploy, ulq_retrain_params.offset.to(self.offset_dtype))
            copy_tensor(self.h_clip_max, ulq_retrain_params.clip_max)
            copy_tensor(self.h_clip_min, ulq_retrain_params.clip_min)
            copy_tensor(self.h_clip_max_pre, ulq_retrain_params.clip_max)
            copy_tensor(self.h_clip_min_pre, ulq_retrain_params.clip_min)
        self.h_do_init = not is_init
        if not self.h_do_init:
            del self.h_init_module
 
    def _h_quant(self, initial_h):
        """
        Call the core method of ulq_retrain.
        """
        ulq_retrain_params = ULQRetrainParams(self.h_scale, self.h_offset_deploy,
                                              self.h_clip_max, self.h_clip_min,
                                              self.h_clip_max_pre, self.h_clip_min_pre)
        outputs, ulq_retrain_params_update = self.do_ulq_retrain(
            initial_h, ulq_retrain_params, self.retrain_data_config, self.distribute_config, self.training)
        # Update quantization related parameters.
        with torch.no_grad():
            copy_tensor(self.h_scale, ulq_retrain_params_update.scale)
            copy_tensor(self.h_offset_deploy, ulq_retrain_params_update.offset.to(self.offset_dtype))
            copy_tensor(self.h_clip_max, ulq_retrain_params_update.clip_max)
            copy_tensor(self.h_clip_min, ulq_retrain_params_update.clip_min)
            copy_tensor(self.h_clip_max_pre, ulq_retrain_params_update.clip_max)
            copy_tensor(self.h_clip_min_pre, ulq_retrain_params_update.clip_min)
        return outputs
  
    def _recurrence_wts_quant(self):
        """do weight quant using arq or ulq algo based on config
        """
        wts_quant_dict = {
            'arq_retrain': self.wts_quant_arq,
            'ulq_retrain': self.wts_quant_ulq
        }
        weights_retrain_algo = self.retrain_weight_config.get('weights_retrain_algo', 'arq_retrain')
        rec_wts_param = WtsRetrainParams(self.recurrence_wts_scales,
                                         self.recurrence_wts_offsets,
                                         self.recurrence_wts_offsets_deploy,
                                         self.wts_group_num)
        quantized_recurrence_weight, scales, offsets = \
            wts_quant_dict.get(weights_retrain_algo)(self.recurrence_weight, rec_wts_param)
 
        with torch.no_grad():
            copy_tensor(self.recurrence_weight, quantized_recurrence_weight)
            copy_tensor(self.recurrence_wts_scales, scales)
            copy_tensor(self.recurrence_wts_offsets, offsets)
            copy_tensor(self.recurrence_wts_offsets_deploy, offsets.to(torch.int8))
        return quantized_recurrence_weight

    def _register_rnn_qat_params(self):
        """
        Register quantitative parameters for rnn
        """
        self.register_parameter('h_clip_max',
                                Parameter(tensor(self.retrain_data_config.get(CLIP_MAX, 1.0),
                                                 device=self.quant_device),
                                          requires_grad=True))
        self.register_parameter('h_clip_min',
                                Parameter(tensor(self.retrain_data_config.get(CLIP_MIN, -1.0),
                                                 device=self.quant_device),
                                          requires_grad=True))
        self.register_buffer('h_clip_max_pre',
                             tensor(np.nan, device=self.quant_device))
        self.register_buffer('h_clip_min_pre',
                             tensor(np.nan, device=self.quant_device))
        self.register_buffer('h_scale',
                             tensor([np.nan], device=self.quant_device))
        self.register_buffer('h_offset_deploy',
                             tensor([0], dtype=self.offset_dtype, device=self.quant_device))
        if self.retrain_weight_config.get('channel_wise', True):
            num_channel = self.out_channels
        else:
            num_channel = self.wts_group_num
        self.register_parameter('recurrence_wts_scales',
                                Parameter(tensor([np.nan] * num_channel,
                                                 device=self.quant_device)))
        self.register_parameter('recurrence_wts_offsets',
                                Parameter(tensor([np.nan] * num_channel,
                                                 device=self.quant_device)))
        self.register_buffer('recurrence_wts_offsets_deploy',
                             tensor([0] * num_channel, dtype=torch.int8,
                                    device=self.quant_device))
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
from torch import _C
import torch.nn as nn
from torch.nn.parameter import Parameter

from .....amct_pytorch.common.utils.check_params import check_params
from .....amct_pytorch.utils.log import LOGGER
from .....amct_pytorch.nn.module.quantization.rnn_qat_base import RnnQatBase
from .....amct_pytorch.custom_op.utils import copy_tensor

OPTIONAL_CONSTANT = 'prim::Constant'  # input that can be none in onnx
GRU_TENSOR_NUM = 3
RNN_INPUT_DIM = 3


class GRUQAT(nn.GRU, RnnQatBase):
    """
    Function: Quantization module class after GRU encapsulation.
    APIs: __init__, check_quantifiable, forward, from_float
    """
    _float_module = nn.GRU
    _required_params = ('input_size', 'hidden_size', 'num_layers', 'bias',
                        'batch_first', 'dropout', 'bidirectional')

    @check_params(input_size=int, hidden_size=int, num_layers=int, bias=bool, batch_first=bool,
                  dropout=(float, int), bidirectional=bool, config=(dict, type(None)))
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        device=None,
        dtype=None,
        config=None
    ) -> None:
        """Init GRUQAT amct op module"""
        nn.GRU.__init__(self, input_size, hidden_size, num_layers, bias=bias,
            batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        if dtype is not None and dtype != torch.float32:
            raise RuntimeError('GRUQAT only support dtype float32')
        self.out_channels = GRU_TENSOR_NUM * self.hidden_size
        if config is None:
            config = dict()

        RnnQatBase.__init__(self, 'GRU', device=device, config=config)
        self.to(device, dtype)

    @classmethod
    @check_params(mod=nn.Module, config=(dict, type(None)))
    def from_float(cls, mod, config=None):
        """
        Create a qat module from a float module
        Args: `mod` a float module, 'config' amct op quant config
        """
        if not isinstance(mod, cls._float_module):
            raise RuntimeError('{}.from_float can only works for {}'.format(
                cls.__name__, cls._float_module.__name__))
        if not config:
            config = dict()
        config['from_float'] = True
        qat_op = cls(
            mod.input_size,
            mod.hidden_size,
            mod.num_layers,
            bias=mod.bias,
            batch_first=mod.batch_first,
            dropout=mod.dropout,
            bidirectional=mod.bidirectional,
            config=config
        )
        RnnQatBase.set_rnn_op_trained_params(mod, qat_op, 'GRU')
        qat_op.to(mod.weight_ih_l0.device)
 
        LOGGER.logi('Convert {} to QAT op successfully.'.format(
            cls._float_module.__name__))
        return qat_op

    def forward(self, inputs, hx=None):
        """Defines the computation performed at every call."""
        self.check_input_data(inputs, hx)
        if not isinstance(hx, torch.Tensor):
            raise RuntimeError('GRUQAT hx must be Tensor, but your input is {}'.format(type(hx)))
        if len(hx.shape) != RNN_INPUT_DIM:
            raise RuntimeError("GRU quantize only support input hx dim 3,"\
                               " but your input dim is {}".format(len(hx.shape)))
        if self.batch_first:
            batch_size = int(inputs.shape[0])
        else:
            batch_size = int(inputs.shape[1])
 
        quantized_acts, quantized_h_x, quantized_wts, quantized_recurrence_weight = \
            self.forward_qat(inputs, hx)
        if self.bias_value is None:
            bias_ih_l0, bias_hh_l0 = None, None
        else:
            bias_ih_l0, bias_hh_l0 = self.bias_ih_l0, self.bias_hh_l0
        with torch.enable_grad():
            y, y_h = GRUMiddleWare.apply(quantized_acts,
                                         quantized_h_x,
                                         quantized_wts,
                                         quantized_recurrence_weight,
                                         self.bias_value,
                                         self.hidden_size,
                                         self.input_size,
                                         batch_size,
                                         self.batch_first,
                                         bias_ih_l0,
                                         bias_hh_l0)
        return y, y_h


class GRUMiddleWare(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, h, w, r, bias, hidden_size, input_size, batch_size,
                 batch_first, bias_ih_l0, bias_hh_l0):
        """ define the structure of exported onnx model for gruqat """
        if batch_first:
            x = g.op('Transpose', x, perm_i=list([1, 0, 2]))
        if not bias:
            bias = g.op(OPTIONAL_CONSTANT)
            bias.setType(_C.OptionalType.ofTensor())
        sequence_length = g.op(OPTIONAL_CONSTANT)
        sequence_length.setType(_C.OptionalType.ofTensor())
        gru = g.op('GRU', x, w, r, bias, sequence_length, h,
                   hidden_size_i=hidden_size, linear_before_reset_i=1, outputs=2)
        shape = g.op('Constant', value_t=torch.tensor([1, batch_size, hidden_size], dtype=torch.int64))
        out_node = g.op('Reshape', gru[0], shape)
        if batch_first:
            out_node = g.op('Transpose', out_node, perm_i=list([1, 0, 2]))
        return out_node, gru[1]

    @staticmethod
    def forward(ctx, x, h, w, r, bias, hidden_size, input_size, batch_size,
                batch_first, bias_ih_l0, bias_hh_l0):
        """ gru forward """
        mod = torch.nn.GRU(input_size, hidden_size, batch_first=batch_first, bias=bias_ih_l0 is not None)
        mod.weight_ih_l0 = Parameter(RnnQatBase.reorganize_tensor(w, 'GRU', back=True).to(device=x.device))
        mod.weight_hh_l0 = Parameter(RnnQatBase.reorganize_tensor(r, 'GRU', back=True).to(device=x.device))
        if bias_ih_l0 is not None:
            mod.bias_ih_l0 = Parameter(bias_ih_l0)
            mod.bias_hh_l0 = Parameter(bias_hh_l0)
        y, yh = mod(x, h)
        return y, yh

    @staticmethod
    def backward(ctx, grad_x, grad_h):
        """ backward funtion required by torch torch.autograd """
        ret = (None, ) * 11
        return ret
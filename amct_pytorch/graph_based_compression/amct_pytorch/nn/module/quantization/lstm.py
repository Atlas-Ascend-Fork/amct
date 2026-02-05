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
from collections.abc import Iterable
import torch
from torch import _C
import torch.nn as nn
from torch.nn.parameter import Parameter

from .....amct_pytorch.utils.log import LOGGER
from .....amct_pytorch.common.utils.check_params import check_params
from .....amct_pytorch.nn.module.quantization.rnn_qat_base import RnnQatBase
from .....amct_pytorch.custom_op.utils import copy_tensor
from .....amct_pytorch.common.utils.util import version_higher_than

OPTIONAL_CONSTANT = 'prim::Constant'  # input that can be none in onnx
LSTM_TENSOR_NUM = 4
RNN_INPUT_DIM = 3


class LSTMQAT(nn.LSTM, RnnQatBase):
    """
    Function: Quantization module class after LSTM encapsulation.
    APIs: __init__, check_quantifiable, forward, from_float
    """
    _float_module = nn.LSTM
    _required_params = ('input_size', 'hidden_size', 'num_layers', 'bias',
                        'batch_first', 'dropout', 'bidirectional', 'proj_size')

    @check_params(input_size=int, hidden_size=int, num_layers=int, bias=bool, batch_first=bool,
                  dropout=(float, int), bidirectional=bool, proj_size=int, config=(dict, type(None)))
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        proj_size=0,
        device=None,
        dtype=None,
        config=None
    ) -> None:
        """Init LSTMQAT amct op module"""
        kwargs = {'input_size': input_size,
                  'hidden_size': hidden_size,
                  'num_layers': num_layers,
                  'bias': bias,
                  'batch_first': batch_first,
                  'dropout': dropout,
                  'bidirectional': bidirectional}
        # proj_size only effective for torch higher than 1.8.0
        if version_higher_than(torch.__version__, '1.8.0'):
            kwargs['proj_size'] = proj_size
        nn.LSTM.__init__(self, **kwargs)
        if dtype is not None and dtype != torch.float32:
            raise RuntimeError('LSTMQAT only support dtype float32')
        self.out_channels = LSTM_TENSOR_NUM * self.hidden_size
        if config is None:
            config = dict()

        RnnQatBase.__init__(self, 'LSTM', device=device, config=config)
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
        proj_size = 0
        if version_higher_than(torch.__version__, '1.8.0'):
            proj_size = mod.proj_size
        if config is None:
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
            proj_size=proj_size,
            config=config
        )

        RnnQatBase.set_rnn_op_trained_params(mod, qat_op, 'LSTM')
        qat_op.to(mod.weight_ih_l0.device)
 
        LOGGER.logi('Convert {} to QAT op successfully.'.format(
            cls._float_module.__name__))
        return qat_op

    def check_quantifiable(self):
        """check qat config for LSTMQAT"""
        super().check_quantifiable()
        if hasattr(self, 'proj_size') and self.proj_size != 0:
            raise RuntimeError('Do not support LSTM with proj_size {}'.format(self.proj_size))
        return True

    def forward(self, inputs, hx=None):
        """Defines the computation performed at every call."""
        self.check_input_data(inputs, hx)
        if not isinstance(hx, Iterable) or \
            not isinstance(hx[0], torch.Tensor) or\
            not isinstance(hx[1], torch.Tensor):
            raise RuntimeError('LSTMQAT hx must be (Tensor, Tensor), but your input is {}'.format(type(hx)))
        if len(hx[0].shape) != RNN_INPUT_DIM:
            raise RuntimeError("LSTM quantize only support input hx dim 3,"\
                               " but your input dim is {}".format(len(hx[0].shape)))
        if self.batch_first:
            batch_size = int(inputs.shape[0])
        else:
            batch_size = int(inputs.shape[1])

        quantized_acts, quantized_h_x, quantized_wts, quantized_recurrence_weight = \
            self.forward_qat(inputs, hx[0])
        if self.bias_value is None:
            bias_ih_l0, bias_hh_l0 = None, None
        else:
            bias_ih_l0, bias_hh_l0 = self.bias_ih_l0, self.bias_hh_l0
        with torch.enable_grad():
            y, y_h, y_c = LSTMMiddleWare.apply(quantized_acts,
                                               quantized_h_x,
                                               hx[1],
                                               quantized_wts,
                                               quantized_recurrence_weight,
                                               self.bias_value,
                                               self.hidden_size,
                                               self.input_size,
                                               batch_size,
                                               self.batch_first,
                                               bias_ih_l0,
                                               bias_hh_l0)
        return y, (y_h, y_c)


class LSTMMiddleWare(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, h, c, w, r, bias, hidden_size, input_size, batch_size,
                 batch_first, bias_ih_l0, bias_hh_l0):
        """ define the structure of exported onnx model for lstmqat """
        if batch_first:
            x = g.op('Transpose', x, perm_i=[1, 0, 2])
        if not bias:
            bias = g.op(OPTIONAL_CONSTANT)
            bias.setType(_C.OptionalType.ofTensor())
        sequence_length = g.op(OPTIONAL_CONSTANT)
        sequence_length.setType(_C.OptionalType.ofTensor())
        p = g.op(OPTIONAL_CONSTANT)
        p.setType(_C.OptionalType.ofTensor())
        lstm = g.op('LSTM', x, w, r, bias, sequence_length, h, c, p, 
                    hidden_size_i=hidden_size, outputs=3)
        shape = g.op('Constant', value_t=torch.tensor([1, batch_size, hidden_size], dtype=torch.int64))
        out_node = g.op('Reshape', lstm[0], shape)
        if batch_first:
            out_node = g.op('Transpose', out_node, perm_i=[1, 0, 2])
        return out_node, lstm[1], lstm[2]

    @staticmethod
    def forward(ctx, x, h, c, w, r, bias, hidden_size, input_size, batch_size,
                batch_first, bias_ih_l0, bias_hh_l0):
        """ lstm forward """
        mod = torch.nn.LSTM(input_size, hidden_size, batch_first=batch_first, bias=bias_ih_l0 is not None)
        mod.weight_ih_l0 = Parameter(RnnQatBase.reorganize_tensor(w, 'LSTM', back=True).to(device=x.device))
        mod.weight_hh_l0 = Parameter(RnnQatBase.reorganize_tensor(r, 'LSTM', back=True).to(device=x.device))
        if bias_ih_l0 is not None:
            mod.bias_ih_l0 = Parameter(bias_ih_l0)
            mod.bias_hh_l0 = Parameter(bias_hh_l0)
        y, (yh, yc) = mod(x, (h, c))
        return y, yh, yc

    @staticmethod
    def backward(ctx, grad_x, grad_h, grad_c):
        """ backward funtion required by torch torch.autograd """
        ret = (None, ) * 12
        return ret
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

define onnx::BasicLSTMInplaceFillWindowCache op

"""
from .rnn_base import RnnQuantOpBase


class BasicLSTMInplaceFillWindowCache(RnnQuantOpBase):
    """
    Function: define onnx::BasicLSTMInplaceFillWindowCache op
    APIs: construct_node_proto
    """
    required_attrs = ['hidden_size']
    attrs = {
        'activation_alpha': ('FLOATS', []),
        'activation_beta': ('FLOATS', []),
        'activations': ('STRINGS', []),
        'clip': ('FLOAT', -1.0),
        'direction': ('STRING', bytes('forward', 'utf-8')),
        'input_forget': ('INT', 0)
    }

    ''' attr_name, (attr_type, default_value) '''
    quant_sqrt_mode_attrs = {
        'quant_sqrt_mode_x': ('INT', 0),
        'quant_sqrt_mode_h': ('INT', 0)
    }

    ''' name in record, (attr_name, attr_type) '''
    quant_attrs = {
        'data_scale': ('quant_scale_x', 'FLOAT'),
        'data_offset': ('quant_offset_x', 'FLOAT'),
        'h_scale': ('quant_scale_h', 'FLOAT'),
        'h_offset': ('quant_offset_h', 'FLOAT'),
        'act_type': ('quant_dtype', 'INT')
    }

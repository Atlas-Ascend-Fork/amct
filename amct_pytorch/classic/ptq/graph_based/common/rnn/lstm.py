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

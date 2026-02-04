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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from amct_pytorch.graph_based_compression.amct_pytorch.capacity import CAPACITY

INT4_BIT = 4
INT8_BIT = 8
INT16_BIT = 16
FP16_BIT = 16
FEATURE_LIST = ['approximate', 'ptq']
# ge::DataType value
DT_INT8 = 2
DT_INT16 = 6

INT4 = 'INT4'
INT8 = 'INT8'
INT16 = 'INT16'
DATA_OFFSET_RANGE = [-128, 127]
DATA_OFFSET_RANGE_INT16 = [-32768, 32767]

SUPPORT_ACT_ALGO = ('ifmr', 'hfmg')
DEFAULT_NUM_BITS = 8
KV_QUANT_SUPPORT_NUM_BITS = (8,)

# RNN constant
RNN_LAYER_TYPE = ('LSTM', 'GRU')
RNN_TENSOR_NUM = {'LSTM': 4, 'GRU': 3}
RNN_SEQ_LENS_INDEX = 4
RNN_H_INDEX = 5
LSTM_C_INDEX = 6
LSTM_P_INDEX = 7
LSTM_OUTPUT_NUMS = 3
GRU_OUTPUT_NUMS = 2
LSTM_ATTRS_LIMIT_MAP = {
    'activation_alpha': None,
    'activation_beta': None,
    'activations': None,
    'clip': None,
    'direction': 'forward',
    'input_forget': 0,
    'layout': 0
}
GRU_ATTRS_LIMIT_MAP = {
    'activation_alpha': None,
    'activation_beta': None,
    'activations': None,
    'direction': 'forward',
    'layout': 0
}
RNN_INPUT_ORDER_MAP = {
    'LSTM': {
        0: 0, # X
        1: 1, # W
        2: 2, # R
        3: 5, # B
        4: 6, # sequence_lens
        5: 3, # initial_h
        6: 4, # initial_c
    },
    'GRU': {
        0: 0, # X
        1: 1, # W
        2: 2, # R
        3: 4, # B
        4: 5, # sequence_lens
        5: 3, # initial_h
    }
}
RNN_DEQ_SCALE_INDEX = {
    'LSTM': 8,
    'GRU': 7
}

# ifmr default params
DEFAULT_MAX_PERCENTILE = 0.999999
DEFAULT_MIN_PERCENTILE = 0.999999
DEFAULT_SEARCH_RANGE_START = 0.7
DEFAULT_SEARCH_RANGE_END = 1.3
DEFAULT_SEARCH_STEP = 0.01
# hfmg default params
DEFUALT_NUM_OF_BINS = 4096
# quant_granularity
PER_TENSOR_IDX = 0
PER_CHANNEL_IDX = 1
WINOGRAD_NUM_BITS = (6, 7)

DEFAULT = 'DEFAULT'
FORCE_FP16_QUANT = 'FORCE_FP16_QUANT'

RETRAIN_DATA_TYPES = CAPACITY.get_value('RETRAIN_DATA_TYPES') \
    if CAPACITY.get_value('RETRAIN_DATA_TYPES') is not None else [INT8]
RETRAIN_ACT_WTS_TYPES = CAPACITY.get_value('RETRAIN_ACT_WTS_TYPES') \
    if CAPACITY.get_value('RETRAIN_ACT_WTS_TYPES') is not None else ['A8W8']

ACT_INDEX = 'act_index'
WTS_INDEX = 'weight_index'
BIAS_INDEX = 'bias_index'
INITIAL_H_INDEX = 'initial_h_index'
SEQUENCE_LENS_INDEX = 'sequence_lens_index'
RECURRENCE_WEIGHT_INDEX = 'recurrence_weight_index'
QUANT_INDEXES_MAP = {
    'Conv': {ACT_INDEX: 0,
             WTS_INDEX: 1,
             BIAS_INDEX: 2},
    'ConvTranspose': {ACT_INDEX: 0,
                      WTS_INDEX: 1,
                      BIAS_INDEX: 2},
    'Gemm': {ACT_INDEX: 0,
             WTS_INDEX: 1,
             BIAS_INDEX: 2},
    'MatMul': {ACT_INDEX: 0,
               WTS_INDEX: 1,
               BIAS_INDEX: None},
    'AveragePool': {ACT_INDEX: 0,
                    WTS_INDEX: None,
                    BIAS_INDEX: None},
    'LSTM': {ACT_INDEX: 0,
             WTS_INDEX: 1,
             BIAS_INDEX: 3,
             INITIAL_H_INDEX: 5,
             RECURRENCE_WEIGHT_INDEX: 2},
    'GRU': {ACT_INDEX: 0,
            WTS_INDEX: 1,
            BIAS_INDEX: 3,
            INITIAL_H_INDEX: 5,
            RECURRENCE_WEIGHT_INDEX: 2
    }
}
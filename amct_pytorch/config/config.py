# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
''' predefined quant config '''

INT4_AWQ_WEIGHT_QUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'int4',
            'symmetric': True,
            'strategy': 'channel'
        },
    },
    'algorithm': {'awq': {'grids_num': 20}},
    'skip_layers': {'lm_head'}

}


INT4_GPTQ_WEIGHT_QUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'int4',
            'symmetric': True,
            'strategy': 'channel',
        },
    },
    'algorithm': {'gptq'},
    'skip_layers': {'lm_head'}

}


INT4_FLAT_QUANT_CFG = {
    'batch_num': 4,
    'quant_cfg': {
        'inputs': {
            'enable_quant': True,
            'type': 'int4',
            'symmetric': True,
            'strategy': 'token'
        },
        'weights': {
            'type': 'int4',
            'symmetric': True,
            'strategy': 'channel',
        },
    },
    'algorithm': {
        'flatquant_attn': {
            'use_kcache_quant': False,
            'k_bits': 16,
            'use_vcache_quant': False,
            'v_bits': 16,
            'use_o_quant': False
        },
        'flatquant_attn_spda': {},
        'flatquant_mlp': {}
    },
    'skip_layers': {'lm_head'}
}


INT8_SMOOTHQUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'channel',
        },
        'inputs': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'smoothquant': {'smooth_strength': 0.5}},
    'skip_layers': {'lm_head'}
}


INT8_MINMAX_WEIGHT_QUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'channel',
        },
    },
    'algorithm': {'minmax'},
}
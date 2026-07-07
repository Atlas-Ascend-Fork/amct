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
'''Predefined low-bit quant configs for community task submission.

These two configs extend the existing preset registry in
amct_pytorch.common.config.config.  Once upstreamed, users can import them
directly from ``amct_pytorch``::

    from amct_pytorch import HIFP8_OFMR_CHANNEL_CFG, MXFP4_GPTQ_WEIGHT_QUANT_CFG

Until then, this module serves as a drop-in shim::

    from config import HIFP8_OFMR_CHANNEL_CFG, MXFP4_GPTQ_WEIGHT_QUANT_CFG
'''

HIFP8_OFMR_CHANNEL_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'channel',
        },
        'inputs': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'ofmr'},
    'skip_layers': {'lm_head'}
}


MXFP4_GPTQ_WEIGHT_QUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'mxfp4_e2m1',
            'symmetric': True,
            'strategy': 'group',
            'group_size': 32,
        },
    },
    'algorithm': {'gptq'},
    'skip_layers': {'lm_head'}
}

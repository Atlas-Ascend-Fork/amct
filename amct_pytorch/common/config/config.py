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
'''predefined quant config'''

INT4_AWQ_WEIGHT_QUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {'type': 'int4', 'symmetric': True, 'strategy': 'channel'},
    },
    'algorithm': {'awq': {'grids_num': 20}},
    'skip_layers': {'lm_head'},
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
    'skip_layers': {'lm_head'},
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
    'skip_layers': {'lm_head'},
}


# FlatQuant W4A8 (int8/int4): fake-quant accuracy evaluation only — the
# deploy operators only support INT4 activations, convert() rejects this
# combination. Use INT4 activations for the full deploy pipeline.
INT8_INT4_FLAT_QUANT_CFG = {
    'quant_cfg': {
        'inputs': {'type': 'int8', 'symmetric': True, 'strategy': 'token'},
        'weights': {'type': 'int4', 'symmetric': True, 'strategy': 'channel'},
    },
    'algorithm': {
        'flatquant': {
            'use_kcache_quant': False,
            'k_bits': 16,
            'k_sym': False,
            'use_vcache_quant': False,
            'v_bits': 16,
            'v_sym': False,
            'use_o_quant': False,
            'use_down_quant': True,
            'add_diag': True,
            'lac': False,
            'lwc': True,
            'diag_alpha': 0.3,
            'epochs': 15,
            'cali_bsz': 4,
            'flat_lr': 5e-3,
            'cali_trans': True,
        },
    },
    'skip_layers': {'lm_head'},
}


# FlatQuant W4A4 (int4/int4): full pipeline including deployment. Parameters
# follow the Qwen3 tuning used by examples/algorithms/flatquant.
INT4_INT4_FLAT_QUANT_CFG = {
    'quant_cfg': {
        'inputs': {'type': 'int4', 'symmetric': True, 'strategy': 'token'},
        'weights': {'type': 'int4', 'symmetric': True, 'strategy': 'channel'},
    },
    'algorithm': {
        'flatquant': {
            'use_kcache_quant': False,
            'k_bits': 16,
            'k_sym': False,
            'use_vcache_quant': False,
            'v_bits': 16,
            'v_sym': False,
            'use_o_quant': False,
            'use_down_quant': False,
            'add_diag': True,
            'lac': True,
            'lwc': True,
            'diag_alpha': 0.8,
            'epochs': 15,
            'cali_bsz': 4,
            'flat_lr': 3e-3,
            'cali_trans': True,
        },
    },
    'skip_layers': {'lm_head'},
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
    'skip_layers': {'lm_head'},
}


HIFP8_OFMR_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',
        },
        'inputs': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'ofmr'},
}


FP8_OFMR_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'float8_e4m3fn',
            'symmetric': True,
            'strategy': 'tensor',
        },
        'inputs': {
            'type': 'float8_e4m3fn',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'ofmr'},
    'skip_layers': {'lm_head'},
}


MXFP8_QUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'mxfp8_e4m3fn',
            'symmetric': True,
            'strategy': 'group',
            'group_size': 32,
        },
        'inputs': {
            'type': 'mxfp8_e4m3fn',
            'symmetric': True,
            'strategy': 'group',
        },
    },
    'algorithm': {'mxquant'},
    'skip_layers': {'lm_head'},
}


MXFP4_AWQ_WEIGHT_QUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'mxfp4_e2m1',
            'symmetric': True,
            'strategy': 'group',
            'group_size': 32,
        },
    },
    'algorithm': {'awq': {'grids_num': 20}},
    'skip_layers': {'lm_head'},
}

HIFP8_CAST_CFG = {
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
    'algorithm': {'cast'},
    'skip_layers': {'lm_head'},
}


HIFP8_QUANTILE_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',
        },
        'inputs': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'quantile'},
    'skip_layers': {'lm_head'},
}


HIFP8_KVCACHE_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'kvcache': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'quantile'},
}


HIFP8_SMOOTHQUANT_CFG = {
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
    'algorithm': {'smoothquant': {'smooth_strength': 0.5}},
    'skip_layers': {'lm_head'},
}

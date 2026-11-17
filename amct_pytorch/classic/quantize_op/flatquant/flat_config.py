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
#
# The code snippet comes from FlatQuant (https://github.com/ruikangliu/FlatQuant).
#
# Copyright (c) 2024 ruikangliu. All rights reserved.
#
# This file is part of FlatQuant, which is released under the MIT License.
# See the LICENSE file in the root directory of this source tree
# or at https://opensource.org/licenses/MIT for details.
#
# --------------------------------------------------------------------------------
"""FlatQuant single-module config, parsed once per replaced module."""


class FlatConfig:
    """FlatQuant per-module config, class attributes are fallback defaults."""

    # activation
    a_bits = 8
    a_sym = True

    # weight
    w_bits = 4
    w_sym = True
    # weight granularity: 'channel' (per-channel scale) or 'tensor'
    # (single scale); resolved from weights_cfg.strategy at parse time.
    w_granularity = 'channel'

    # KV-cache quantization (calibration side only)
    use_kcache_quant = False
    use_vcache_quant = False
    k_bits = 16
    k_sym = False
    v_bits = 16
    v_sym = False

    # special controls for o_proj & down_proj
    use_o_quant = False
    use_down_quant = True

    # flatquant algorithm knobs
    add_diag = True
    lwc = True
    lac = False
    diag_alpha = 0.3

    # calibration
    epochs = 15
    cali_bsz = 4
    flat_lr = 5e-3
    cali_trans = True

    # activation-quant carrier dtype ('fp32' default; 'fp16' mirrors the
    # npu_kronecker_quant fp16 rounding for deploy-faithful calibration)
    act_dtype = 'fp32'


def flat_config_parser(layer_config):
    """Parse one layer's detail_config entry into a FlatConfig instance.

    Bit widths become ints here; downstream reads FlatConfig only.
    """
    # README documents the set style {'flatquant'} — a set literal parses as
    # {'flatquant': None}; normalize both that and a None value to {} so the
    # set style behaves exactly like {'flatquant': {}}.
    algorithm_cfg = layer_config.get('algorithm') or {}
    if not isinstance(algorithm_cfg, dict):
        algorithm_cfg = dict.fromkeys(algorithm_cfg)
    flatquant_cfg = algorithm_cfg.get('flatquant') or {}

    cfg = FlatConfig()
    # quant params
    cfg.a_bits = int(
        layer_config.get('inputs_cfg').get('quant_type').replace('int', '')
    )
    cfg.w_bits = int(
        layer_config.get('weights_cfg').get('quant_type').replace('int', '')
    )
    cfg.a_sym = layer_config.get('inputs_cfg').get('symmetric')
    cfg.w_sym = layer_config.get('weights_cfg').get('symmetric')
    w_strategy = layer_config.get('weights_cfg').get('strategy')
    if w_strategy in ('channel', 'tensor'):
        cfg.w_granularity = w_strategy
    elif w_strategy is not None:
        raise ValueError(
            f"flatquant weights strategy must be 'channel' or 'tensor', "
            f'got {w_strategy!r}'
        )
    # algorithm knobs, class defaults as fallback
    cfg.lac = flatquant_cfg.get('lac', FlatConfig.lac)
    cfg.lwc = flatquant_cfg.get('lwc', FlatConfig.lwc)
    cfg.use_o_quant = flatquant_cfg.get('use_o_quant', FlatConfig.use_o_quant)
    cfg.use_down_quant = flatquant_cfg.get('use_down_quant', FlatConfig.use_down_quant)
    # kv cache
    cfg.use_kcache_quant = flatquant_cfg.get(
        'use_kcache_quant', FlatConfig.use_kcache_quant
    )
    cfg.k_bits = flatquant_cfg.get('k_bits', FlatConfig.k_bits)
    cfg.k_sym = flatquant_cfg.get('k_sym', FlatConfig.k_sym)
    cfg.use_vcache_quant = flatquant_cfg.get(
        'use_vcache_quant', FlatConfig.use_vcache_quant
    )
    cfg.v_bits = flatquant_cfg.get('v_bits', FlatConfig.v_bits)
    cfg.v_sym = flatquant_cfg.get('v_sym', FlatConfig.v_sym)
    # calibration
    cfg.add_diag = flatquant_cfg.get('add_diag', FlatConfig.add_diag)
    cfg.epochs = flatquant_cfg.get('epochs', FlatConfig.epochs)
    cfg.cali_bsz = flatquant_cfg.get('cali_bsz', FlatConfig.cali_bsz)
    cfg.flat_lr = flatquant_cfg.get('flat_lr', FlatConfig.flat_lr)
    cfg.cali_trans = flatquant_cfg.get('cali_trans', FlatConfig.cali_trans)
    cfg.act_dtype = flatquant_cfg.get('act_dtype', FlatConfig.act_dtype)
    cfg.diag_alpha = flatquant_cfg.get('diag_alpha', FlatConfig.diag_alpha)
    return cfg

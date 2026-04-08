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

from amct_pytorch.utils.log import LOGGER
from amct_pytorch.config.fields import QuantConfig, WeightsCfgField, InputsCfgField
from amct_pytorch.config.config import INT8_MINMAX_WEIGHT_QUANT_CFG
from amct_pytorch.utils.vars import ALGORITHM_SUPPORTED_QUANT_TYPE_COMB, ALLOWED_WEIGHT_DTYPES
from amct_pytorch.utils.vars import WTS_GRANULARITY_SUPPORT_MAP, ACT_GRANULARITY_SUPPORT_MAP
from amct_pytorch.algorithm import BUILT_IN_ALGORITHM


def set_default_config():
    '''default quant config'''
    LOGGER.logd('Using INT8_MINMAX_WEIGHT_QUANT_CFG as default config')
    return INT8_MINMAX_WEIGHT_QUANT_CFG


def check_skip_layer(layer_name, skip_layers):
    """
    Check whether the current layer is a effective layers
    Params:
    skip_layers: list, Contains layers that need to skip quantization
    layer_name: string, Specifies whether to skip the quantized layer name
    Return: bool, Indicates whether to skip quantization at the current layer
    """
    # If Skip_layers is not defined, the default value is None.
    if not skip_layers:
        return False

    for skip_layer in skip_layers:
        if skip_layer in layer_name:
            LOGGER.logd('The {} quantifiable module is skipped'.format(layer_name))
            return True

    return False


def check_config(quant_data_comb, quant_config, algo):
    """
    Check whether the config is valid 
    Params:
    quant_data_comb: quantize data type combination str
    quant_config: quant parameters
    algo: quantization algorithm
    Return: bool, Indicates whether config is supported to quantization
    """
    if quant_data_comb not in ALGORITHM_SUPPORTED_QUANT_TYPE_COMB.keys():
        raise ValueError(f'Do not support combination {quant_data_comb} of act and weight quant dtype.')
    if algo not in ALGORITHM_SUPPORTED_QUANT_TYPE_COMB[quant_data_comb]:
        raise ValueError(f'Algorithm {algo} do not support act and weight quant dtype {quant_data_comb}')

    weight_strategy = quant_config.quant_cfg.weights_cfg.strategy
    if quant_data_comb not in WTS_GRANULARITY_SUPPORT_MAP.get(weight_strategy):
        raise ValueError(f'act_dtype and wts_dtype {quant_data_comb} do not support weight '
            f'quant strategy {weight_strategy}')

    if quant_config.quant_cfg.inputs_cfg.quant_input:
        act_strategy = quant_config.quant_cfg.inputs_cfg.strategy
        act_dtype = quant_config.quant_cfg.inputs_cfg.quant_type
        if act_dtype not in ['mxfp8_e4m3fn'] and quant_data_comb not in ACT_GRANULARITY_SUPPORT_MAP.get(act_strategy):
            raise ValueError(f'act_dtype and wts_dtype {quant_data_comb} do not support '
                f'activation quant strategy {act_strategy}')
        if act_dtype in ['mxfp8_e4m3fn'] and act_strategy != 'group':
            raise ValueError(f'act_dtype and wts_dtype {quant_data_comb} only support activation quant strategy group')
    
    if quant_config.quant_cfg.weights_cfg.group_size is not None:
        group_size = quant_config.quant_cfg.weights_cfg.group_size
        if group_size < 32 or group_size % 32 != 0:
            raise ValueError(f'act_type and wts_type {quant_data_comb} only support group_size larger than 32 and '
                f'integer multiple of 32, current is {group_size}')


def _check_fuzzy_config_warnings(all_layer_names, quant_config):
    """
    Check for warnings related to fuzzy matching configuration
    Params:
        all_layer_names: list, all layer names in the model
        quant_config: QuantConfig object
    """
    from amct_pytorch.config.utils import match_fuzzy_pattern
    
    fuzzy_weights = quant_config.quant_cfg.fuzzy_configs['weights']
    fuzzy_inputs = quant_config.quant_cfg.fuzzy_configs['inputs']
    skip_layers = quant_config.skip_layers.skip_layers
    
    if not fuzzy_weights and not fuzzy_inputs:
        return
    
    if not skip_layers:
        skip_layers_set = set()
    else:
        skip_layers_set = set(skip_layers)
    
    all_fuzzy_patterns = fuzzy_weights + fuzzy_inputs
    all_fuzzy_pattern_strs = [cfg['pattern'] for cfg in all_fuzzy_patterns]
    
    matched_fuzzy_layers = set()
    
    for fuzzy_cfg in all_fuzzy_patterns:
        pattern = fuzzy_cfg['pattern']
        for layer_name in all_layer_names:
            if match_fuzzy_pattern(layer_name, pattern):
                matched_fuzzy_layers.add(layer_name)
                if layer_name in skip_layers_set:
                    LOGGER.logw(f'Fuzzy matched layer {layer_name} is also in skip_layers, '
                               f'quantization will be skipped', 'ConfigParser')
    
    for pattern in all_fuzzy_pattern_strs:
        has_match = False
        for layer_name in all_layer_names:
            if match_fuzzy_pattern(layer_name, pattern):
                has_match = True
                break
        if not has_match:
            LOGGER.logw(f'Fuzzy pattern {pattern} does not match any layer in the model', 'ConfigParser')


def check_quant_op_constraint(mod, layer_name, quant_data_comb, quant_config):
    """
    Check whether the current layer fits requirements of deploy op 
    Params:
    layer_name: string, Specifies whether to skip the quantized layer name
    quant_data_comb: quantize data type combination str
    quant_config: quant parameters
    Return: bool, Indicates whether the current layer is supported to quantization
    """
    mod_type = type(mod).__name__
    if mod_type != 'Linear':
        return True

    # npu op check cin length be integer multiple of 64
    support_quant_dtype_comb = ['float8_e4m3fn float4_e2m1']
    if quant_data_comb in support_quant_dtype_comb and mod.weight.shape[1] % 64 != 0:
        LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} has shape requirement '
                    'cin length should be integer multiple of 64'.format(layer_name, quant_data_comb))
        return False
    # npu op check no bias
    if quant_data_comb in support_quant_dtype_comb and mod.bias is not None:
        LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} has shape requirement '
            'bias is not supported'.format(layer_name, quant_data_comb))
        return False

    # npu op check cin length ceildiv(cin, 32) must be even number
    check_mxfp8_mxfp8_dtype = 'mxfp8_e4m3fn mxfp8_e4m3fn'
    if quant_data_comb == 'mxfp8_e4m3fn mxfp8_e4m3fn' and ((mod.weight.shape[1] + 31) // 32) % 2 != 0:
        LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} has shape requirement '
            'cin length ceildiv(cin, 32) must be even number'.format(layer_name, quant_data_comb))
        return False

    # npu op check cin length be integer multiple of 32b
    if quant_data_comb in ['NOT_QUANTIZE mxfp4_e2m1', 'NOT_QUANTIZE float4_e2m1']:
        if mod.weight.shape[1] % 64 != 0 or mod.weight.shape[0] % 64 != 0:
            LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} has shape requirement'
                'cin and cout length should be integer multiple of 64'.format(layer_name, quant_data_comb))
            return False

    if quant_data_comb == 'NOT_QUANTIZE int4':
        if mod.weight.shape[0] % 8 != 0 or mod.weight.shape[1] % 8 != 0:
            LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} has shape requirement'
                ' cin and cout length should be integer multiple of 8'.format(layer_name, quant_data_comb))
            return False

    group_size = quant_config.quant_cfg.weights_cfg.group_size
    if group_size is None:
        return True

    if group_size >= mod.weight.shape[1]:
        LOGGER.logd(f'group size should less than cout, current group_size is {group_size}, '
            f'cout is {mod.weight.shape[1]}, skip quantization of current layer {layer_name}')
        return False 
    return True


def get_supported_layers(model, quant_config, registed_alg):
    '''get supported layers based on quant config and registed algorithm'''
    algos = quant_config.algorithm.names
    layer_types = dict()
    quant_type_comb = None
    for algo in algos:
        src_ops = registed_alg.algo.get(algo).keys()
        for src_op in src_ops:
            layer_types[src_op] = algo
        if algo not in BUILT_IN_ALGORITHM:
            LOGGER.logd("Customized Algorithm {} is used fot quant".format(algo))
        else:
            act_type = quant_config.quant_cfg.inputs_cfg.quant_type if \
                        quant_config.quant_cfg.inputs_cfg.quant_input else 'NOT_QUANTIZE'
            wts_type = quant_config.quant_cfg.weights_cfg.quant_type
            quant_type_comb = act_type + ' ' + wts_type
            check_config(quant_type_comb, quant_config, algo)

    detail_config = dict()
    all_layer_names = [name for name, mod in model.named_modules()
                       if type(mod) in layer_types.keys() or type(mod).__name__ in layer_types.keys()]
    
    _check_fuzzy_config_warnings(all_layer_names, quant_config)

    for name, mod in model.named_modules():
        if type(mod) not in layer_types.keys() and type(mod).__name__ not in layer_types.keys():
            continue
        if type(mod).__name__ == 'Conv2d' and mod.padding_mode != 'zeros':
            LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} has requirement '
                        'on padding_mode of conv2d, padding_mode should be zero.'.format(name, quant_type_comb))
            continue
        if check_skip_layer(name, quant_config.skip_layers.skip_layers):
            LOGGER.logd('layer:{} is skipped'.format(name))
            continue
        algo = layer_types[type(mod).__name__]
        if algo in BUILT_IN_ALGORITHM:
            if quant_type_comb is not None and hasattr(mod, 'weight') and \
               mod.weight.dtype not in ALLOWED_WEIGHT_DTYPES.get(quant_type_comb):
                LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} only support ori dtype {} '
                'but got {}'.format(name, quant_type_comb, ALLOWED_WEIGHT_DTYPES.get(quant_type_comb),
                mod.weight.dtype))
                continue
            if quant_type_comb is not None and not check_quant_op_constraint(mod, name, quant_type_comb, quant_config):
                continue
        
        layer_quant_cfg = _get_layer_quant_config(name, quant_config)
        if layer_quant_cfg is None:
            continue
        
        detail_config_algo = {'algorithm': {algo: quant_config.algorithm.get_value()['algorithm'][algo]}} 
        detail_config[name] = quant_config.batch_num.get_value() \
            | layer_quant_cfg | detail_config_algo
    return detail_config


def _get_layer_quant_config(layer_name, quant_config):
    """
    Get quant config for a specific layer, considering fuzzy matching
    Params:
        layer_name: str, the layer name
        quant_config: QuantConfig object
    Return: dict, the quant config for this layer
    """
    quant_cfg = quant_config.quant_cfg.get_value().copy()
    
    fuzzy_weights_cfg = quant_config.quant_cfg.get_fuzzy_config(layer_name, 'weights')
    if fuzzy_weights_cfg:
        weights_cfg = WeightsCfgField(fuzzy_weights_cfg)
        quant_cfg['weights_cfg'] = weights_cfg.get_value()
    elif quant_cfg['weights_cfg'] is None:
        return None
    
    fuzzy_inputs_cfg = quant_config.quant_cfg.get_fuzzy_config(layer_name, 'inputs')
    if fuzzy_inputs_cfg:
        inputs_cfg = InputsCfgField(fuzzy_inputs_cfg)
        quant_cfg['inputs_cfg'] = inputs_cfg.get_value()
    elif quant_cfg['inputs_cfg'] is None:
        quant_cfg['inputs_cfg'] = {'enable_quant': False}
    
    return quant_cfg


def parse_config(model, config, registed_alg):
    '''parse quant config, get quant config for each layer'''
    quant_config = QuantConfig(config, registed_alg)
    detail_config = get_supported_layers(model, quant_config, registed_alg)

    return detail_config
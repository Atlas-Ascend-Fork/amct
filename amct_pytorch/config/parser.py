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
from amct_pytorch.config.fields import QuantConfig
from amct_pytorch.config.config import INT8_MINMAX_WEIGHT_QUANT_CFG
from amct_pytorch.utils.vars import SUPPORTED_QUANT_DTYPE_COMB
from amct_pytorch.utils.vars import ALGORITHM_SUPPORTED_QUANT_TYPE_COMB, ALLOWED_WEIGHT_DTYPES
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
    if quant_data_comb not in SUPPORTED_QUANT_DTYPE_COMB:
        raise ValueError(f'Do not support combination {quant_data_comb} of act and weight quant dtype.')
    if algo not in ALGORITHM_SUPPORTED_QUANT_TYPE_COMB[quant_data_comb]:
        raise ValueError(f'Algorithm {algo} do not support act and weight quant dtype {quant_data_comb}')

    if quant_data_comb in ['int8 int8']:
        if quant_config.quant_cfg.weights_cfg.strategy == 'group':
            raise ValueError(f'act_dtype and wts dtype {quant_data_comb} do not support weight quant strategy group')
    
    if quant_config.quant_cfg.weights_cfg.group_size is not None:
        group_size = quant_config.quant_cfg.weights_cfg.group_size
        if group_size < 32 or group_size % 32 != 0:
            raise ValueError(f'act_type and wts_type {quant_data_comb} only support group_size larger than 32 and '
                f'integer multiple of 32, current is {group_size}')


def check_quant_op_constraint(mod, layer_name, quant_data_comb, quant_config):
    """
    Check whether the current layer fits requirements of deploy op 
    Params:
    layer_name: string, Specifies whether to skip the quantized layer name
    quant_data_comb: quantize data type combination str
    quant_config: quant parameters
    Return: bool, Indicates whether the current layer is supported to quantization
    """
    if quant_data_comb not in ['NOT_QUANTIZE int4', 'NOT_QUANTIZE int8']:
        return True
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
    for algo in algos:
        layer_types[registed_alg.algo.get(algo)[0]] = algo

        if algo not in BUILT_IN_ALGORITHM:
            LOGGER.logd("Customized Algorithm {} is used fot quant".format(algo))
        else:
            act_type = quant_config.quant_cfg.inputs_cfg.quant_type if \
                        quant_config.quant_cfg.inputs_cfg.quant_input else 'NOT_QUANTIZE'
            wts_type = quant_config.quant_cfg.weights_cfg.quant_type
            quant_type_comb = act_type + ' ' + wts_type
            check_config(quant_type_comb, quant_config, algo)

    detail_config = dict()
    for name, mod in model.named_modules():
        if type(mod) not in layer_types and type(mod).__name__ not in layer_types:
            continue
        if check_skip_layer(name, quant_config.skip_layers.skip_layers):
            LOGGER.logd('layer:{} is skipped'.format(name))
            continue
        algo = layer_types[type(mod).__name__]
        if algo in BUILT_IN_ALGORITHM:
            if hasattr(mod, 'weight') and mod.weight.dtype not in ALLOWED_WEIGHT_DTYPES.get(quant_type_comb):
                LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} only support ori dtype {} '
                'but got {}'.format(name, quant_type_comb, ALLOWED_WEIGHT_DTYPES.get(quant_type_comb),
                mod.weight.dtype))
                continue
            if not check_quant_op_constraint(mod, name, quant_type_comb, quant_config):
                continue
        # each module can only be assigned one algorithm
        detail_config_algo = {'algorithm': {algo: quant_config.algorithm.get_value()['algorithm'][algo]}} 
        detail_config[name] = quant_config.batch_num.get_value() \
            | quant_config.quant_cfg.get_value() | detail_config_algo
    return detail_config


def parse_config(model, config, registed_alg):
    '''parse quant config, get quant config for each layer'''
    quant_config = QuantConfig(config, registed_alg)
    detail_config = get_supported_layers(model, quant_config, registed_alg)

    return detail_config
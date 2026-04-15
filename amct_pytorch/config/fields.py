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

from amct_pytorch.utils.vars import SUPPORT_WEIGHT_QUANT_DTYPE, SUPPORT_INPUT_QUANT_DTYPE
from amct_pytorch.utils.vars import SUPPORT_QUANT_STRATEGY_WEIGHT, SUPPORT_QUANT_STRATEGY_INPUT
from amct_pytorch.utils.vars import SUPPORT_QUANT_DYNAMIC_INPUT
from amct_pytorch.utils.vars import WTS_ASYMMETRIC_DTYPE, GROUP_SIZE_SUPPORTED_DTYPE
from amct_pytorch.utils.vars import GROUP_SIZE_SUPPORTED_MAP
from amct_pytorch.config.utils import get_alg_name_from_config
from amct_pytorch.algorithm import AlgorithmRegistry


class BatchNumField():
    def __init__(self, batch_num):
        self.batch_num = batch_num
        self.check()
        self.value = self.set_value()
    
    def set_value(self):
        return {'batch_num': self.batch_num}
    
    def get_value(self):
        return self.value

    def check(self):
        if not isinstance(self.batch_num, int) or self.batch_num <= 0:
            raise ValueError(f'Batch num only support positive int, but got {self.batch_num}')


class WeightsCfgField():
    def __init__(self, config):
        if config:
            self.quant_type = config.get('type')
            self.symmetric = config.get('symmetric')
            self.strategy = config.get('strategy')
            self.group_size = config.get('group_size', None)
            self.check()
        else:
            self.quant_type = None
            self.symmetric = None
            self.strategy = None
            self.group_size = None
        self.value = self.set_value()
    
    def set_value(self):
        if self.quant_type is None:
            return None
        if self.group_size is None:
            return {'quant_type': self.quant_type,
                    'symmetric': self.symmetric,
                    'strategy': self.strategy}
        return {'quant_type': self.quant_type,
                'symmetric': self.symmetric,
                'strategy': self.strategy,
                'group_size': self.group_size}
    
    def get_value(self):
        return self.value

    def check(self):
        if self.quant_type not in SUPPORT_WEIGHT_QUANT_DTYPE:
            raise ValueError(f'Weights quant_dtype only support {SUPPORT_WEIGHT_QUANT_DTYPE}, '
                f'but got {self.quant_type}')
        if self.symmetric not in [True, False]:
            raise ValueError(f'Weights symmetric only support bool [True, False], but got {self.symmetric}')
        if self.symmetric == False and self.quant_type not in WTS_ASYMMETRIC_DTYPE:
            raise ValueError(f'Weights symmetric only support to be True when '
                f'weight quant_type is {self.quant_type}')
        if self.strategy not in SUPPORT_QUANT_STRATEGY_WEIGHT:
            raise ValueError(f'Weights strategy only support{SUPPORT_QUANT_STRATEGY_WEIGHT}, but got {self.strategy}')
        if self.group_size is not None and self.strategy != 'group':
            raise ValueError(f'Weights group_size only support strategy group, but got {self.strategy}')
        if self.group_size is None and self.strategy == 'group':
            raise ValueError(f'Weights group_size is necessary, when weights strategy is group')
        if self.group_size is not None:
            self.check_group_size()

    def check_group_size(self):
        if self.strategy != 'group':
            raise ValueError(f'Weights group_size only support strategy group, but got {self.strategy}')
        if not isinstance(self.group_size, int) or self.group_size <= 0:
            raise ValueError(f'group_size only support positive int, but got {self.group_size}')
        if self.quant_type not in GROUP_SIZE_SUPPORTED_DTYPE:
            raise ValueError(f'Weights group_size only support for '
                f'weight dtype: {GROUP_SIZE_SUPPORTED_DTYPE}')
        if self.quant_type in GROUP_SIZE_SUPPORTED_MAP.keys() and self.group_size not in \
            GROUP_SIZE_SUPPORTED_MAP.get(self.quant_type):
            raise ValueError(f'wts_type {self.quant_type} only support group_size value: '
                            f'{GROUP_SIZE_SUPPORTED_MAP.get(self.quant_type)}')


class InputsCfgField():
    def __init__(self, config):
        self.quant_input = config.get('enable_quant', True)
        if self.quant_input:
            self.quant_type = config.get('type')
            self.symmetric = config.get('symmetric')
            self.strategy = config.get('strategy')
            self.dynamic = config.get('dynamic', None)
            if self.quant_type is not None:
                self.check()
        self.value = self.set_value()
    
    def set_value(self):
        if not self.quant_input:
            return {'enable_quant': False}
        if self.quant_type is None:
            return None
        return {'quant_type': self.quant_type,
                'symmetric': self.symmetric,
                'strategy': self.strategy,
                'dynamic': self.dynamic}
    
    def get_value(self):
        return self.value

    def check(self):
        if self.quant_type not in SUPPORT_INPUT_QUANT_DTYPE:
            raise ValueError(f'Inputs quant_dtype only support {SUPPORT_INPUT_QUANT_DTYPE}, but got {self.quant_type}')
        if self.symmetric not in [True, False]:
            raise ValueError(f'Inputs symmetric only support bool [True, False], but got {self.symmetric}')
        if self.symmetric == False and self.quant_type not in ['int8']:
            raise ValueError(f'Inputs symmetric is unsupported to be False when Inputs quant_type is {self.quant_type}')
        if self.strategy not in SUPPORT_QUANT_STRATEGY_INPUT and self.quant_type not in ['mxfp8_e4m3fn']:
            raise ValueError(f'Inputs strategy only support{SUPPORT_QUANT_STRATEGY_INPUT}, but got {self.strategy}')
        if self.strategy == 'token' and self.symmetric == False:
            raise ValueError(f'Inputs strategy token do not support asymmetric quantization')
        if self.dynamic is not None and self.strategy not in SUPPORT_QUANT_DYNAMIC_INPUT:
            raise ValueError(f'Inputs dynamic only support strategy {SUPPORT_QUANT_DYNAMIC_INPUT}, '
                f'but got {self.strategy}')


class QuantCfgField():
    def __init__(self, config):
        self.weights_cfg = WeightsCfgField(config.get('weights', {}))
        self.inputs_cfg = InputsCfgField(config.get('inputs', {'enable_quant': False}))
        self.fuzzy_configs = QuantCfgField.extract_fuzzy_configs(config)
        
        self._validate_weights_config()
        
        self.value = self.set_value()

    @staticmethod
    def extract_fuzzy_configs(config):
        """
        Extract fuzzy matching configs from quant_cfg
        Patterns like '*down_proj.weights' or '*self_attn.q_proj.inputs'
        """
        fuzzy_configs = {'weights': [], 'inputs': []}
        
        for key, value in config.items():
            if '*' in key:
                if key.endswith('.weights'):
                    fuzzy_configs['weights'].append({
                        'pattern': key,
                        'config': value
                    })
                elif key.endswith('.inputs'):
                    fuzzy_configs['inputs'].append({
                        'pattern': key,
                        'config': value
                    })
        
        return fuzzy_configs

    def set_value(self):
        return {'weights_cfg': self.weights_cfg.get_value(),
                'inputs_cfg': self.inputs_cfg.get_value()}

    def get_value(self):
        return self.value

    def get_fuzzy_config(self, layer_name, config_type):
        """
        Get matching fuzzy config for a layer
        Params:
            layer_name: str, the layer name
            config_type: str, 'weights' or 'inputs'
        Return: dict or None, the matching config or None
        """
        from amct_pytorch.config.utils import match_fuzzy_pattern
        
        for fuzzy_cfg in self.fuzzy_configs[config_type]:
            if match_fuzzy_pattern(layer_name, fuzzy_cfg['pattern']):
                return fuzzy_cfg['config']
        return None

    def _validate_weights_config(self):
        """
        Validate that at least one weights configuration exists
        Either a precise weights configuration or a fuzzy matching *.weights configuration
        """
        has_weights_config = self.weights_cfg.get_value() is not None
        has_fuzzy_weights_config = len(self.fuzzy_configs['weights']) > 0
        
        if not has_weights_config and not has_fuzzy_weights_config:
            raise ValueError(
                'Configuration must include at least one weights configuration: either '
                'a "weights" field or a fuzzy matching pattern like "*.weights"'
            )


class AwqField():
    def __init__(self, attrs):
        self.attrs = attrs
        self.grids_num = None
        self.check()
        self.value = self.set_value()
    
    def set_value(self):
        return {'awq': {'grids_num': self.grids_num}} 
    
    def get_value(self):
        return self.value

    def check(self):
        if self.attrs is None or self.attrs.get('grids_num') is None:
            raise ValueError(f'Awq grids_num is necessary param, pls check and set')
        self.grids_num = self.attrs.get('grids_num')
        if not isinstance(self.grids_num, int) or self.grids_num <= 0:
            raise ValueError(f'Awq grids_num only support positive int, but got {self.grids_num}')


class GptqField():
    def __init__(self, attrs):
        self.value = self.set_value()
    
    @staticmethod
    def set_value():
        return {'gptq'} 
    
    def get_value(self):
        return self.value


class SmoothQuantField():
    def __init__(self, attrs):
        self.attrs = attrs
        self.smooth_strength = None
        self.check()
        self.value = self.set_value()
    
    def set_value(self):
        return {'smoothquant': {'smooth_strength': self.smooth_strength}} 
    
    def get_value(self):
        return self.value

    def check(self):
        if self.attrs is None or self.attrs.get('smooth_strength') is None:
            raise ValueError(f'Smoothquant smooth_strength is necessary param, pls check and set')
        self.smooth_strength = self.attrs.get('smooth_strength')
        if not isinstance(self.smooth_strength, float) or self.smooth_strength <= 0 or self.smooth_strength >= 1:
            raise ValueError(f'Smoothquant smooth_strength only support float (0, 1), but got {self.smooth_strength}')


class MinmaxField():
    def __init__(self, attrs):
        self.value = self.set_value()
    
    @staticmethod
    def set_value():
        return {'minmax'} 
    
    def get_value(self):
        return self.value


class MxQuant():
    def __init__(self, attrs):
        self.value = self.set_value()
    
    @staticmethod
    def set_value():
        return {'mxquant'} 
    
    def get_value(self):
        return self.value


class CastField():
    def __init__(self, attrs):
        self.value = self.set_value()
    
    @staticmethod
    def set_value():
        return {'cast'} 
    
    def get_value(self):
        return self.value


class CustomAlgField():
    def __init__(self, name, attrs):
        self.value = {name: attrs}

    def get_value(self):
        return self.value


class AlgorithmField():
    def __init__(self, alg_cfg, alg_reg):
        alg_names, alg_attrs = get_alg_name_from_config(alg_cfg)
        assert len(alg_names) == len(alg_attrs), "alg_names and alg_attrs should have the same length"
        for alg_name in alg_names:
            if alg_reg.algo.get(alg_name) is None:
                raise ValueError(f'Not support algorithm {alg_name}, pls regiter it first')

        registed_algo_field = {
            'awq': AwqField,
            'gptq': GptqField,
            'smoothquant': SmoothQuantField,
            'minmax': MinmaxField,
            'mxquant': MxQuant,
            'cast': CastField,
            'custom': CustomAlgField
        }

        self.algs = []
        supported_src_op = []
        for alg_name, alg_attr in zip(alg_names, alg_attrs):
            for src_op in AlgorithmRegistry.algo.get(alg_name).keys():
                if src_op in supported_src_op:
                    raise ValueError(f'One src_op only support one algorithm, current algo {alg_name} '
                        f'support src_op {src_op} is duplicate')
                    supported_src_op.append(src_op)
            if registed_algo_field.get(alg_name):
                self.algs.append(registed_algo_field[alg_name](alg_attr))
            else:
                self.algs.append(registed_algo_field['custom'](alg_name, alg_attr))

        self.names = alg_names
        self.value = self.set_value()

    @staticmethod
    def check(alg_cfg):
        if not isinstance(alg_cfg, str) and len(alg_cfg) != 1:
            raise ValueError(f'Algorithm only support 1 str, but got {alg_cfg}')

    def get_value(self):
        return self.value

    def set_value(self):
        total_alg = {}
        for alg in self.algs:
            alg_value = alg.get_value()
            assert len(alg_value) == 1
            if isinstance(alg_value, set):
                total_alg |= {next(iter(alg_value)): {}}
            else:
                total_alg |= alg_value
        return {'algorithm': total_alg}


class SkipLayersField():
    def __init__(self, layers):
        self.skip_layers = layers
        self.check()
        self.value = self.set_value()
    
    def set_value(self):
        return {'skip_layers': self.skip_layers}
    
    def get_value(self):
        return self.value

    def check(self):
        for layer in self.skip_layers:
            if not isinstance(layer, str):
                raise ValueError(f'Skip layers must be str')



class QuantConfig:
    def __init__(self, config, registed_alg):
        self.batch_num = BatchNumField(config.get('batch_num', 1))
        self.quant_cfg = QuantCfgField(config.get('quant_cfg', {}))
        self.algorithm = AlgorithmField(config.get('algorithm', {}), registed_alg)
        self.skip_layers = SkipLayersField(config.get('skip_layers', []))
        self._layer_config_cache = {}

    def get_layer_config(self, layer_name):
        """
        Get quant config for a specific layer, with caching
        Params:
            layer_name: str, layer name
        Return: dict or None, quant config for this layer
        """
        if layer_name in self._layer_config_cache:
            return self._layer_config_cache[layer_name]
        
        quant_cfg = self.quant_cfg.get_value().copy()
        
        fuzzy_weights_cfg = self.quant_cfg.get_fuzzy_config(layer_name, 'weights')
        if fuzzy_weights_cfg:
            weights_cfg = WeightsCfgField(fuzzy_weights_cfg)
            quant_cfg['weights_cfg'] = weights_cfg.get_value()
        elif quant_cfg['weights_cfg'] is None:
            self._layer_config_cache[layer_name] = None
            return None
        
        fuzzy_inputs_cfg = self.quant_cfg.get_fuzzy_config(layer_name, 'inputs')
        if fuzzy_inputs_cfg:
            inputs_cfg = InputsCfgField(fuzzy_inputs_cfg)
            quant_cfg['inputs_cfg'] = inputs_cfg.get_value()
        elif quant_cfg['inputs_cfg'] is None:
            quant_cfg['inputs_cfg'] = {'enable_quant': False}
        
        self._layer_config_cache[layer_name] = quant_cfg
        return quant_cfg

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
from amct_pytorch.config.utils import get_alg_name_from_config


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
        self.quant_type = config.get('type')
        self.symmetric = config.get('symmetric')
        self.strategy = config.get('strategy')
        self.group_size = config.get('group_size', None)
        self.check()
        self.value = self.set_value()
    
    def set_value(self):
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
        if self.strategy not in SUPPORT_QUANT_STRATEGY_WEIGHT:
            raise ValueError(f'Weights strategy only support{SUPPORT_QUANT_STRATEGY_WEIGHT}, but got {self.strategy}')
        if self.group_size is not None and self.strategy != 'group':
            raise ValueError(f'Weights group_size only support strategy group, but got {self.strategy}')
        if self.group_size is None and self.strategy == 'group':
            raise ValueError(f'Weights group_size is necessary, when weights strategy is group')
        if self.group_size is not None and (not isinstance(self.group_size, int) or self.group_size <= 0):
            raise ValueError(f'group_size only support positive int, but got {self.group_size}')


class InputsCfgField():
    def __init__(self, config):
        self.quant_input = config.get('enable_quant', True)
        if self.quant_input:
            self.quant_type = config.get('type')
            self.symmetric = config.get('symmetric')
            self.strategy = config.get('strategy')
            self.check()
        self.value = self.set_value()
    
    def set_value(self):
        if self.quant_input:
            return {'quant_type': self.quant_type,
                    'symmetric': self.symmetric,
                    'strategy': self.strategy}
        else:
            return {'enable_quant': False}
    
    def get_value(self):
        return self.value

    def check(self):
        if self.quant_type not in SUPPORT_INPUT_QUANT_DTYPE:
            raise ValueError(f'Inputs quant_dtype only support {SUPPORT_INPUT_QUANT_DTYPE}, but got {self.quant_type}')
        if self.symmetric not in [True, False]:
            raise ValueError(f'Inputs symmetric only support bool [True, False], but got {self.symmetric}')
        if self.strategy not in SUPPORT_QUANT_STRATEGY_INPUT:
            raise ValueError(f'Inputs strategy only support{SUPPORT_QUANT_STRATEGY_INPUT}, but got {self.strategy}')
        if self.strategy == 'token' and self.symmetric == False:
            raise ValueError(f'Inputs strategy token do not support asymmetric quantization')


class QuantCfgField():
    def __init__(self, config):
        self.weights_cfg = WeightsCfgField(config.get('weights', {}))
        self.inputs_cfg = InputsCfgField(config.get('inputs', {'enable_quant': False}))
        
        self.value = self.set_value()
    
    def set_value(self):
        return {'weights_cfg': self.weights_cfg.get_value(),
                'inputs_cfg': self.inputs_cfg.get_value()}
    
    def get_value(self):
        return self.value


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
            'custom': CustomAlgField
        }

        self.algs = []
        for alg_name, alg_attr in zip(alg_names, alg_attrs):
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
    
    def get_value(self):
        return self.value


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

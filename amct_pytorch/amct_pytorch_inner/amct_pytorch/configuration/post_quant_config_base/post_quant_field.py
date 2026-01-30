#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

field defination in json config.

"""
import copy
from collections import OrderedDict

from amct_pytorch.common.retrain_config.retrain_field import ConfigItem
from amct_pytorch.common.retrain_config.retrain_field import BatchNum
from amct_pytorch.utils.vars import BATCH_NUM, HIFLOAT8, INT8, INT4, MXFP4_E2M1, FLOAT4_E2M1, FLOAT4_E1M2
from amct_pytorch.utils.vars import DEFAULT_GRIDS_NUM, DEFAULT_SMOOTH_STRENGTH
from amct_pytorch.utils.vars import POST_QUANTIZE_ACT_TYPES
from amct_pytorch.utils.vars import POST_QUANTIZE_WTS_TYPES, POST_QUANTIZE_WTS_ONLY_TYPES


class GroupSize(ConfigItem):
    '''an object for GroupSize field'''
    def build(self, val, extra):
        '''build with user params'''
        self.check_type('GroupSize', val, int, extra[0])
        if val <= 0:
            raise ValueError("Group_Size must be a positive integer")
        self.value = val
    
    def build_default(self, extra):
        '''build using default params'''
        return


class WeightGranularity(ConfigItem):
    '''an object for WeightGranularity field'''
    def build(self, val, extra):
        '''build with user params'''
        self.check_type('WeightGranularity', val, str, extra[0])
        if val not in ('PER_TENSOR', 'PER_CHANNEL', 'PER_GROUP'):
            raise RuntimeError('weight_granularity only support PER_TENSOR, PER_CHANNEl and PER_GROUP')
        self.value = val

    def build_default(self, extra):
        '''build using default params'''
        self.value = 'PER_TENSOR'


class ActGranularity(ConfigItem):
    '''an object for ActGranularity field'''
    def build(self, val, extra):
        '''build with user params'''
        self.check_type('ActGranularity', val, str, extra[0])
        if val not in ('PER_TENSOR', 'PER_TOKEN'):
            raise RuntimeError('act_granularity only support PER_TENSOR and PER_TOKEN')
        self.value = val

    def build_default(self, extra):
        '''build using default params'''
        self.value = 'PER_TENSOR'


class RoundMode(ConfigItem):
    '''an object for RoundMode field'''
    def build(self, val, extra):
        '''build with user params'''
        self.check_type('RoundMode', val, str, extra[0])
        if val not in ('ROUND', 'RINT', 'HYBRID'):
            raise RuntimeError('round_mode only support RINT, ROUND and HYBRID')
        self.value = val

    def build_default(self, extra):
        '''build using default params'''
        return


class ActType(ConfigItem):
    '''an object for ActType field'''
    def build(self, val, extra):
        '''build with user params'''
        self.check_type('ActType', val, str, extra[0])
        if val not in POST_QUANTIZE_ACT_TYPES:
            raise RuntimeError('act_type only support {}'.format(POST_QUANTIZE_ACT_TYPES))
        self.value = val

    def build_default(self, extra):
        '''build using default params'''
        self.value = INT8


class WtsType(ConfigItem):
    '''an object for WtsType field'''
    def build(self, val, extra):
        '''build with user params'''
        self.check_type('WtsType', val, str, extra[0])
        support_wts_type_list = list(set(POST_QUANTIZE_WTS_TYPES + POST_QUANTIZE_WTS_ONLY_TYPES))
        if val not in support_wts_type_list:
            raise RuntimeError('wts_type only support {}'.format(support_wts_type_list))
        self.value = val

    def build_default(self, extra):
        '''build using default params'''
        self.value = INT8


class ActAsymmetric(ConfigItem):
    '''an object for ActAsymmetric field'''
    def build(self, val, extra):
        '''build with user params'''
        self.check_type('ActAsymmetric', val, bool, extra[0])
        self.value = val

    def build_default(self, extra):
        '''build using default params'''
        self.value = False


class WtsAsymmetric(ConfigItem):
    '''an object for WtsAsymmetric field'''
    def build(self, val, extra):
        '''build with user params'''
        self.check_type('WtsAsymmetric', val, bool, extra[0])
        self.value = val

    def build_default(self, extra):
        '''build using default params'''
        self.value = False

class WeightCompressOnly(ConfigItem):
    '''an object for WeightCompressOnly field'''
    def build(self, val, extra):
        '''build with user params'''
        self.check_type('WeightCompressOnly', val, bool, extra[0])
        self.value = val

    def build_default(self, extra):
        '''build using default params'''
        self.value = False


class WtsAlgo(ConfigItem):
    '''an object for WtsAlgo field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('WtsAlgo', val, str, extra[0])
        if val not in ("awq_quantize", "lut_quantize", "default"):
            raise ValueError("wts_algo {} is not supported".format(val))
        self.value = val

    def build_default(self, extra=None):
        '''inner method'''
        self.value = "default"


class GridsNum(ConfigItem):
    '''an object for GridsNum field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('GridsNum', val, int, extra[0])
        if val <= 0:
            raise ValueError("grids_num must be a positive integer")
        self.value = val

    def build_default(self, extra=None):
        '''inner method'''
        self.value = DEFAULT_GRIDS_NUM


class LutAlgo(ConfigItem):
    '''an object for LutAlgo field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('LutAlgo', val, str, extra[0])
        if val not in ('ARCTAN', 'CLUSTER'):
            raise ValueError('lut_algo only support ARCTAN and CLUSTER')
        self.value = val

    def build_default(self, extra=None):
        '''inner method'''
        self.value = 'CLUSTER'


class AwqAlgoParams(ConfigItem):
    ''' an object for AwqAlgoParams field '''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('AwqAlgoParams', val, dict, extra[0])
        awq_params_fields = {
            'grids_num': GridsNum
        }
        for key, cls in awq_params_fields.items():
            if key not in val.keys():
                self.build_default_util(key, cls, extra)
            else:
                self.build_util(key, cls, val.get(key), extra)


class LutAlgoParams(ConfigItem):
    ''' an object for LutAlgoParams field '''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('LutAlgoParams', val, dict, extra[0])
        lut_params_fields = {
            'lut_algo': LutAlgo
        }
        for key, cls in lut_params_fields.items():
            if key not in val.keys():
                self.build_default_util(key, cls, extra)
            else:
                self.build_util(key, cls, val.get(key), extra)


class GptqAlgoParams(ConfigItem):
    ''' an object for LutAlgoParams field '''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('GptqAlgoParams', val, dict, extra[0])
        gpt_params_fields = {}
        for key, cls in gpt_params_fields.items():
            if key not in val.keys():
                self.build_default_util(key, cls, extra)
            else:
                self.build_util(key, cls, val.get(key), extra)


class SmoothStrength(ConfigItem):
    '''an object for SmoothStrength field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('SmoothStrength', val, float, extra[0])
        self.value = val
 
    def build_default(self, extra=None):
        '''inner method'''
        self.value = DEFAULT_SMOOTH_STRENGTH
 
 
class ActSmoothParamField(ConfigItem):
    '''an object for ActSmooth field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('ActSmoothParamField', val, dict, extra[0])
        act_smooth_params_fields = {
            'smooth_strength': SmoothStrength
        }
        for key, cls in act_smooth_params_fields.items():
            if key not in val.keys():
                self.build_default_util(key, cls, extra)
            else:
                self.build_util(key, cls, val.get(key), extra)

    def build_default(self, extra):
        '''build using default params'''
        return


class WeightQuantAlgoParams(ConfigItem):
    '''an object for WeightQuantAlgoParams field'''
    def build(self, val, extra):
        '''build with user params'''
        self.check_type('WeightQuantAlgoParams', val, dict, extra[0])
        fields = {}
        wts_quant_algo_params_val = dict()
        if "awq_quantize" in val.keys():
            fields["awq_quantize"] = AwqAlgoParams
            wts_quant_algo_params_val.update(val.get("awq_quantize"))

        if "lut_quantize" in val.keys():
            fields["lut_quantize"] = LutAlgoParams
            wts_quant_algo_params_val.update(val.get("lut_quantize"))

        if "gptq_quantize" in val.keys():
            fields["gptq_quantize"] = GptqAlgoParams
            val["gptq_quantize"] = OrderedDict()
            wts_quant_algo_params_val.update(val.get("gptq_quantize"))

        for key, cls in fields.items():
            self.build_util(key, cls, wts_quant_algo_params_val, extra)

    def build_default(self, extra):
        '''build using default params'''
        return


class LayerConfig(ConfigItem):
    '''an object for layer quant config field'''
    fields = {
        'act_type': ActType,
        'wts_type': WtsType,
        'act_asymmetric': ActAsymmetric,
        'act_granularity': ActGranularity,
        'weight_compress_only': WeightCompressOnly,
        'weight_granularity': WeightGranularity,
        'round_mode': RoundMode
    }

    weight_only_fields = {
        'wts_type': WtsType,
        'weight_compress_only': WeightCompressOnly,
        'weight_granularity': WeightGranularity,
        'wts_asymmetric': WtsAsymmetric,
        'weight_quant_algo_params': WeightQuantAlgoParams,
        'round_mode': RoundMode
    }

    def build(self, val, extra):
        '''build with user params'''
        self.check_type('LayerConfig', val, dict)
        if val.get('weight_compress_only', False):
            fields = copy.deepcopy(self.weight_only_fields)
            if not val.get('weight_quant_algo_params'):
                del fields['weight_quant_algo_params']
            if val.get('weight_granularity') == 'PER_GROUP' and val.get('wts_type') in [
                MXFP4_E2M1, FLOAT4_E2M1, FLOAT4_E1M2, INT4, INT8]:
                fields['group_size'] = GroupSize
        else:
            fields = copy.deepcopy(self.fields)
            if val.get('act_smooth') is not None:
                fields['act_smooth'] = ActSmoothParamField
            
        for key, cls in fields.items():
            if key not in val.keys():
                self.build_default_util(key, cls, extra)
            else:
                self.build_util(key, cls, val.get(key), extra)
                del val[key]

    def build_default(self, extra):
        '''build using default params'''
        for key, cls in self.fields.items():
            self.build_default_util(key, cls, extra)


class RootConfig(ConfigItem):
    def get_global_keys(self):
        '''Get global config's keys '''
        global_keys = [BATCH_NUM]
        return global_keys

    def build(self, value, extra):
        '''inner method'''
        self.check_type('RootConfig', value, dict)

        if BATCH_NUM not in value:
            self.build_default_util(BATCH_NUM, BatchNum)
        else:
            self.build_util(BATCH_NUM, BatchNum, value.get(BATCH_NUM))
            del value[BATCH_NUM]

        for layer, layer_config in value.items():
            if layer not in extra.keys():
                raise ValueError("unsupported layer %s, please check the module type." % {layer})
            self.build_util(layer, LayerConfig, layer_config,
                            (layer, extra[layer]))
            self.fill_default_config(layer)
            del extra[layer]

    def build_default(self, extra):
        """
        build using default params
        Set the default value, including the LayerConfig and Batch_NUM global parameters under the root node
        """
        self.build_default_util(BATCH_NUM, BatchNum)
        if not extra:
            raise ValueError('No layer quant enabled')
        for layer, layer_type in extra.items():
            self.build_default_util(layer, LayerConfig, (layer, layer_type))
            self.fill_default_config(layer)

    def fill_default_config(self, layer):
        """
        fill default config in config tree
        If some default values vary with the data type, assign default values here
        """
        layer_config = self.children.get(layer).children
        if not layer_config.get('round_mode').value:
            if layer_config.get('wts_type').value == HIFLOAT8:
                layer_config['round_mode'].value = 'ROUND'
            else:
                layer_config['round_mode'].value = 'RINT'

        if not layer_config.get('weight_granularity').value:
            if layer_config.get('wts_type').value in (MXFP4_E2M1, FLOAT4_E2M1, FLOAT4_E1M2):
                layer_config['weight_granularity'].value = 'PER_GROUP'
            elif "lut_quantize" in layer_config.get('weight_quant_algo_params'):
                # LUT4
                layer_config['weight_granularity'].value = 'PER_GROUP'
            else:
                layer_config['weight_granularity'].value = 'PER_TENSOR'

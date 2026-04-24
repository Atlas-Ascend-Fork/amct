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
from collections import OrderedDict

from ....amct_pytorch.common.retrain_config.retrain_field import ConfigItem
from ....amct_pytorch.common.retrain_config.retrain_field import BatchNum
from ....amct_pytorch.common.config.field import NUM_OF_BINS_RANGE
from ....amct_pytorch.utils.vars import ACT_ALGO, IFMR, HFMG, BATCH_NUM, ASYMMETRIC,\
    ACTIVATION_OFFSET, QUANT_GRANULARITY, MAX_PERCENTILE, MIN_PERCENTILE, SEARCH_STEP, SEARCH_RANGE, NUM_OF_BINS
from ....amct_pytorch.common.utils.vars_util import DEFAULT_MAX_PERCENTILE, DEFAULT_MIN_PERCENTILE,\
    DEFAULT_MIN_PERCENTILE, DEFAULT_SEARCH_RANGE_START, DEFAULT_SEARCH_RANGE_END, DEFAULT_SEARCH_STEP,\
    DEFUALT_NUM_OF_BINS, SUPPORT_ACT_ALGO, PER_TENSOR_IDX, PER_CHANNEL_IDX


class ActAlgo(ConfigItem):
    '''an object for ActAlgo field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('ActAlgo', val, str, extra[0])
        if val not in SUPPORT_ACT_ALGO:
            raise ValueError('Act algo {} is not supported.'\
                'Supported act algo include ifmr and hfmg'.format(val))
        self.value = val
    
    def build_default(self, extra):
        '''inner method'''
        self.value = IFMR


class MaxPercentile(ConfigItem):
    '''an object for MaxPercentile field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('MaxPercentile', val, float, extra[0])
        if val <= 0.5 or val > 1.0:
            raise ValueError("The max_percentile must be greater than 0.5 "\
                "and less than or equal to 1.0")
        self.value = val

    def build_default(self, extra):
        '''inner method'''
        self.value = DEFAULT_MAX_PERCENTILE


class MinPercentile(ConfigItem):
    '''an object for MinPercentile field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('MinPercentile', val, float, extra[0])
        self.value = val

    def build_default(self, extra):
        '''inner method'''
        self.value = DEFAULT_MIN_PERCENTILE


class SearchRange(ConfigItem):
    '''an object for SearchRange field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('SearchRange', val, (list, tuple), extra[0])
        if len(val) != 2:
            raise ValueError("Search range must have two floating-point number")
        if val[0] > val[1]:
            raise ValueError('Search range start must smaller than search range end')
        self.value = val

    def build_default(self, extra):
        '''inner method'''
        self.value = [DEFAULT_SEARCH_RANGE_START, DEFAULT_SEARCH_RANGE_END]


class SearchStep(ConfigItem):
    '''an object for SearchStep field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('SearchStep', val, float, extra[0])
        if val <= 0:
            raise ValueError("Search step must be greater than zero")
        self.value = val

    def build_default(self, extra):
        '''inner method'''
        self.value = DEFAULT_SEARCH_STEP


class Asymmetric(ConfigItem):
    '''an object for Asymmetric field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('Asymmetric', val, (bool, type(None)), extra[0])
        self.value = val

    def build_default(self, extra):
        '''inner method'''
        self.value = None


class NumOfBins(ConfigItem):
    '''an object for NumOfBins field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('NumOfBins', val, int, extra[0])
        if val not in NUM_OF_BINS_RANGE:
            raise ValueError("num_of_bins {} must be in {}".format(
                val, NUM_OF_BINS_RANGE))
        self.value = val

    def build_default(self, extra):
        '''inner method'''
        self.value = DEFUALT_NUM_OF_BINS


class ActivationOffset(ConfigItem):
    '''an object for ActivationOffset field'''
    def build(self, val):
        '''inner method'''
        self.check_type("ActivationOffset", val, bool)
        self.value = val

    def build_default(self):
        '''inner method'''
        self.value = True


class ActQuantGranularity(ConfigItem):
    '''an object for ActQuantGranularity field'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('QuantGranularity', val, int, extra[0])
        if extra[1] not in ['Linear'] and val == PER_CHANNEL_IDX:
            raise ValueError('%s layer can not be channewised' % extra[0])
        self.value = val

    def build_default(self, extra):
        '''inner method'''
        self.value = PER_TENSOR_IDX


class KVCacheDataConfig(ConfigItem):
    '''an object for KVCacheDataConfig field'''
    def build(self, val, extra):
        self.check_type('KVCacheDataConfig', val, dict, extra[0])
        if ACT_ALGO not in val.keys():
            act_algo = IFMR
            self.build_default_util(ACT_ALGO, ActAlgo, extra)
        else:
            act_algo = val.get(ACT_ALGO)
            self.build_util(ACT_ALGO, ActAlgo, val.get(ACT_ALGO), extra)
            del val[ACT_ALGO]

        if act_algo == IFMR:
            self.build_config_by_key(MAX_PERCENTILE, MaxPercentile, val, extra)
            self.build_config_by_key(MIN_PERCENTILE, MinPercentile, val, extra)
            self.build_config_by_key(SEARCH_RANGE, SearchRange, val, extra)
            self.build_config_by_key(SEARCH_STEP, SearchStep, val, extra)
            self.build_config_by_key(ASYMMETRIC, Asymmetric, val, extra)
            self.build_config_by_key(QUANT_GRANULARITY, ActQuantGranularity, val, extra)
        else:
            self.build_config_by_key(NUM_OF_BINS, NumOfBins, val, extra)
            self.build_config_by_key(ASYMMETRIC, Asymmetric, val, extra)
            self.build_config_by_key(QUANT_GRANULARITY, ActQuantGranularity, val, extra)
    
    def build_config_by_key(self, key, cls, val, extra):
        '''inner method'''
        if key in val.keys():
            self.build_util(key, cls, val.get(key), extra)
        else:
            self.build_default_util(key, cls, extra)

    def build_default(self, extra):
        self.build_default_util(ACT_ALGO, ActAlgo, extra)
        self.build_default_util(MAX_PERCENTILE, MaxPercentile, extra)
        self.build_default_util(MIN_PERCENTILE, MinPercentile, extra)
        self.build_default_util(SEARCH_RANGE, SearchRange, extra)
        self.build_default_util(SEARCH_STEP, SearchStep, extra)
        self.build_default_util(ASYMMETRIC, Asymmetric, extra)
        self.build_default_util(QUANT_GRANULARITY, ActQuantGranularity, extra)


class KVCacheLayerConfig(ConfigItem):
    '''an object for KVCacheLayerConfig field'''
    fields = {
        'kv_data_quant_config': KVCacheDataConfig
    }

    def build(self, val, extra):
        '''inner method'''
        self.check_type('KVCacheLayerConfig', val, dict)
        for key, cls in self.fields.items():
            if key not in val.keys():
                self.build_default_util(key, cls, extra)
            else:
                self.build_util(key, cls, val.get(key), extra)

    def build_default(self, extra):
        '''inner method'''
        for key, cls in self.fields.items():
            self.build_default_util(key, cls, extra)


class KVCacheRootConfig(ConfigItem):
    '''an object for KVCacheRootConfig field'''
    def build(self, value, extra):
        kv_quant_layers = extra.get('kv_cache_quant_layers')
        for layer, layer_config in value.items():
            if layer not in kv_quant_layers.keys():
                raise ValueError('unsupported layer %s'.format(layer))
            self.build_util(layer, KVCacheLayerConfig, layer_config,
                            (layer, kv_quant_layers[layer]))

    def build_default(self, extra):
        '''inner method'''
        kv_quant_layers = extra.get('kv_cache_quant_layers')
        for layer, layer_type in kv_quant_layers.items():
            self.build_default_util(layer, KVCacheLayerConfig, (layer, layer_type))


class CalibrationGeneralConfigItem(ConfigItem):
    '''an object for CalibrationGeneralConfigItem field'''
    def build(self, value, extra):
        self.check_type('CalibrationGeneralConfigItem', value, dict)
        # handle batch_num
        if self.capacity.is_enable('BATCH_NUM'):
            key = BATCH_NUM
            self._build(key, value)
        else:
            if BATCH_NUM in value.keys():
                raise ValueError("unsupported batch_num")
        key = ACTIVATION_OFFSET
        self._build(key, value)

    def build_default(self, extra):
        '''inner method'''
        if self.capacity.is_enable('BATCH_NUM'):
            key = BATCH_NUM
            self.build_default_util(key, BatchNum)
        key = ACTIVATION_OFFSET
        self.build_default_util(key, ActivationOffset)

    def _build(self, key, value):
        class_dict = {BATCH_NUM: BatchNum, ACTIVATION_OFFSET: ActivationOffset}
        if key not in value and self.strong_check:
            raise ValueError("must has %s" % {key})
        if key not in value:
            self.build_default_util(key, class_dict.get(key))
        else:
            self.build_util(key, class_dict.get(key), value.get(key))
            del value[key]


class QuantCalibrationConfigRoot(ConfigItem):
    '''an object for QuantCalibrationConfigRoot field'''
    sub_fields = (
        ('general_config', CalibrationGeneralConfigItem),
        ('kv_config', KVCacheRootConfig)
    )

    def build(self, value, extra):
        self.check_type('QuantCalibrationConfigRoot', value, dict)
        for name, field in self.sub_fields:
            self.build_util(name, field, value, extra)

    def build_default(self, extra):
        for name, field in self.sub_fields:
            self.build_default_util(name, field, extra)

    def dump(self):
        ordered_config = OrderedDict()
        for _, val in self.children.items():
            child_tree = val.dump()
            for child_name, child_val in child_tree.items():
                if child_name not in ordered_config:
                    ordered_config[child_name] = child_val
                else:
                    ordered_config[child_name].update(child_val)
        return ordered_config

    def get_global_keys(self):
        '''Get global config's keys '''
        global_keys = [ACTIVATION_OFFSET]
        if self.capacity.is_enable('BATCH_NUM'):
            global_keys.append(BATCH_NUM)
        return global_keys
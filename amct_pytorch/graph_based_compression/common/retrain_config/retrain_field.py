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
from enum import Enum, unique
from ..utils.vars_util import INT4, INT8, INT16
from ..utils.vars_util import RETRAIN_DATA_TYPES
from ..utils.vars_util import RETRAIN_ACT_WTS_TYPES

DST_TYPE = 'dst_type'
CLIP_MAX = 'clip_max'
CLIP_MIN = 'clip_min'
FIXED_MIN = 'fixed_min'
ALGO = 'algo'
PRUNE_RATIO = 'prune_ratio'
CHANNEL_WISE = 'channel_wise'
BATCH_NUM = 'batch_num'
BATCH_NUM_UPPER = 'BATCH_NUM'
FAKEQUANT_PRECISION_MODE = ['DEFAULT', 'FORCE_FP16_QUANT']


@unique
class FakequantPrecision(Enum):
    '''enumeration of scale precision'''
    DEFAULT = 'DEFAULT'
    FORCE_FP16_QUANT = 'FORCE_FP16_QUANT'


class ConfigItem():
    '''an class for ConfigItem filed'''
    def __init__(self, graph_querier, capacity, strong_check=True):
        '''inner method'''
        self.graph_querier = graph_querier
        self.capacity = capacity
        self.children = {}
        self.value = None
        self.strong_check = strong_check

    @staticmethod
    def check_type(name, variable, typeinfo, layer=None):
        '''inner method'''
        error_msg = "Type of %s should be %s, but is %s" \
                % (name, typeinfo, type(variable))
        if layer is not None:
            error_msg = "%s for layer %s" % (error_msg, layer)
        if not isinstance(variable, typeinfo):
            raise TypeError(error_msg)

    def set_strong_check(self, check_type):
        '''inner method'''
        self.strong_check = check_type

    def add_child(self, key, item):
        '''inner method'''
        self.children[key] = item

    def dump(self):
        '''inner method'''
        if self.children.keys():
            ordered_config = OrderedDict()
            for k, val in self.children.items():
                ordered_config[k] = val.dump()
            return ordered_config
        return self.value

    def build_util(self, key, cls, value, extra=None):
        '''inner method'''
        item = cls(self.graph_querier, self.capacity, self.strong_check)
        if extra:
            item.build(value, extra)
        else:
            item.build(value)
        self.add_child(key, item)

    def build_default_util(self, key, cls, extra=None):
        '''inner method'''
        item = cls(self.graph_querier, self.capacity, self.strong_check)
        if extra:
            item.build_default(extra)
        else:
            item.build_default()
        self.add_child(key, item)


class Version(ConfigItem):
    '''an object for Version filed'''
    def build(self, val):
        '''inner method'''
        self.check_type("Version", val, int)
        if val != 1:
            raise ValueError("version should be 1")
        self.value = 1

    def build_default(self):
        '''inner method'''
        self.value = 1


class BatchNum(ConfigItem):
    '''an object for BatchNum filed'''
    def build(self, val):
        '''inner method'''
        self.check_type("BatchNum", val, int)
        if val <= 0:
            raise ValueError("batch_num(%d) should be greater than zero." \
                             % val)
        self.value = val

    def build_default(self):
        '''inner method'''
        self.value = 1


class FakequantPrecisionMode(ConfigItem):
    '''an object for FakequantPrecisionMode filed'''
    def build(self, val):
        '''inner method'''
        self.check_type("FakequantPrecisionMode", val, str)
        if val not in FAKEQUANT_PRECISION_MODE:
            raise ValueError(
                "Type of FakequantPrecisionMode should be {}".format(FAKEQUANT_PRECISION_MODE))
        self.value = val
 
    def build_default(self):
        '''inner method'''
        self.value = FakequantPrecision.DEFAULT.value


class RetrainEnable(ConfigItem):
    '''an object for RetrainEnable filed'''
    def build(self, val, extra=None):
        '''inner method'''
        self.check_type("RetrainEnable", val, bool, extra[0])
        self.value = val

    def build_default(self):
        '''inner method'''
        self.value = True


class DataAlgo(ConfigItem):
    '''an object for DataAlgo filed'''
    def build(self, val, extra=None):
        '''inner method'''
        self.check_type('DataAlgo', val, str)
        if val != 'ulq_quantize':
            raise ValueError('DataAlgo only support ulq_quantize for layer %s'\
                    % (extra[0]))
        self.value = val

    def build_default(self):
        '''inner method'''
        self.value = 'ulq_quantize'


class ClipMax(ConfigItem):
    '''an object for ClipMax filed'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('ClipMax', val, float, extra[0])
        if val <= 0:
            raise ValueError('clip max should be larger than 0 for layer %s' \
                    % extra[0])
        self.value = val


class ClipMin(ConfigItem):
    '''an object for ClipMin filed'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('ClipMin', val, float, extra[0])
        if val >= 0:
            raise ValueError('clip min should be less than 0 for layer %s'\
                    % extra[0])
        self.value = val


class FixedMin(ConfigItem):
    '''an object for FixedMin filed'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('FixedMin', val, bool, extra[0])
        self.value = val


class DataType(ConfigItem):
    '''an object for DataType filed'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type(DST_TYPE, val, str, extra[0])
        if val not in RETRAIN_DATA_TYPES:
            raise ValueError(
                "now only support {}, but is {}".format(RETRAIN_DATA_TYPES, val))
        self.value = val

    def build_default(self, extra=None): # pylint: disable=W0613
        '''inner method'''
        self.value = INT8


class RetrainDataConfig(ConfigItem):
    '''an object for RetrainDataConfig filed'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('RetrainDataConfig', val, dict, extra[0])
        key = ALGO
        if key not in val.keys():
            self.build_default_util(key, DataAlgo)
        else:
            self.build_util(key, DataAlgo, val.get(key), extra)
            del val[key]

        if CLIP_MAX in val.keys() and CLIP_MIN in val.keys():
            self.build_util(CLIP_MAX, ClipMax, val.get(CLIP_MAX), extra)
            self.build_util(CLIP_MIN, ClipMin, val.get(CLIP_MIN), extra)
            del val[CLIP_MAX]
            del val[CLIP_MIN]
        if FIXED_MIN in val.keys():
            self.build_util(FIXED_MIN, FixedMin, val.get(FIXED_MIN), extra)
            del val[FIXED_MIN]

        if DST_TYPE in val.keys():
            self.build_util(DST_TYPE, DataType, val.get(DST_TYPE), extra)
            del val[DST_TYPE]
        else:
            self.build_default_util(DST_TYPE, DataType)

        if val.keys():
            raise ValueError('Invalid keys in data config for layer %s'\
                    % extra[0])

    def build_default(self):
        '''inner method'''
        self.build_default_util(ALGO, DataAlgo)
        self.build_default_util(DST_TYPE, DataType)


class WeightAlgo(ConfigItem):
    '''an object for WeightAlgo filed'''
    def build(self, val, extra=None):
        '''inner method'''
        self.check_type('WeightAlgo', val, str, extra[0])
        if val not in ['arq_retrain', 'ulq_retrain']:
            raise ValueError(
                'WeightAlgo only supports [arq_retrain, ulq_retrain] ' \
                    'for layer %s' % (extra[0]))
        self.value = val

    def build_default(self, extra=None): # pylint: disable=W0613
        '''inner method'''
        self.value = 'arq_retrain'


class ChannelWise(ConfigItem):
    '''an object for ChannelWise filed'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('ChannelWise', val, bool, extra[0])
        if extra[1] in [
                'Linear', 'MatMul', 'InnerProduct', 'Pooling', 'AvgPool'
        ] and val is True:
            raise ValueError(' %s layer can not be channewised' % extra[0])
        self.value = val

    def build_default(self, extra):
        '''inner method'''
        if extra[1] in [
                'Linear', 'MatMul', 'InnerProduct', 'Pooling', 'AvgPool'
        ]:
            self.value = False
        else:
            self.value = True


class RetrainWeightConfig(ConfigItem):
    '''an object for RetrainWeightConfig filed'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('RetrainWeightConfig', val, dict, extra[0])
        wts_algo = ''
        key = ALGO
        if key not in val.keys():
            wts_algo = 'arq_retrain'
            self.build_default_util(key, WeightAlgo)
        else:
            wts_algo = val.get(key)
            self.build_util(key, WeightAlgo, val.get(key), extra)
            del val[key]

        if wts_algo == 'arq_retrain':
            self.build_config_by_key(CHANNEL_WISE, ChannelWise, val, extra)
            self.build_config_by_key(DST_TYPE, DataType, val, extra)
        elif wts_algo == 'ulq_retrain':
            self.build_config_by_key(DST_TYPE, DataType, val, extra)
            self.build_config_by_key(CHANNEL_WISE, ChannelWise, val, extra)
        else:
            raise ValueError("only support [arq_retrain, ulq_retrain], "
                             " but is {}".format(wts_algo))
        if val.keys():
            raise ValueError('Invalid keys in weight config for layer %s'\
                    % extra[0])

    def build_default(self, extra):
        '''inner method'''
        self.build_default_util(ALGO, WeightAlgo)
        self.build_default_util(CHANNEL_WISE, ChannelWise, extra)
        self.build_default_util(DST_TYPE, DataType)

    def build_config_by_key(self, key, cls, val, extra):
        '''inner method'''
        if key in val.keys():
            self.build_util(key, cls, val.get(key), extra)
            del val[key]
        else:
            self.build_default_util(key, cls, extra)


class RegularPruneEnable(ConfigItem):
    '''an object for RegularPruneEnable filed'''
    def build(self, val, extra=None):
        '''inner method'''
        self.check_type("RegularPruneEnable", val, bool, extra[0])
        self.value = val

    def build_default(self):
        '''inner method'''
        self.value = True


class PruneRatio(ConfigItem):
    '''an object for PruneRatio filed'''
    @staticmethod
    def build_default():
        '''inner method'''
        pass

    def build(self, val, extra=None):
        '''inner method'''
        self.check_type("PruneRatio", val, float, extra[0])
        if val <= 0 or val >= 1:
            raise ValueError('prune ration should be greater than 0 and less '\
                    'than 1 for layer %s' % extra[0])
        self.value = val


class AscendOptimized(ConfigItem):
    '''an object for AscendOptimized filed'''
    def build(self, val, extra=None):
        '''inner method'''
        self.check_type("AscendOptimized", val, bool, extra[0])
        self.value = val

    def build_default(self):
        '''inner method'''
        self.value = True


class NOutOfMType(ConfigItem):
    '''an object for NOutOfMType filed'''
    def __init__(self, graph_querier, capacity, strong_check=True):
        super().__init__(graph_querier, capacity, strong_check)
        self.support_n_out_of_m_type = ['M4N2']

    @staticmethod
    def build_default():
        '''inner method'''
        pass

    def build(self, val, extra=None):
        '''inner method'''
        self.check_type("NOutOfMType", val, str, extra[0])
        if val not in self.support_n_out_of_m_type:
            raise ValueError(
                'NOutOfMType only supports %s for layer %s' \
                % (self.support_n_out_of_m_type, extra[0]))
        self.value = val


class UpdateFreq(ConfigItem):
    '''an object for UpdateFreq filed'''
    @staticmethod
    def build_default():
        '''inner method'''
        pass

    def build(self, val, extra=None):
        '''inner method'''
        self.check_type("UpdateFreq", val, int, extra[0])
        if val < 0:
            raise ValueError('selective prune update freq should be no less than 0 '\
                    'for layer %s' % extra[0])
        self.value = val


class PruneType(ConfigItem):
    '''an object for PruneType filed'''
    __default_value = 'no_prune_enable'

    def __init__(self, graph_querier, capacity, strong_check=True):
        super().__init__(graph_querier, capacity, strong_check)
        self.support_algos = ['filter_prune', 'selective_prune']

    @property
    def val(self):
        '''
        get value of current algo
        Params: None
        Return: value, a string
        '''
        return self.value

    def build(self, val, extra=None):
        '''inner method'''
        self.check_type('PruneType', val, str, extra[0])
        if val not in self.support_algos:
            raise ValueError(
                'PruneType only supports %s for layer %s' \
                % (self.support_algos, extra[0]))
        self.value = val

    def build_default(self, extra=None): # pylint: disable=W0613
        '''inner method'''
        self.value = self.__default_value


class RegularPruneAlgo(ConfigItem):
    '''an object for RegularPruneAlgo filed'''
    def __init__(self, graph_querier, capacity, strong_check=True):
        super().__init__(graph_querier, capacity, strong_check)
        self.support_algos = ['balanced_l2_norm_filter_prune', 'l1_selective_prune']

    @property
    def val(self):
        '''
        get value of current algo
        Params: None
        Return: value, a string
        '''
        return self.value

    def build(self, val, extra=None):
        '''inner method'''
        self.check_type('RegularPruneAlgo', val, str, extra[0])
        if val not in self.support_algos:
            raise ValueError(
                'RegularPruneAlgo only supports %s for layer %s' \
                % (self.support_algos, extra[0]))
        self.value = val

    def build_default(self, extra=None): # pylint: disable=W0613
        '''inner method'''
        pass


class BcpPruneConfig(ConfigItem):
    '''an object for BcpPruneConfig filed'''
    fields = {
        PRUNE_RATIO: PruneRatio,
        'ascend_optimized': AscendOptimized,
    }

    def build(self, val, extra):
        '''inner method'''
        self.check_type('BcpPruneConfig', val, dict, extra[0])

        def add_one_field(key, field, val):
            if key not in val.keys():
                self.build_default_util(key, field)
            else:
                self.build_util(key, field, val.get(key), extra)
                del val[key]

        for key in [PRUNE_RATIO, 'ascend_optimized']:
            add_one_field(key, self.fields.get(key), val)

        if val.keys():
            raise ValueError('Invalid keys %s in prune config for layer %s'\
                    % (val.keys(), extra[0]))

    def build_default(self):
        '''inner method'''
        for key, field in self.fields.items():
            self.build_default_util(key, field)

    def build_config_by_key(self, key, cls, val, extra):
        '''inner method'''
        if key in val.keys():
            self.build_util(key, cls, val.get(key), extra)
            del val[key]
        else:
            self.build_default_util(key, cls, extra)


class RegularPruneConfig(ConfigItem):
    '''an object for PruneConfig filed'''
    fields = {
        'prune_type': PruneType,
        ALGO: RegularPruneAlgo,
        # for balanced_l2_norm_filter_prune
        PRUNE_RATIO: PruneRatio,
        'ascend_optimized': AscendOptimized,
        # for l1_selective_prune
        'n_out_of_m_type': NOutOfMType,
        'update_freq': UpdateFreq
    }

    def build(self, val, extra):
        '''inner method'''
        self.check_type('PruneConfig', val, dict, extra[0])

        def add_one_field(key, field, val):
            if key not in val.keys():
                self.build_default_util(key, field)
            else:
                self.build_util(key, field, val.get(key), extra)
                del val[key]

        for key in ['prune_type', ALGO]:
            add_one_field(key, self.fields.get(key), val)

        if self.children.get(ALGO).val == 'balanced_l2_norm_filter_prune':
            for key in [PRUNE_RATIO, 'ascend_optimized']:
                add_one_field(key, self.fields.get(key), val)
        elif self.children.get(ALGO).val == 'l1_selective_prune':
            for key in ['n_out_of_m_type', 'update_freq']:
                add_one_field(key, self.fields.get(key), val)

        if val.keys():
            raise ValueError('Invalid keys in prune config for layer %s'\
                    % extra[0])

    def build_default(self):
        '''inner method'''
        for key, field in self.fields.items():
            self.build_default_util(key, field)

    def build_config_by_key(self, key, cls, val, extra):
        '''inner method'''
        if key in val.keys():
            self.build_util(key, cls, val.get(key), extra)
            del val[key]
        else:
            self.build_default_util(key, cls, extra)


class LayerConfig(ConfigItem):
    '''an object for LayerConfig filed'''
    fields = {
        'retrain_enable': RetrainEnable,
        'retrain_data_config': RetrainDataConfig,
        'retrain_weight_config': RetrainWeightConfig,
        'regular_prune_enable': RegularPruneEnable,
        'regular_prune_config': RegularPruneConfig,
    }

    @staticmethod
    def retrain_fields():
        """
        Get fields for quant retrain
        Params: None
        Returns: a list
        """
        return ['retrain_enable', 'retrain_data_config', 'retrain_weight_config']

    @staticmethod
    def prune_fields():
        """
        Get fields for prune retrain
        Params: None
        Returns: a list
        """
        return ['regular_prune_enable', 'regular_prune_config']

    def build(self, val, extra):
        '''inner method'''
        self.check_type('LayerConfig', val, dict)
        for key, cls in self.fields.items():
            if key not in val.keys():
                if key == 'retrain_weight_config':
                    self.build_default_util(key, cls, extra)
                else:
                    self.build_default_util(key, cls)
            else:
                if key == 'retrain_enable':
                    self.build_util(key, cls, val.get(key), extra)
                else:
                    self.build_util(key, cls, val.get(key), extra)
                del val[key]

    def build_default(self, extra):
        '''inner method'''
        for key, cls in self.fields.items():
            if key == 'retrain_weight_config':
                self.build_default_util(key, cls, extra)
            else:
                self.build_default_util(key, cls)


class RootConfig(ConfigItem):
    '''an object for RootConfig filed'''
    def check_layer_config_legal(self, layer):
        '''check layer config legal'''
        weight_config = self.children.get(layer).children.get('retrain_weight_config').children
        act_config = self.children.get(layer).children.get('retrain_data_config').children
        if DST_TYPE in weight_config:
            weight_dst = weight_config.get(DST_TYPE).value
            act_dst = act_config.get(DST_TYPE).value
            act_wts_type = 'A{}W{}'.format(act_dst.split('INT')[-1], weight_dst.split('INT')[-1])
            if act_wts_type not in RETRAIN_ACT_WTS_TYPES:
                raise ValueError(
                    "Now do not support activation and weights "
                    "data_type, activation is {} and weight "
                    "is {}".format(act_dst, weight_dst))
            if "AvgPool" in layer and self.children.get(layer).children.get('retrain_enable').value:
                if weight_dst == 'INT4':
                    raise RuntimeError(
                        'Now AvgPool Layer does not support INT4')

    def build(self, value, extra):
        '''inner method'''
        self.check_type('RootConfig', value, dict)
        # handle version
        key = 'version'
        self._build(key, value)
        # handle fakequant_precision_mode
        key = 'fakequant_precision_mode'
        self._build(key, value)

        # handle batch_num
        if self.capacity.is_enable(BATCH_NUM_UPPER):
            key = BATCH_NUM
            self._build(key, value)
        else:
            if BATCH_NUM in value.keys():
                raise ValueError("unsupported batch_num")
        all_disable = True
        for layer, layer_config in value.items():
            if layer not in extra.keys():
                raise ValueError("unsupported layer %s" % {layer})
            if layer_config.get('retrain_enable') or layer_config.get('regular_prune_enable'):
                all_disable = False
            self.build_util(layer, LayerConfig, layer_config,
                            (layer, extra[layer]))
            self.check_layer_config_legal(layer)
            del extra[layer]
        disable_config = {}
        for layer, layer_type in extra.items():
            disable_config['retrain_enable'] = False
            self.build_util(layer, LayerConfig, disable_config,
                            (layer, layer_type))

        if all_disable:
            raise ValueError('No layer retrain enabled')

    def build_default(self, extra):
        '''inner method'''
        key = 'version'
        self.build_default_util(key, Version)
        key = 'fakequant_precision_mode'
        self.build_default_util(key, FakequantPrecisionMode)
        if self.capacity.is_enable(BATCH_NUM_UPPER):
            key = BATCH_NUM
            self.build_default_util(key, BatchNum)
        if not extra:
            raise ValueError('No layer retrain enabled')
        for layer, layer_type in extra.items():
            self.build_default_util(layer, LayerConfig, (layer, layer_type))

    def get_global_keys(self):
        '''Get global config's keys '''
        global_keys = ['version', 'fakequant_precision_mode']
        if self.capacity.is_enable(BATCH_NUM_UPPER):
            global_keys.append(BATCH_NUM)
        return global_keys

    def _build(self, key, value):
        class_dict = {"version": Version, "batch_num": BatchNum, "fakequant_precision_mode": FakequantPrecisionMode}
        if key not in value and self.strong_check and key != "fakequant_precision_mode":
            raise ValueError("must has %s" % {key})
        if key not in value:
            self.build_default_util(key, class_dict.get(key))
        else:
            self.build_util(key, class_dict.get(key), value.get(key))
            del value[key]

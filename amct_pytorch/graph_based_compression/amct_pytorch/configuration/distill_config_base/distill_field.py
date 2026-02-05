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
from ....amct_pytorch.common.retrain_config.retrain_field import Version
from ....amct_pytorch.common.retrain_config.retrain_field import BatchNum
from ....amct_pytorch.common.retrain_config.retrain_field import DataAlgo
from ....amct_pytorch.common.retrain_config.retrain_field import ClipMax
from ....amct_pytorch.common.retrain_config.retrain_field import ClipMin
from ....amct_pytorch.common.retrain_config.retrain_field import FixedMin
from ....amct_pytorch.common.retrain_config.retrain_field import ChannelWise
from ....amct_pytorch.common.utils.vars_util import INT4, INT8
from ....amct_pytorch.utils.vars import DST_TYPE
from ....amct_pytorch.utils.vars import CLIP_MAX
from ....amct_pytorch.utils.vars import CLIP_MIN
from ....amct_pytorch.utils.vars import FIXED_MIN


class GroupSize(ConfigItem):
    '''an object for GroupSize filed'''
    def build(self, val):
        '''inner method'''
        self.check_type("GroupSize", val, int)
        if val <= 0:
            raise ValueError("group_size({}) should be greater than zero.".format(val))
        self.value = val

    def build_default(self):
        '''inner method'''
        self.value = 1


class DataDump(ConfigItem):
    '''an object for DataDump filed'''
    def build(self, val):
        '''inner method'''
        self.check_type("DataDump", val, bool)
        self.value = val

    def build_default(self):
        '''inner method'''
        self.value = False


class QuantEnable(ConfigItem):
    '''an object for QuantEnable filed'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type("QuantEnable", val, bool, extra[0])
        self.value = val

    def build_default(self):
        '''inner method'''
        self.value = True


class DataType(ConfigItem):
    '''an object for DataType filed'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type(DST_TYPE, val, str, extra[0])
        if val not in [INT4, INT8]:
            raise ValueError(
                "now only support ['INT4', 'INT8'], but is {}".format(val))
        self.value = val

    def build_default(self, extra=None):
        '''inner method'''
        self.value = INT8


class DistillDataConfig(ConfigItem):
    '''an object for DistillDataConfig filed'''
    def build(self, val, extra=None):
        '''inner method'''
        self.check_type('DistillDataConfig', val, dict, extra[0])
        key = 'algo'
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
            raise ValueError('Invalid keys{} in data config for layer {}'.format(list(val.keys()), extra[0]))

    def build_default(self):
        '''inner method'''
        self.build_default_util('algo', DataAlgo)
        self.build_default_util(DST_TYPE, DataType)


class WeightAlgo(ConfigItem):
    '''an object for WeightAlgo filed'''
    def build(self, val, extra=None):
        '''inner method'''
        self.check_type('WeightAlgo', val, str, extra[0])
        if val not in ['arq_distill', 'ulq_distill']:
            raise ValueError(
                'WeightAlgo only supports [arq_distill, ulq_distill] for layer {}'.format(extra[0]))
        self.value = val

    def build_default(self, extra=None):
        '''inner method'''
        self.value = 'arq_distill'


class DistillWeightConfig(ConfigItem):
    '''an object for DistillWeightConfig filed'''
    def build(self, val, extra):
        '''inner method'''
        self.check_type('DistillWeightConfig', val, dict, extra[0])
        wts_algo = ''
        key = 'algo'
        if key not in val.keys():
            wts_algo = 'arq_distill'
            self.build_default_util(key, WeightAlgo)
        else:
            wts_algo = val.get(key)
            self.build_util(key, WeightAlgo, val.get(key), extra)
            del val[key]

        self.build_config_by_key('channel_wise', ChannelWise, val, extra)
        self.build_config_by_key(DST_TYPE, DataType, val, extra)

        if val.keys():
            raise ValueError('Invalid keys{} in weight config for layer {}'.format(list(val.keys()), extra[0]))

    def build_default(self, extra):
        '''inner method'''
        self.build_default_util('algo', WeightAlgo)
        self.build_default_util('channel_wise', ChannelWise, extra)
        self.build_default_util(DST_TYPE, DataType)

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
        'quant_enable': QuantEnable,
        'distill_data_config': DistillDataConfig,
        'distill_weight_config': DistillWeightConfig,
    }

    def build(self, val, extra):
        '''inner method'''
        self.check_type('LayerConfig', val, dict)
        for key, cls in self.fields.items():
            if key not in val.keys():
                if key == 'distill_weight_config':
                    self.build_default_util(key, cls, extra)
                else:
                    self.build_default_util(key, cls)
            else:
                if key == 'quant_enable':
                    self.build_util(key, cls, val.get(key), extra)
                else:
                    self.build_util(key, cls, val.get(key), extra)
                del val[key]

    def build_default(self, extra):
        '''inner method'''
        for key, cls in self.fields.items():
            if key == 'distill_weight_config':
                self.build_default_util(key, cls, extra)
            else:
                self.build_default_util(key, cls)


class DistillGroup(ConfigItem):
    '''an object for DistillGroup filed'''
    def build(self, val):
        '''inner method'''
        self.check_type('DistillGroup', val, list)
        self.value = val


class DistillRootConfig(ConfigItem):
    '''an object for DistillRootConfig filed'''
    def check_layer_config_legal(self, layer):
        '''check layer config legal'''
        weight_config = self.children.get(layer).children.get('distill_weight_config').children
        act_config = self.children.get(layer).children.get('distill_data_config').children
        if DST_TYPE in weight_config:
            weight_dst = weight_config.get(DST_TYPE).value
            act_dst = act_config.get(DST_TYPE).value
            if weight_dst != act_dst:
                raise ValueError(
                    "Now do not support activation and weights with"
                    " different data_type, activation is {} and weight "
                    "is {}".format(act_dst, weight_dst))

    def build(self, value, extra):
        '''inner method'''
        self.check_type('DistillRootConfig', value, dict)
        # handle version
        self._build('version', value)

        # handle batch_num
        self._build('batch_num', value)

        # handle group_size
        self._build('group_size', value)

        # handle data_dump
        self._build('data_dump', value)

        # handle distill_group
        key = 'distill_group'
        if key in value.keys():
            self.build_util(key, DistillGroup, value[key])
            del value[key]

        all_disable = True
        for layer, layer_config in value.items():
            if layer not in extra.keys():
                raise ValueError("unsupported layer {} for distillation".format(layer))
            if layer_config.get('quant_enable'):
                all_disable = False
            self.build_util(layer, LayerConfig, layer_config,
                            (layer, extra[layer]))
            self.check_layer_config_legal(layer)
            del extra[layer]
        disable_config = {}
        for layer, layer_type in extra.items():
            disable_config['quant_enable'] = False
            self.build_util(layer, LayerConfig, disable_config,
                            (layer, layer_type))

        if all_disable:
            raise ValueError('No layer enabled distillation')

    def build_default(self, groups, extra):
        '''inner method'''
        self.build_default_util('version', Version)
        self.build_default_util('batch_num', BatchNum)
        self.build_default_util('group_size', GroupSize)
        self.build_default_util('data_dump', DataDump)
        self.build_util('distill_group', DistillGroup, groups)
        for layer, layer_type in extra.items():
            self.build_default_util(layer, LayerConfig, (layer, layer_type))

    def _build(self, key, value):
        class_dict = {"version": Version,
                      "batch_num": BatchNum,
                      "group_size": GroupSize,
                      "data_dump": DataDump}
        if key not in value and self.strong_check:
            raise ValueError("must has {}".format(key))
        if key not in value:
            self.build_default_util(key, class_dict.get(key))
        else:
            self.build_util(key, class_dict.get(key), value.get(key))
            del value[key]

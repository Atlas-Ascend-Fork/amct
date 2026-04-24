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

from ...amct_pytorch.common.utils.prune_record_attr_util import AttrProtoHelper
from ...amct_pytorch.configuration.retrain_config import RetrainConfig
from ...amct_pytorch.utils.module_info import ModuleInfo
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.custom_op import bcp


PRUNE_RATIO = 'prune_ratio'


class PuneIndexHelper:
    """ Helper for get prune index"""
    def __init__(self, model_helper, prune_record, record_helper):
        """ inti function"""
        self.param = {}
        self.model_helper = model_helper
        self.conf = RetrainConfig()
        self.record = prune_record
        self.record_helper = record_helper

        self.global_prune_ratio = None
        self.global_prune_group = None
        self.global_ascend_optimized = False

    @staticmethod
    def do_bcp(input_tensors, prune_axises, bcp_param):
        """ find cout mask by algorithm BCP"""
        cout_mask = bcp(input_tensors, prune_axises, bcp_param.get('prune_ratio'),
            bcp_param.get('prune_group'), bcp_param.get('ascend_optimized'))
        return cout_mask

    @staticmethod
    def prune_cout_to_prune_index(prune_cout):
        """ trans mask to index"""
        prune_index = [idx for idx, val in enumerate(prune_cout) if val == 0]
        return prune_index

    def prepare_param(self):
        """process param before call algorithm, including: 1. get tensor 2. set global prune ratio
        """
        def get_module(name):
            try:
                module = self.model_helper.get_module(name)
                return module
            except RuntimeError:
                LOGGER.logd('Cannot find "%s" in model, cannot do pune '
                            % (name))
                return None

        self.global_prune_group = self.record_helper.get_prune_group(self.record)

        for producer in self.record.producer:
            name = producer.name
            if name in self.param:
                raise RuntimeError('node {} has already been added.'.format(name))
            self.param[name] = {}
            # use min prune ration
            layer_prune_config = self.conf.get_layer_prune_config(name)
            if not layer_prune_config.get(PRUNE_RATIO):
                layer_prune_config[PRUNE_RATIO] = 0
            if self.global_prune_ratio is None:
                self.global_prune_ratio = layer_prune_config.get(PRUNE_RATIO)
            else:
                self.global_prune_ratio = min(self.global_prune_ratio, layer_prune_config.get(PRUNE_RATIO))
            if layer_prune_config.get("ascend_optimized") and layer_prune_config.get("ascend_optimized"):
                self.global_ascend_optimized = True

            prune_range = self.record_helper.get_range(self.record, name)
            module = get_module(name)
            weight_tensor = module.weight.data
            cout_axis, _ = ModuleInfo.get_wts_cout_cin(module)
            if cout_axis == 0:
                weight_tensor = weight_tensor[prune_range[0]:prune_range[1]]
            else:
                raise RuntimeError('unexpected weight cout axis is {}'.format(cout_axis))
            self.param.get(name)['axis'] = cout_axis
            self.param.get(name)['tensor'] = weight_tensor

    def add_to_record(self, prune_cout):
        """add prune index to record"""
        if not prune_cout:
            LOGGER.logd('no index to prune {}.'.format([producer.name for producer in self.record.producer]))
            return
        for producer in self.record.producer:
            attr_helper = AttrProtoHelper(producer)
            begin = attr_helper.get_attr_value('begin')
            prune_index = [idx + begin for idx in prune_cout]
            attr_helper.set_attr_value('prune_index', 'INTS', prune_index)

        for consumer in self.record.consumer:
            attr_helper = AttrProtoHelper(consumer)
            begin = attr_helper.get_attr_value('begin')
            prune_index = [idx + begin for idx in prune_cout]
            attr_helper.set_attr_value('prune_index', 'INTS', prune_index)

    def cal_prune_cout(self,):
        """calculate prune cout index and add it to record"""
        self.prepare_param()
        if not self.param:
            raise RuntimeError('no node is to calculate prune cout index.')

        tensors = []
        axises = []
        for name in self.param:
            tensors.append(self.param[name]['tensor'])
            axises.append(self.param[name]['axis'])
        prune_param = {
            'prune_ratio': self.global_prune_ratio,
            'ascend_optimized': self.global_ascend_optimized,
            'prune_group': self.global_prune_group
        }

        # tensor of 1/0 
        prune_mask = self.do_bcp(tensors, axises, prune_param)
        # list of prune cout index
        prune_index = PuneIndexHelper.prune_cout_to_prune_index(prune_mask) 

        self.add_to_record(prune_index)


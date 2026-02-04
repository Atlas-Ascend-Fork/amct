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
import copy
import torch

from ...amct_pytorch.common.prune.prune_recorder_helper import PruneRecordHelper
from ...amct_pytorch.common.utils.prune_record_attr_util import AttrProtoHelper

from ...amct_pytorch.utils.module_info import create_prune_helper
from ...amct_pytorch.utils.module_info import ModulePruneHelper
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.vars import PASSIVE_PRUNABLE_ONNX_TYPES

ACTIVE_PRUNE_SPLIT = 'active_prune_split'
PASSIVE_PRUNE_SPLIT = 'passive_prune_split'


class RegularModelPruner:
    """ Really do regular prune model """
    def __init__(self, graph, record, mode_input_data):
        """ init function"""
        self.graph = graph
        self.record = record
        self.mode_input_data = mode_input_data
        self.model_helper = ModuleHelper(self.graph.model)
        self.record_helper = PruneRecordHelper(self.record, self.graph)
        self.split_info = {}

    def do_prune(self):
        """really do prune"""
        # try to run the original model in train mode
        if not self.run_model_train_forward():
            raise RuntimeError("the model cannot do prune for do run forward fail in training mode with "
                               "input_data.", "RegularModelPruner")
        self.split_info = PruneRecordHelper.prepare_split_info(self.record.prune_record)
        uncertain_index = [idx for idx in range(len(self.record.prune_record))]
        delete_records = []

        def prune_by_index(left, right):
            """
            Function: prune model according to prune_record[left:right] with dichotomy
            param:left, int number
            param:right, int number
            """
            if right - left <= 0:
                return
            success = self.prune_by_several_records(uncertain_index[left:right])
            if success:
                return
            # if prune is fail and only do one prune_record, the prune_record is invalid
            if right - left == 1:
                delete_records.extend(self.record.prune_record[left:right])
                return
            # if prune is fail, do prune in smaller range
            bound = (left + right) // 2
            prune_by_index(left, bound)
            prune_by_index(bound, right)

        prune_by_index(0, len(uncertain_index))

        for delete_record in delete_records:
            for producer in delete_record.producer:
                LOGGER.logw("Disable prune layer {} for fail to run forward. Please skip it by setting "
                            "regular_prune_skip_layers, which will help to reduce execution time of "
                            "create_prune_retrain_model".format(producer.name), "RegularModelPruner")
            self.record.prune_record.remove(delete_record)

    def prune_by_several_records(self, record_indexes):
        """
        Function: do prune by prune_record and record_indexes
        param:record_indexes, a list, containing the index of prune_record which is for prune
        return: success, bool, whether the prune is good for forward
        """
        # prune module
        model_backup = {}
        for index in record_indexes:
            prune_record = self.record.prune_record[index]
            # whether this record need to do real prune
            if not prune_record.producer:
                continue
            attr_helper = AttrProtoHelper(prune_record.producer[0])
            if not attr_helper.has_attr('prune_index'):
                continue
            for producer in prune_record.producer:
                if producer.name not in model_backup:
                    model_backup[producer.name] = {}
                node_backup = model_backup[producer.name]
                self.prune_producer_cout(producer, node_backup)
            for consumer in prune_record.consumer:
                if consumer.name not in model_backup:
                    model_backup[consumer.name] = {}
                node_backup = model_backup[consumer.name]
                self.prune_consumer_cin(consumer, node_backup)

        # test to do train forward
        if self.run_model_train_forward():
            return True

        # restore module
        for node_name, node_backup in model_backup.items():
            if not node_backup:
                continue
            module = self.get_module(node_name)
            ModulePruneHelper.restore(module, node_backup['module_backup'], node_name)
            if ACTIVE_PRUNE_SPLIT in node_backup:
                self.split_info[node_name][ACTIVE_PRUNE_SPLIT] = node_backup[ACTIVE_PRUNE_SPLIT]
            if PASSIVE_PRUNE_SPLIT in node_backup:
                self.split_info[node_name][PASSIVE_PRUNE_SPLIT] = node_backup[PASSIVE_PRUNE_SPLIT]
        return False

    def run_model_train_forward(self):
        """
        Function: run model forward function in training mode
        param: None
        return: None
        """
        self.graph.model.train()
        buffer = copy.deepcopy(self.graph.model.state_dict())
        if isinstance(self.mode_input_data, torch.Tensor):
            self.mode_input_data = (self.mode_input_data, )
        success = True
        try:
            self.graph.model.forward(*self.mode_input_data)
        except Exception:
            success = False
        finally:
            self.graph.model.load_state_dict(buffer)
            del buffer
        return success

    def get_module(self, name):
        """ get module by name"""
        try:
            module = self.model_helper.get_module(name)
            return module
        except RuntimeError:
            LOGGER.logd('Cannot find "%s" in model, cannot do pune '
                        % (name))
            return None

    def prune_producer_cout(self, producer, node_backup):
        """
        Function: prune producer's cout channel
        param:producer, proto type
        param:node_backup, dict, backup info of corresponding module
        return:None
        """
        name = producer.name
        # get begin and prune_index
        attr_helper = AttrProtoHelper(producer)
        ori_begin = attr_helper.get_attr_value('begin')
        prune_index = attr_helper.get_attr_value('prune_index')
        real_begin = self.split_info[name][ACTIVE_PRUNE_SPLIT]['ori_begin_{}'.format(ori_begin)]
        # really prune filter cout
        module = self.get_module(name)
        prune_helper = create_prune_helper(module)
        module_backup = prune_helper.do_prune(
            [], cout_prune_list=[idx - ori_begin + real_begin for idx in prune_index])
        if 'module_backup' not in node_backup:
            node_backup['module_backup'] = module_backup
        # update prune_split info
        if ACTIVE_PRUNE_SPLIT not in node_backup:
            node_backup[ACTIVE_PRUNE_SPLIT] = copy.deepcopy(self.split_info[name][ACTIVE_PRUNE_SPLIT])
        for key in self.split_info[name][ACTIVE_PRUNE_SPLIT]:
            if int(key[10:]) > ori_begin:
                self.split_info[name][ACTIVE_PRUNE_SPLIT][key] -= len(prune_index)


    def prune_consumer_cin(self, consumer, node_backup):
        """
        Function: prune consumer's cin channel
        param:consumer, proto type
        param:node_backup, dict, backup info of corresponding module
        return:None
        """
        name = consumer.name
        attr_helper = AttrProtoHelper(consumer)
        node_type = attr_helper.get_attr_value('type')
        if node_type not in PASSIVE_PRUNABLE_ONNX_TYPES:
            return
        # get begin and prune_index
        attr_helper = AttrProtoHelper(consumer)
        ori_begin = attr_helper.get_attr_value('begin')
        prune_index = attr_helper.get_attr_value('prune_index')
        real_begin = self.split_info[name][PASSIVE_PRUNE_SPLIT]['ori_begin_{}'.format(ori_begin)]
        # really prune filter cin
        module = self.get_module(name)
        prune_helper = create_prune_helper(module)
        module_backup = prune_helper.do_prune(cin_prune_list=[idx - ori_begin + real_begin for idx in prune_index],
                                              cout_prune_list=[])
        if 'module_backup' not in node_backup:
            node_backup['module_backup'] = module_backup
        # update prune_split info
        if PASSIVE_PRUNE_SPLIT not in node_backup:
            node_backup[PASSIVE_PRUNE_SPLIT] = copy.deepcopy(self.split_info[name][PASSIVE_PRUNE_SPLIT])
        for key in self.split_info[name][PASSIVE_PRUNE_SPLIT]:
            if int(key[10:]) > ori_begin:
                self.split_info[name][PASSIVE_PRUNE_SPLIT][key] -= len(prune_index)

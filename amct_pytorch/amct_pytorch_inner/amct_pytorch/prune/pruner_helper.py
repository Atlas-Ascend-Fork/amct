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
import os
import stat
import threading

from google.protobuf import text_format

import amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer as opt
from ...amct_pytorch.common.optimizer.delete_pass_through_node import DeletePassThroughNodePass
from ...amct_pytorch.common.prune.prune_recorder_helper import PruneRecordHelper
from ...amct_pytorch.common.utils.prune_record_attr_util import AttrProtoHelper
from ...amct_pytorch.parser.parse_record_file import RecordFileParser

from ...amct_pytorch.prune.filter_prune_helper import create_filter_prune_helper
from ...amct_pytorch.prune.regular_prune_model import RegularModelPruner
from ...amct_pytorch.prune.find_prune_index import PuneIndexHelper

from ...amct_pytorch.utils.singleton_record import SingletonScaleOffsetRecord
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.vars import CHANNEL_UNRELATED_ONNX_TYPES
from ...amct_pytorch.utils.log import LOGGER


class PruneHelper:
    """ Helper to do Prune"""
    def __init__(self, graph, input_data, record_file):
        self.graph = graph
        self.record_file = record_file
        self.mode_input_data = input_data
        self.record = SingletonScaleOffsetRecord().record
        self.record_helper = PruneRecordHelper(self.record, self.graph)

    def create_prune_model(self):
        """create prune model"""
        self.preprocess_graph()
        self.simplify_graph()
        self.find_prune_consumers()
        self._find_prune_cout()
        # really prune model
        RegularModelPruner(self.graph, self.record, self.mode_input_data).do_prune()

        self.save_record()

    def restore_prune_model(self):
        """restore prune model"""
        # parse record_file
        lock = threading.Lock()
        lock.acquire()
        record_parser = RecordFileParser(self.record_file, self.graph, '', enable_quant=False, enable_prune=True)
        if record_parser.is_records_empty():
            LOGGER.logw(
                "record_file is empty, no layers to be pruned. "
            )
        prune_record, _ = record_parser.parse()
        lock.release()
        self.record.prune_record.extend(prune_record)
        # really prune model
        RegularModelPruner(self.graph, self.record, self.mode_input_data).do_prune()


    def preprocess_graph(self):
        """
        Function: preprocess graph to add torch_type for node if it can be found in torch
        Param: None
        Returns: None
        """
        model_helper = ModuleHelper(self.graph.model)
        for node in self.graph.nodes:
            try:
                mod = model_helper.get_module(node.name)
            except RuntimeError:
                pass
            else:
                node.set_attr('torch_type', type(mod).__name__)

    def simplify_graph(self):
        """
        Function: simplify graph before prune.
        param: None
        Return: None
        """
        optimizer = opt.GraphOptimizer()
        optimizer.add_pass(opt.DeleteCircularPaddingPass())
        optimizer.add_pass(opt.ReplaceAvgpoolFlattenPass())
        optimizer.add_pass(opt.ReplaceAvgpoolReshapePass())
        optimizer.add_pass(opt.DeleteResizePass())
        optimizer.add_pass(opt.DeleteLinearAddPass())
        optimizer.add_pass(DeletePassThroughNodePass(CHANNEL_UNRELATED_ONNX_TYPES))
        optimizer.add_pass(opt.DeleteIsolatedNodePass())
        optimizer.do_optimizer(self.graph, self.graph.model)

    def save_record(self):
        """
        Function: save record to file.
        param: None
        Return: None
        """
        lock = threading.Lock()
        lock.acquire()
        file_flags = os.O_WRONLY + os.O_CREAT + os.O_TRUNC
        file_mode = stat.S_IRUSR + stat.S_IWUSR + stat.S_IRGRP
        with os.fdopen(os.open(self.record_file, file_flags, file_mode), 'w',
                   encoding='UTF-8', newline='') as fid:
            fid.write(text_format.MessageToString(self.record,
                                                  as_utf8=True))
        lock.release()

    def find_prune_consumers(self):
        """ find prune producer and consumer"""
        for node in self.graph.nodes + self.graph._in_out_nodes:
            helper = create_filter_prune_helper(node)
            helper.process(self.record_helper)
        self._delete_redundant_record()

    def _delete_redundant_record(self):
        """
        Function: delete redundant content in prune_record
        Param: None
        Return: None
        """
        # delete some attr like prune_axis
        self.record_helper.delete_redundant_attr(['prune_axis'])

        # delete fake node D4toD2
        for prune_record in self.record.prune_record:
            del_consumer = None
            for consumer in prune_record.consumer:
                attr_helper = AttrProtoHelper(consumer)
                consumer_type = attr_helper.get_attr_value('type')
                if consumer_type == 'D4toD2':
                    del_consumer = consumer
                    break
            if del_consumer:
                prune_record.consumer.remove(del_consumer)


    def _find_prune_cout(self):
        """ find prune index"""
        model_helper = ModuleHelper(self.graph.model)
        for prune_record in self.record.prune_record:
            index_helper = PuneIndexHelper(model_helper, prune_record, self.record_helper)
            index_helper.cal_prune_cout()

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
import torch
import numpy as np
from torch.autograd import Function
from google.protobuf import text_format

from ....amct_pytorch.custom_op import selective_mask_gen
from ....amct_pytorch.utils.log import LOGGER
from ....amct_pytorch.common.utils.prune_record_attr_util import AttrProtoHelper
from ....amct_pytorch.common.prune.prune_recorder_helper import PruneRecordHelper
from ....amct_pytorch.utils.singleton_record import SingletonScaleOffsetRecord
from ....amct_pytorch.common.prune.selective_prune_helper_base import SelectivePruneHelperBase
from ....amct_pytorch.configuration import retrain_config as conf
from ....amct_pytorch.utils.vars import SELECTIVE_PRUNABLE_TYPES
from ....amct_pytorch.utils.model_util import ModuleHelper
from ....amct_pytorch.proto import scale_offset_record_pb2

SELECTIVE_PRUNE_CONFIG = {'M4N2': (4, 2), }


class SelectivePruneFunction(Function):
    """
    Function: selective prune operator base class.
    APIs: forward, backward.
    """
    @staticmethod
    def forward(ctx, module_weight, n_out_of_m_type, prune_axis):
        group_size, pruned_size = SELECTIVE_PRUNE_CONFIG.get(n_out_of_m_type)
        return selective_mask_gen(module_weight.detach(), prune_axis, group_size, pruned_size)


def save_record_tofile(record_file, record):
    """
    Function: save record to file.
    param: None
    Return: None
    """
    lock = threading.Lock()
    lock.acquire()
    file_flags = os.O_WRONLY + os.O_CREAT + os.O_TRUNC
    file_mode = stat.S_IRUSR + stat.S_IWUSR + stat.S_IRGRP
    with os.fdopen(os.open(record_file, file_flags, file_mode), 'w',
                encoding='UTF-8', newline='') as fid:
        fid.write(text_format.MessageToString(record, as_utf8=True))
    lock.release()


def add_mask_to_record(layer_name, prune_mask):
    """add prune mask to record"""
    selective_prune_record = SingletonScaleOffsetRecord().prune_record[layer_name]
    attr_helper = AttrProtoHelper(selective_prune_record.selective_prune)
    prune_mask_cpu = (prune_mask != 0).cpu()
    pack_uint8 = np.packbits(prune_mask_cpu)
    if len(pack_uint8) % 8 != 0:
        pack_uint8 = np.append(pack_uint8, np.zeros(8 - len(pack_uint8) % 8, dtype=np.uint8))
    pack_mask = np.frombuffer(pack_uint8.tobytes(), dtype=np.int64)
    attr_helper.set_attr_value('selective_mask', 'INTS', pack_mask)

    record_file = SingletonScaleOffsetRecord().record_file
    if len(SingletonScaleOffsetRecord().record.record) != 0 or \
        len(SingletonScaleOffsetRecord().record.prune_record) != 0:
        save_record_tofile(record_file, SingletonScaleOffsetRecord().record)
    else:
        new_record = scale_offset_record_pb2.ScaleOffsetRecord()
        for _, slt_record in SingletonScaleOffsetRecord().prune_record.items():
            prune_record = new_record.prune_record.add()
            prune_record.CopyFrom(slt_record)
        save_record_tofile(record_file, new_record)


def get_mask_from_record(layer_name):
    """get prune mask from record"""
    selective_prune_record = SingletonScaleOffsetRecord().prune_record[layer_name]
    attr_helper = AttrProtoHelper(selective_prune_record.selective_prune)
    mask_shape = attr_helper.get_attr_value('mask_shape')
    size = 1
    for shape in mask_shape:
        size *= shape
    pack_mask = np.asarray(attr_helper.get_attr_value('selective_mask'))
    pack_mask = np.frombuffer(pack_mask.tobytes(), dtype=np.uint8)
    unpack_mask = np.unpackbits(pack_mask)[:size].reshape(mask_shape)
    return torch.Tensor(unpack_mask)


def create_selective_prune_record(graph):
    """
    Function:
        create selective prune record to records
    """
    records = SingletonScaleOffsetRecord().record
    record_helper = PruneRecordHelper(records, graph)
    model_helper = ModuleHelper(graph.model)
    for node in graph.nodes:
        if model_helper.named_module_dict.get(node.name):
            mod = model_helper.get_module(node.name)
            if type(mod).__name__ in SELECTIVE_PRUNABLE_TYPES:
                node.set_attr('wgt_shape', list(mod.weight.shape))
            node.set_attr('torch_type', type(mod).__name__)
        else:
            node.set_attr('torch_type', 'empty')
        if not SelectivePruneHelper.match_pattern(node):
            continue
        operator = SelectivePruneHelper(node)
        operator.process(record_helper)
        SingletonScaleOffsetRecord().prune_record[node.name] = \
            node.get_attr('selective_prune_records')[0]

    record_file = SingletonScaleOffsetRecord().record_file
    save_record_tofile(record_file, SingletonScaleOffsetRecord().record)


def restore_selective_prune_record():
    """
    Function:
        restore selective prune record to records
    """
    records = SingletonScaleOffsetRecord().record
    selective_prune_record = SingletonScaleOffsetRecord().prune_record
    for record in records.prune_record:
        if record.HasField('selective_prune'):
            selective_prune_record[record.selective_prune.name] = record


class SelectivePruneHelper(SelectivePruneHelperBase):
    """SelectivePruneHelper for active type node"""

    @staticmethod
    def match_pattern(node):
        """
        Function: Match pattern of select prune for node in graph
        Parameter: None
        Return: bool, matched or not
        """
        node_type = node.get_attr('torch_type')
        if node_type not in SELECTIVE_PRUNABLE_TYPES:
            return False
        if not conf.RetrainConfig().selective_prune_enable(node.name):
            return False
        return True

    def get_mask_shape(self):
        """
        Function: get the length of cout of self.node
        Param: None
        Return: a number
        """
        return self.node.get_attr('wgt_shape')

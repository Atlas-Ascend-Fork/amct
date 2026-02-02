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
"""
Operation functions for scale and offset record file
"""
from __future__ import print_function
import os
import sys
import numpy as np

from google.protobuf import text_format
from amct_pytorch.amct_pytorch_inner.amct_pytorch.proto import scale_offset_record_pb2
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.quant_node import QuantOpInfo


def create_file(record_file,
                layers_name,
                err_layers=None,
                scale_value=0.01,
                offset_value=0,
                graph=None,
                channel_wise=False,
                record_weight=[True, True],
                record_data=[True, True],
                err_length=False,
                unmatch_length=[False, False],
                skip_n_layers=None,
                skip_fusion_layers=None,
                no_fusion_layers=None):
    if err_layers is None:
        err_layers = []
    if skip_n_layers is None:
        skip_n_layers = []
    if skip_fusion_layers is None:
        skip_fusion_layers = []
    if no_fusion_layers is None:
        no_fusion_layers = []
    if graph is None:
        err_layers += layers_name
        layers_name = []

    layers_length = dict()
    for layer_name in layers_name:
        compute_op = graph.get_node_by_name(layer_name)
        _, scale_length_except = QuantOpInfo.get_scale_shape(
            compute_op, channel_wise)
        layers_length[layer_name] = scale_length_except

    generate_file(record_file, layers_length, err_layers, scale_value,
                  offset_value, channel_wise, record_weight, record_data,
                  err_length, unmatch_length, skip_n_layers,
                  skip_fusion_layers, no_fusion_layers)


def generate_file(record_file,
                  layers_length,
                  err_layers=None,
                  scale_value=0.01,
                  offset_value=0,
                  channel_wise=False,
                  record_weight=[True, True],
                  record_data=[True, True],
                  err_length=False,
                  unmatch_length=[False, False],
                  skip_n_layers=None,
                  skip_fusion_layers=None,
                  no_fusion_layers=None):
    if err_layers is None:
        err_layers = []
    if skip_n_layers is None:
        skip_n_layers = []
    if skip_fusion_layers is None:
        skip_fusion_layers = []
    if no_fusion_layers is None:
        no_fusion_layers = []

    records = scale_offset_record_pb2.ScaleOffsetRecord()
    for layer_name in layers_length:
        if record_weight:
            scale_length_except = layers_length[layer_name]
            if err_length:
                scale_length_except += 1
            scale = [scale_value] * scale_length_except
            if unmatch_length[0]:
                offset_length_except = scale_length_except + 1
            else:
                offset_length_except = scale_length_except
            offset = [offset_value] * offset_length_except
            record_weights_scale_offset(records, layer_name, scale, offset,
                                        record_weight)
        if record_data:
            scale = scale_value
            offset = offset_value
            record_data_scale_offset(records, layer_name, scale, offset,
                                     record_data)
        if layer_name not in skip_n_layers:
            scale_length_except = layers_length[layer_name]
            if err_length:
                scale_length_except += 1
            if unmatch_length[1]:
                n_length_except = scale_length_except + 1
            else:
                n_length_except = scale_length_except
        if layer_name not in no_fusion_layers:
            if layer_name in skip_fusion_layers:
                record_do_fusion(records, layer_name, True)
            else:
                record_do_fusion(records, layer_name, False)

    for layer_name in err_layers:
        if record_weight:
            scale = [scale_value] * 3
            offset = [offset_value] * 3
            record_weights_scale_offset(records, layer_name, scale, offset,
                                        record_weight)
        if record_data:
            scale = scale_value
            offset = offset_value
            record_data_scale_offset(records, layer_name, scale, offset,
                                     record_data)

    file_realpath = os.path.realpath(record_file)
    file_dir = os.path.split(file_realpath)
    if not os.path.isdir(file_dir[0]):
        os.makedirs(file_dir[0])
    with open(file_realpath, 'w') as file:
        file.write(text_format.MessageToString(records, as_utf8=True))


def record_weights_scale_offset(records,
                                layer_name,
                                scale,
                                offset,
                                is_record=[True, True]):
    """
    Function: Write scale_w and offset_w to record file
    Parameters: records: ScaleOffsetRecord() object to write
                layer_name: layer name of scale_w, offset_w
                scale: vector of scale_w
                offset: vector of offset_w
    Return: None
    """
    done_flag = False
    for record in records.record:
        if record.key == layer_name:
            if is_record[0]:
                record.value.scale_w.extend(scale)
            if is_record[1]:
                record.value.offset_w.extend(offset)
            done_flag = True
            break
    if not done_flag:
        record = records.record.add()
        record.key = layer_name
        if is_record[0]:
            record.value.scale_w.extend(scale)
        if is_record[1]:
            record.value.offset_w.extend(offset)


def record_data_scale_offset(records,
                             layer_name,
                             scale,
                             offset,
                             is_record=[True, True]):
    """
    Function: Write scale_w and offset_w to record file
    Parameters: records: ScaleOffsetRecord() object to write
                layer_name: layer name of scale_w, offset_w
                scale: vector of scale_w
                offset: vector of offset_w
    Return: None
    """
    done_flag = False
    for record in records.record:
        if record.key == layer_name:
            if is_record[0]:
                record.value.scale_d = scale
            if is_record[1]:
                record.value.offset_d = offset
            done_flag = True
            break
    if not done_flag:
        record = records.record.add()
        record.key = layer_name
        if is_record[0]:
            record.value.scale_d = scale
        if is_record[1]:
            record.value.offset_d = offset


def record_do_fusion(records, layer_name, skip_fusion, is_record=True):
    """
    Function: Write scale_w and offset_w to record file
    Parameters: records: ScaleOffsetRecord() object to write
                layer_name: layer name of scale_w, offset_w
                scale: vector of scale_w
                offset: vector of offset_w
    Return: None
    """
    done_flag = False
    for record in records.record:
        if record.key == layer_name:
            if is_record:
                record.value.skip_fusion = skip_fusion
            done_flag = True
            break
    if not done_flag:
        record = records.record.add()
        record.key = layer_name
        if is_record:
            record.value.skip_fusion = skip_fusion


def read_weights_scale_offset(records, layer_name):
    """
    Function: Read scale_w and offset_w from record file
    Parameters: records: ScaleOffsetRecord() object to read
                layer_name: layer name of scale_w, offset_w
    Return: scale: vector of scale_w
            offset: vector of offset_w
    """
    done_flag = False
    scale = []
    offset = []
    for record in records.record:
        if record.key == layer_name:
            # Read scale_w from record file
            if not record.value.scale_w:
                raise RuntimeError("Cannot find scale_w of layer '{}' " \
                    "in record file".format(layer_name))
            scale.extend(record.value.scale_w)
            # Read offset_w from record file
            if not record.value.offset_w:
                raise RuntimeError("Cannot find offset_w of layer \'{}\' " \
                    "in record file".format(layer_name))
            offset.extend(record.value.offset_w)
            done_flag = True
            break
    if not done_flag:
        raise RuntimeError("Cannot find layer '{}' in record " \
            "file".format(layer_name))
    return scale, offset


def read_activation_scale_offset(records, layer_name):
    """
    Function: Read scale_d and offset_d from record file
    Parameters: records: ScaleOffsetRecord() object to read
                layer_name: layer name of scale_d, offset_d
    Return: scale: scalar of scale_d
            offset: scalar of offset_d
    """
    done_flag = False
    scale = 1
    offset = 0
    for record in records.record:
        if record.key == layer_name:
            # Read scale_d from record file
            if not record.value.HasField('scale_d'):
                raise RuntimeError("Cannot find scale_d of layer '{}' " \
                    "in record file".format(layer_name))
            scale = record.value.scale_d
            # Read offset_d from record file
            if not record.value.HasField('offset_d'):
                raise RuntimeError("Cannot find offset_d of layer '{}' " \
                    "in record file".format(layer_name))
            offset = record.value.offset_d
            done_flag = True
            break
    if not done_flag:
        raise RuntimeError("Cannot find layer '{}' in record " \
            "file".format(layer_name))
    return scale, offset


def generate_records(layers_length, scale_value=0.1, offset_value=0):
    records = {}
    for layer_name in layers_length:
        length = layers_length[layer_name]
        record = dict()
        record['data_scale'] = np.array(scale_value, dtype=np.float32)
        record['data_offset'] = np.array(offset_value, dtype=np.int8)
        record['weight_scale'] = np.array([scale_value]*length, dtype=np.float32)
        record['weight_offset'] = np.array([offset_value]*length, dtype=np.int8)

        records[layer_name] = record

    return records


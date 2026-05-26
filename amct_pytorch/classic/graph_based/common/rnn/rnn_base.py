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
from onnx import onnx_pb

from ..utils.onnx_node_util import AttributeProtoHelper
from ..utils.vars_util import INT8, INT16
from ..utils.vars_util import DT_INT8, DT_INT16


class RnnQuantOpBase:
    """
    Function: define base op of rnn quantized op
    APIs: construct_node_proto
    """

    required_attrs = list()
    ''' attr_name, (attr_type, default_value) '''
    attrs = dict()

    ''' attr_name, (attr_type, default_value) '''
    quant_sqrt_mode_attrs = dict()

    ''' name in record, (attr_name, attr_type) '''
    quant_attrs = dict()

    quant_dtype_map = {
        INT8: DT_INT8,
        INT16: DT_INT16
    }

    @classmethod
    def construct_node_proto(cls, ori_proto, records):
        """
        Function: construct rnn quantized op node in onnx
        Parameters:
            ori_proto: proto of orignal RNN node
            records: dict including quant factors
        Return:
            node_proto: proto of rnn quantized op
        """
        node_proto = onnx_pb.NodeProto()
        node_proto.name = ori_proto.name
        node_proto.op_type = cls.__name__
        node_proto.input.extend(ori_proto.input)
        node_proto.output.extend(ori_proto.output)

        added_attrs = set()
        for attr in ori_proto.attribute:
            if attr.name in cls.required_attrs:
                node_proto.attribute.append(attr)
            if attr.name in cls.attrs.keys():
                node_proto.attribute.append(attr)
                # record attrs added
                added_attrs.add(attr.name)

        attr_helper = AttributeProtoHelper(node_proto)
        # set default value to other attrs
        default_attrs = set(cls.attrs.keys()) - added_attrs
        for attr_name in default_attrs:
            attr_helper.set_attr_value(
                attr_name, cls.attrs[attr_name][0], cls.attrs[attr_name][1])

        record = records.get(ori_proto.name)

        for attr_name, (attr_type, attr_value) in cls.quant_sqrt_mode_attrs.items():
            attr_helper.set_attr_value(attr_name, attr_type, attr_value)

        for quant_factor in cls.quant_attrs.keys():
            if quant_factor == 'act_type':
                quant_value = cls.quant_dtype_map.get(record.get(quant_factor))
            elif 'scale' in quant_factor:
                quant_value = 1 / record.get(quant_factor)
            else:
                quant_value = record.get(quant_factor)
            attr_helper.set_attr_value(
                cls.quant_attrs[quant_factor][0], cls.quant_attrs[quant_factor][1], quant_value)

        return node_proto

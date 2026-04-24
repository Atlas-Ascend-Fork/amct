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
from onnx.onnx_pb import AttributeProto


class AttributeProtoHelper():
    """docstring for AttributeProtoHelper"""
    attribute_type = AttributeProto.AttributeType
    attribute_type_map = {
        # type_string ,type_proto_value, location
        'FLOAT': [attribute_type.FLOAT, 'f'],
        'INT': [attribute_type.INT, 'i'],
        'STRING': [attribute_type.STRING, 's'],
        'TENSOR': [attribute_type.TENSOR, 't'],
        'FLOATS': [attribute_type.FLOATS, 'floats'],
        'INTS': [attribute_type.INTS, 'ints'],
        'STRINGS': [attribute_type.STRINGS, 'strings'],
        'TENSORS': [attribute_type.TENSORS, 'tensors'],
        'GRAPH': [attribute_type.GRAPH, 'g'],
        'GRAPHS': [attribute_type.GRAPHS, 'graphs'],
        'SPARSE_TENSOR': [attribute_type.SPARSE_TENSOR, 'sparse_tensor'],
        'SPARSE_TENSORS': [attribute_type.SPARSE_TENSORS, 'sparse_tensors'],
        'UNDEFINED': [attribute_type.UNDEFINED, None],
    }
    value_location_id = 1
    proto_value_id = 0
    ge_dtype_map = {'FLOAT32': 0, 'FLOAT16': 1}

    def __init__(self, node_proto):
        """
        Function: init object
        Inputs:
            node_proto: onnx.AttributeProto
        Returns: None
        """
        super().__init__()
        self.node_proto = node_proto

    @classmethod
    def map_value_location(cls, proto_value):
        """
        Funtion: find value's location in node_proto with proto_value
        Inputs:
            proto_value: a number in [0, 12], means the location as
            attribute_type_map
        Returns:
            location: a string, where the value locate as attribute_type_map
        """
        for type_str in cls.attribute_type_map:
            value = cls.attribute_type_map[type_str]
            if proto_value == value[0]:
                return value[1]
        raise ValueError('The type{%s} of attr is UNEXCEPTED' % (proto_value))

    def has_attr(self, attr_name):
        """
        Function: Whether node has attr of attr_name
        Inputs:
            attr_name: string, name of sttr
        Returns:
            True/False: True means attr exists and False otherwise
        """
        for attribute in self.node_proto.attribute:
            if attribute.name == attr_name:
                return True

        return False

    def get_attr_value(self, attr_name):
        """
        Function: get attr's value
        Inputs:
            attr_name: string, name of attr
        Returns:
            attr_value: attr's value
        """
        for attr in self.node_proto.attribute:
            if attr.name != attr_name:
                continue
            attr_value = getattr(attr, self.map_value_location(attr.type))
            return attr_value
        raise RuntimeError("node %s has no attribute %s" %
                           (self.node_proto.name, attr_name))

    def set_attr_value(self, attr_name, type_string, value):
        """
        Function: Set attr
        Inputs:
            attr_name: string, name of attr
            type_string: string, indicate type
            value: number or lsit, attr's data
        Returns:
            None
        """
        attr_type = self.attribute_type_map[type_string][self.proto_value_id]
        target_attribute = None
        for attr in self.node_proto.attribute:
            if attr.name == attr_name:
                target_attribute = attr
        if target_attribute is None:
            target_attribute = self.node_proto.attribute.add()
            target_attribute.name = attr_name

        target_attribute.type = attr_type
        value_location = self.map_value_location(target_attribute.type)

        if value_location in ['f', 'i', 's']:
            # set value in f/i/s
            setattr(target_attribute, value_location, value)
        elif value_location in ['floats', 'ints', 'strings']:
            # set value in floats/ints/strings
            target_attribute.ClearField(value_location)
            getattr(target_attribute, value_location).extend(value)
        else:
            # add an attr without value
            return
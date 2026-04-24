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

from ...proto.basic_info_pb2 import AttrProto


class AttrProtoHelper():
    """docstring for AttrProtoHelper"""
    attr = AttrProto.AttrType
    attr_map = {
        # type_string ,type_proto_value, location
        'UNDEFINED': [attr.UNDEFINED, None],
        'FLOAT': [attr.FLOAT, 'f'],
        'INT': [attr.INT, 'i'],
        'STRING': [attr.STRING, 's'],
        'FLOATS': [attr.FLOATS, 'floats'],
        'INTS': [attr.INTS, 'ints'],
        'STRINGS': [attr.STRINGS, 'strings']
    }
    proto_value_id = 0
    value_location_id = 1

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
        attr_map
        Returns:
        location: a string, where the value locate as attr_map
        """
        for key in cls.attr_map:
            value = cls.attr_map[key]
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
        for attr in self.node_proto.attr:
            if attr.name != attr_name:
                continue
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
        for attr in self.node_proto.attr:
            if attr.name != attr_name:
                continue
            attr_value = getattr(attr,
                                 self.map_value_location(attr.type))
            if attr.type == AttrProto.AttrType.STRINGS:
                attr_value = [byte_string.decode('utf-8') \
                    for byte_string in attr_value]
            elif attr.type == AttrProto.AttrType.STRING:
                attr_value = attr_value.decode('utf-8')

            return attr_value
        raise RuntimeError("node %s has no attr %s" %
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
        attr_type = self.attr_map.get(type_string)[self.proto_value_id]
        target_attr = None
        for attr in self.node_proto.attr:
            if attr.name == attr_name:
                target_attr = attr
        if target_attr is None:
            target_attr = self.node_proto.attr.add()
            target_attr.name = attr_name

        target_attr.type = attr_type
        value_location = self.map_value_location(target_attr.type)
        if value_location in ['f', 'i', 's']:
            # set value in f/i/s
            if type_string == "STRING":
                value = bytes(value, 'utf-8')
            setattr(target_attr, value_location, value)
        else:
            # set value in floats/ints/strings
            if type_string == "STRINGS":
                value = [bytes(val, 'utf-8') for val in value]
            target_attr.ClearField(value_location)
            getattr(target_attr, value_location).extend(value)

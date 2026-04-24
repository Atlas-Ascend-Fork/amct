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
from copy import deepcopy


class NodeBase():
    """
    Function: Data structure of node which contains LayerParameter info
    APIs: is_data_node, index, set_index, name, type, layer, get_input_anchor,
          get_input_anchors, get_output_anchor, get_output_anchors,
          get_data, get_all_data, set_data, set_all_data, add_data,
          dump_proto
    """
    def __init__(self, node_id, node_index, node_proto):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        self._basic_info = {'node_index': node_index,
                            'name': node_id,
                            'type': ''}
        self._input_anchors = []
        self._output_anchors = []
        self._attrs = {}
        self._node_proto = node_proto

    @property
    def is_data_node(self):
        """current node is data node"""
        return False

    @property
    def index(self):
        """
        Function: Get index of current node
        Parameter: None
        Return: node index
        """
        return self._basic_info.get('node_index')

    @property
    def name(self):
        """
        Function: Get name of current node
        Parameter: None
        Return: Name of node
        """
        return self._basic_info.get('name')

    @property
    def type(self):
        """
        Function: Get type of current node
        Parameter: None
        Return: Type of node
        """
        return self._basic_info.get('type')

    @property
    def proto(self):
        """
        Function: Get nodeParameter of current node
        Parameter: None
        Return: nodeParameter of node
        """
        return self._node_proto

    @property
    def attrs(self):
        """
        Function: Get all attrs from node
        Parameter: None
        Return: None
        """
        return self._attrs

    @property
    def input_anchors(self):
        """
        Function: Get current node's all input anchors
        Parameter: None
        Return: Input anchors
        """
        return self._input_anchors

    @property
    def output_anchors(self):
        """
        Function: Get current node's all output anchors
        Parameter: None
        Return: Output anchors
        """
        return self._output_anchors

    @property
    def is_isolated(self):
        """
        Function: whether the node is isolated. Node is isolated is it has no any producer and consuemr
        Parameter: None
        Return: True or False
        """
        for input_anchor in self.input_anchors:
            if input_anchor.get_peer_output_anchor() is not None:
                return False
        for output_anchor in self.output_anchors:
            if output_anchor.get_peer_input_anchor():
                return False

        return True

    def set_attr(self, key, value):
        """
        Function: Set attr of 'key' with 'value' to node
        Parameter: key: name of attr to set
                   value: value of attr to set
        Return: None
        """
        self._attrs[key] = value

    def set_attrs(self, attrs):
        """
        Function: Set list of attrs to node
        Parameter: attrs: list of attrs
        Return: None
        """
        self._attrs = deepcopy(attrs)

    def has_attr(self, key):
        """
        Function: Check whether node have attr of 'key'
        Parameter: key: name of attr to find
        Return: None
        """
        if key in self._attrs:
            return True
        return False

    def get_attr(self, key):
        """
        Function: Get attr of 'key' from node
        Parameter: key: name of attr to find
        Return: None
        """
        if key not in self._attrs:
            raise RuntimeError('Cannot find {} in node {}'.format(
                key,
                self._basic_info.get('name')))
        return self._attrs[key]

    def delete_attr(self, key):
        """
        Function: Delete attr of 'key' from node
        Parameter: key: name of attr to find
        Return: None
        """
        if key not in self._attrs:
            raise RuntimeError('Cannot find {} in node {}'.format(
                key,
                self._basic_info.get('name')))
        del self._attrs[key]

    def set_index(self, node_index):
        """
        Function: set index of current node
        Parameter: None
        Return: None
        """
        self._basic_info['node_index'] = node_index

    def get_input_anchor(self, index):
        """
        Function: Get input anchor by index
        Parameter: None
        Return: Input anchor
        """
        if index >= len(self._input_anchors):
            raise RuntimeError('Node:{} get {} input out of range'.format(
                self._basic_info.get('name'), index))
        return self._input_anchors[index]

    def get_output_anchor(self, index):
        """
        Function: Get output anchor by index
        Parameter: None
        Return: Output anchor
        """
        if index >= len(self._output_anchors):
            raise RuntimeError('Node:{} get {} output out of range'.format(
                self._basic_info.get('name'), index))
        return self._output_anchors[index]

    def get_producer(self, input_index):
        """
        Function: Get current node's producer on input[input_index]
        Parameter: input_index, int, indicating which input
        Return: producer, Node
        """
        input_anchor = self.get_input_anchor(input_index)
        peer_anchor = input_anchor.get_peer_output_anchor()
        if peer_anchor is None:
            return None, None
        producer = peer_anchor.node
        out_idx = peer_anchor.index
        return producer, out_idx

    def get_consumers(self, output_index):
        """
        Function: Get current node's consumers on output[output_index]
        Parameter: output_index, int, indicating which output
        Return: consumers, list of node; if has no consumers, empty list
        """
        output_anchor = self.get_output_anchor(output_index)
        peer_anchors = output_anchor.get_peer_input_anchor()
        consumers = [peer_anchor.node for peer_anchor in peer_anchors]
        in_idx = [peer_anchor.index for peer_anchor in peer_anchors]
        return consumers, in_idx

    def dump_proto(self):
        """
        Function: Dump node to proto object
        Parameter: None
        Return: proto object
        """
        raise NotImplementedError


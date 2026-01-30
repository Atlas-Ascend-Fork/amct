#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Data structure of data node which contains input info

"""


class DataNodeBase():
    """
    Function: Data structure of data node which contains input info
    APIs: is_data_node, index, name, get_output_anchor, get_output_anchors
    """
    def __init__(self, index, data_name):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        self._index = index
        self._data_name = data_name
        self._output_anchors = []
        self._type = 'DataNode'

    @property
    def is_data_node(self):
        """current node is data node"""
        return True

    @property
    def index(self):
        """
        Function: Get index of current node
        Parameter: None
        Return: node index
        """
        return self._index

    @property
    def name(self):
        """
        Function: Get name of current node
        Parameter: None
        Return: Name of node
        """
        return self._data_name

    @property
    def type(self):
        """
        Function: Get type of current node
        Parameter: None
        Return: type of node
        """
        return self._type

    @property
    def output_anchors(self):
        """
        Function: Get current node's all output anchors
        Parameter: None
        Return: Output anchors
        """
        return self._output_anchors

    def get_output_anchor(self, index):
        """
        Function: Get output anchor by index
        Parameter: None
        Return: Output anchor
        """
        if index >= len(self._output_anchors):
            raise RuntimeError('Node:{} get {} output out of range'.format(
                self._data_name, index))
        return self._output_anchors[index]

    def get_consumers(self, output_index):
        """
        Function: Get current node's consumers on output[output_index]
        Parameter: output_index, int, indicating which output
        Return: consumers, list of node
        """
        output_anchor = self.get_output_anchor(output_index)
        peer_anchors = output_anchor.get_peer_input_anchor()
        consumers = [peer_anchor.node for peer_anchor in peer_anchors]
        in_idx = [peer_anchor.index for peer_anchor in peer_anchors]
        return consumers, in_idx

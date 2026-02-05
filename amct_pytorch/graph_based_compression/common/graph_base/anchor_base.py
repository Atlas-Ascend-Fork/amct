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


class InputAnchorBase():
    """
    Function: Data structure of anchor which contains graph topologic info
    APIs: node, index, name, set_name, add_link, del_link,
          get_peer_output_anchor
    """
    def __init__(self, node, index, anchor_name):
        """
        Function: init object
        Parameter: node: Node that the InputAnchorBase add to
                   index: index of InputAnchorBase in node
                   anchor_name: InputAnchorBase name as named in layer.bottom
        Return: None
        """
        self._attach_node = node
        self._index = index
        self._anchor_name = anchor_name
        self.__peer_anchor = None

    def __repr__(self):
        anchor_info = '< index: {}, name: {} >'.format(
            self._index, self._anchor_name)
        return anchor_info

    @property
    def node(self):
        """
        Function: Return current anchor's attached node
        Parameter: None
        Return: Node
        """
        return self._attach_node

    @property
    def index(self):
        """
        Function: Return current anchor's index
        Parameter: None
        Return: Node index
        """
        return self._index

    @property
    def name(self):
        """
        Function: Return current anchor's name
        Parameter: None
        Return: Node name
        """
        return self._anchor_name

    def set_name(self, name):
        """
        Function: Set current anchor's name
        Parameter: None
        Return: None
        """
        self._anchor_name = name

    def add_link(self, output_anchor: 'OutputAnchorBase'):
        """
        Function: Add link from current anchor to output_anchor
        Parameter: None
        Return: None
        """
        self.__peer_anchor = output_anchor

    def del_link(self):
        """
        Function: Delete link from current anchor to output_anchor
        Parameter: None
        Return: None
        """
        self.__peer_anchor = None

    def get_peer_output_anchor(self):
        """
        Function: Get current anchor's peer output_anchor
        Parameter: None
        Return: OutputAnchorBase
        """
        return self.__peer_anchor


class OutputAnchorBase():
    """
    Function: Data structure of anchor which contains graph topologic info
    APIs: node, index, name, set_name, add_link, del_link,
          get_peer_input_anchor, get_reused_info
    """
    def __init__(self, node, index, anchor_name):
        """
        Function: init object
        Parameter: node: Node that the OutputAnchorBase add to
                   index: index of OutputAnchorBase in node
                   anchor_name: OutputAnchorBase name as named in layer.top
                   reused_input_index: If 'None', means this output will malloc
                                       a new blob in caffe, otherwise will
                                       reuse input[reused_input_index] blob

        Return: None
        """
        self._attach_node = node
        self._index = index
        self._anchor_name = anchor_name
        self._peer_anchors = []

    def __repr__(self):
        anchor_info = '< index: {}, name: {} >'.format(
            self._index, self._anchor_name)
        return anchor_info

    @property
    def index(self):
        """
        Function: Return current anchor's index
        Parameter: None
        Return: Node index
        """
        return self._index

    @property
    def name(self):
        """
        Function: Return current anchor's name
        Parameter: None
        Return: Node name
        """
        return self._anchor_name

    @property
    def node(self):
        """
        Function: Return current anchor's attached node
        Parameter: None
        Return: Node
        """
        return self._attach_node

    def set_name(self, name):
        """
        Function: Set current anchor's name
        Parameter: None
        Return: None
        """
        self._anchor_name = name

    def add_link(self, input_anchor: 'InputAnchorBase'):
        """
        Function: Add link from input_anchor to current anchor
        Parameter: None
        Return: None
        """
        self._peer_anchors.append(input_anchor)

    def del_link(self, src_anchor: 'InputAnchorBase'):
        """
        Function: Delete link from input_anchor to current anchor
        Parameter: None
        Return: None
        """
        self._peer_anchors.remove(src_anchor)

    def get_peer_input_anchor(self):
        """
        Function: Get current anchor's peer input_anchor
        Parameter: None
        Return: OutputAnchorBase
        """
        return self._peer_anchors

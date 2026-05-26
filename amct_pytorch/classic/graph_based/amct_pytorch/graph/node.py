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
from onnx import onnx_pb # pylint: disable=import-error
from ...amct_pytorch.common.graph_base.node_base import NodeBase
from ...amct_pytorch.graph.anchor import InputAnchor
from ...amct_pytorch.graph.anchor import OutputAnchor

TYPE = 'type'


class Node(NodeBase): # pylint: disable=no-member
    """
    Function: Data structure of node which contains nodeParameter info
    APIs: is_data_node, index, set_index, name, type, node, get_input_anchor,
          get_input_anchors, get_output_anchor, get_output_anchors,
          get_data, get_all_data, set_data, set_all_data, add_data,
          dump_proto
    """
    def __init__(self, node_index, node_proto, model_path=None):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        if isinstance(node_proto, onnx_pb.SparseTensorProto):
            if node_proto.HasField('values'):
                node_id = node_proto.values.name
            elif node_proto.HasField('indices'):
                node_id = node_proto.indices.name
            else:
                raise RuntimeError('Cannot find tensor in sparse_initializer')
        else:
            node_id = node_proto.name
        super().__init__(node_id, node_index, node_proto)
        self._basic_info['ori_name'] = node_id
        # Init inputs, outputs info form node
        self._trans_type()
        self._init()

        # clear intput and outputs
        if isinstance(self._node_proto, onnx_pb.NodeProto):
            self._node_proto.ClearField('input')
            self._node_proto.ClearField('output')
        self.model_path = model_path

    def __repr__(self):
        return '< {}, name: {}, inputs: {}, outputs:{}>'.format(
            self._basic_info.get('ori_name'),
            self._basic_info.get('name'),
            [x.name for x in self._input_anchors],
            [x.name for x in self._output_anchors])

    @property
    def ori_name(self):
        """get ori_name from basic info"""
        return self._basic_info.get('ori_name')

    @property
    def module_name(self):
        """get module_name from basic info"""
        return self._basic_info.get('module_name')

    def set_name(self, name):
        """Set node name"""
        self._basic_info['name'] = name

    def set_module_name(self, name):
        """Set module name of node"""
        self._basic_info['module_name'] = name

    def add_input_anchor(self, name):
        """
        Function: Add input anchor to current node
        Parameter: name: Added anchor's name
        Return: None
        """
        if self._basic_info.get(TYPE) in ('initializer', 'sparse_initializer'):
            raise RuntimeError('Cannot add input anchor to data node: {}'.format(
                self._basic_info.get('name')))
        index = len(self._input_anchors)
        if index > 0 and self._basic_info.get(TYPE) == 'graph_anchor':
            raise RuntimeError('Can only add one anchor to graph inout: {}'.format(
                self._basic_info.get('name')))
        self._input_anchors.append(InputAnchor(self, index, name))

    def get_output_anchor_index(self, name):
        """Return output anchor index accord to name specify"""
        for index, output_anchor in enumerate(self._output_anchors):
            if output_anchor.name == name:
                return index
        raise ValueError('Not output anchor named {}'.format(name))

    def add_output_anchor(self, name):
        """Add output anchor to current node"""
        index = len(self._output_anchors)
        if index > 0 and self._basic_info.get(TYPE) in (
                'graph_anchor', 'initializer', 'sparse_initializer'):
            raise RuntimeError('Can only add one anchor to node: {}'.format(
                self._basic_info.get('name')))
        self._output_anchors.append(OutputAnchor(self, index, name))

    def get_output_anchor_by_name(self, name):
        """Get output anchor by anchor name"""
        for output_anchor in self._output_anchors:
            if output_anchor.name == name:
                return output_anchor
        raise ValueError('No output anchor named {}'.format(name))

    def dump_proto(self):
        """
        Function: Dump current node to onnx_pb.NodeProto format
        """
        # update node
        if isinstance(self._node_proto, onnx_pb.NodeProto):
            return self.__update_node_proto()

        # update initializer
        if isinstance(self._node_proto, onnx_pb.TensorProto):
            node_proto = onnx_pb.TensorProto()
            node_proto.CopyFrom(self._node_proto)
            return node_proto
        # update sparse_initializer
        if isinstance(self._node_proto, onnx_pb.SparseTensorProto):
            node_proto = onnx_pb.SparseTensorProto()
            node_proto.CopyFrom(self._node_proto)
            return node_proto
        # update graph input/output
        if isinstance(self._node_proto, onnx_pb.ValueInfoProto):
            if self._input_anchors and self._output_anchors:
                raise RuntimeError('Graph input/output {} cannot have both ' \
                    'input, output'.format(self._basic_info.get('ori_name')))

            if self._input_anchors:
                peer_anchor = self._input_anchors[0].get_peer_output_anchor()
                if peer_anchor.name != self.name:
                    raise RuntimeError('Cannot change output name "{}"- ' \
                        '!= >"{}"'.format(peer_anchor.name, self.name))

            node_proto = onnx_pb.ValueInfoProto()
            node_proto.CopyFrom(self._node_proto)
            return node_proto
        raise TypeError("Unexpected node_proto type:\"{}\"! only [NodeProto,"
                        " TensorProto, SparseTensorProto ValueInfoProto] are "
                        "supported.".format(type(self._node_proto)))

    def _init(self):
        """Parse node from onnx_pb proto define
        """
        if isinstance(self._node_proto, onnx_pb.NodeProto):
            # init input anchors
            for index, input_name in enumerate(self._node_proto.input):
                self._input_anchors.append(InputAnchor(self, index,
                                                       input_name))
            # init output anchors
            for index, output_name in enumerate(self._node_proto.output):
                self._output_anchors.append(OutputAnchor(self, index,
                                                         output_name))
        elif isinstance(self._node_proto,
                        (onnx_pb.TensorProto, onnx_pb.SparseTensorProto)):
            self._output_anchors.append(OutputAnchor(self, 0,
                                                     self.name))

    def _trans_type(self):
        """translate node type to unique
        """
        if isinstance(self._node_proto, onnx_pb.NodeProto):
            self._basic_info[TYPE] = self._node_proto.op_type
        elif isinstance(self._node_proto, onnx_pb.ValueInfoProto):
            self._basic_info[TYPE] = 'graph_anchor'
        elif isinstance(self._node_proto, onnx_pb.TensorProto):
            self._basic_info[TYPE] = 'initializer'
        elif isinstance(self._node_proto, onnx_pb.SparseTensorProto):
            self._basic_info[TYPE] = 'sparse_initializer'
        else:
            raise TypeError('Unsupported type:%s' % (type(self._node_proto)))

    def __update_node_proto(self):
        """
        Function: update onnx_pb.NodeProto
        """
        node_proto = onnx_pb.NodeProto()
        node_proto.CopyFrom(self._node_proto)
        node_proto.name = self.name
        # Add input
        for input_anchor in self._input_anchors:
            if input_anchor.get_peer_output_anchor() is not None:
                peer_node = input_anchor.get_peer_output_anchor().node
                index = input_anchor.get_peer_output_anchor().index

                if index >= len(peer_node.output_anchors):
                    raise RuntimeError('Get {} output from {} failed, ' \
                        'out of range'.format(index, peer_node.name))
                peer_name = peer_node.get_output_anchor(index).name
                input_anchor.set_name(peer_name)
            else:
                peer_name = input_anchor.name
            # keep input_anchor same with peer_anchor
            node_proto.input.append(peer_name)

        # add output
        for output_anchor in self._output_anchors:
            out_name = output_anchor.name
            node_proto.output.append(out_name)

        return node_proto
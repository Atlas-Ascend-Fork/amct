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

from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.common.graph_base.graph_base import GraphBase
from ...amct_pytorch.graph.node import Node
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE

MODULE_NAME = 'Graph'


class Graph(GraphBase): # pylint: disable=no-member
    """
    Function: Data structure of graph IR
    APIs: add_node, remove_node, net, dump_proto
    """
    def __init__(self, network, model_path=''):
        super().__init__(network)
        self.model_path = model_path
        self._net = network
        self._init_graph()
        self.topologic_sort()
        self.model = None

    @property
    def net(self):
        """
        Function: Return onnx_pb.ModelProto of graph
        """
        return self._net

    @staticmethod
    def _parse_circular_padding_nodes(quant_identity_node):
        """
        Function: apply bfs to find the conv layer from the circular_padding_nodes
        Parameter: quant_identity_node: root nodes quant_identity
        Return: consumer the conv nodes found
        """
        attr_helper = AttributeProtoHelper(quant_identity_node.proto)
        layer_name = str(attr_helper.get_attr_value('op_name'), encoding="utf-8")
        # apply bfs to search conv node
        root = quant_identity_node
        search_queue = []
        search_queue.append(root)
        while search_queue:
            current = search_queue.pop(0)
            if current.type == 'Conv':
                consumer = current
                break

            next_nodes, _ = current.get_consumers(0)
            for next_node in next_nodes:
                next_node.set_attr('padding_target_node', layer_name)
                search_queue.append(next_node)

        consumer.set_attr('padding_circle', True)
        return consumer

    @staticmethod
    def _parse_unsqueeze_nodes(quant_identity_node):
        """
        Function: apply bfs to find the conv layer from the circular_padding_nodes
        Parameter: quant_identity_node: root nodes quant_identity
        Return: consumer the conv nodes found
        """
        # apply bfs to search conv node
        root = quant_identity_node
        search_queue = []
        search_queue.append(root)
        while search_queue:
            current = search_queue.pop(0)
            if current.type == 'Conv' or current.type == 'ConvTranspose' or current.type in RNN_LAYER_TYPE:
                consumer = current
                break

            next_nodes, _ = current.get_consumers(0)
            for next_node in next_nodes:
                search_queue.append(next_node)

        consumer.set_attr('input_dimension_reduction', True)
        return consumer

    def add_node(self, node_proto, index=None):
        """
        Function: Add node that constructed from node_proto to graph
        Parameter: None
        Return: Node that be added
        """
        # Set node_id to be unique by decorate_node_name function
        if isinstance(node_proto, onnx_pb.SparseTensorProto):
            if node_proto.HasField('values'):
                node_id = node_proto.values.name
                node_proto.values.name = self._decorate_node_name(node_id)
            elif node_proto.HasField('indices'):
                node_id = node_proto.indices.name
                node_proto.indices.name = self._decorate_node_name(node_id)
            else:
                raise RuntimeError('Cannot find tensor in sparse_initializer')
        else:
            node_id = node_proto.name
            node_proto.name = self._decorate_node_name(node_id)
        # Generate node to graph
        node = Node(self._tail, node_proto, model_path=self.model_path)
        if index is None:
            self._nodes.insert(0, node)
        else:
            self._nodes.insert(index, node)
        return node

    def remove_node(self, delete_node):
        """
        Function: Remove data node from graph by name
        Parameter: None
        Return: None
        """
        remove_done = False
        for index, node in enumerate(self._nodes + self._in_out_nodes):
            if node == delete_node:
                if isinstance(node.proto, onnx_pb.ValueInfoProto):
                    raise RuntimeError('Cannot remove ValueInfoProto node ' \
                        '{} from graph.'.format(node.name))
                del self._nodes[index]
                remove_done = True
                break
        if not remove_done:
            raise RuntimeError(f'Remove {delete_node.name} from graph failed, cannot found')

    def add_model(self, model):
        """ Add model for graph
        """
        self.model = model

    def dump_proto(self):
        """
        Function: Dump all nodes and weights of graph to onnx_pb.ModelProto format
        """
        LOGGER.logi('Doing whole model dump...', module_name=MODULE_NAME)
        self.topologic_sort()

        net = onnx_pb.ModelProto()
        net.CopyFrom(self._net)
        graph = net.graph # pylint: disable=E1101
        for node in self._nodes:
            if isinstance(node.proto, onnx_pb.NodeProto):
                node_proto = graph.node.add() # pylint: disable=E1101
                node_proto.CopyFrom(node.dump_proto())
            elif isinstance(node.proto, onnx_pb.TensorProto):
                tensor = graph.initializer.add()  # pylint: disable=E1101
                tensor.CopyFrom(node.dump_proto())
            elif isinstance(node.proto, onnx_pb.SparseTensorProto):
                sparse = graph.sparse_initializer.add() # pylint: disable=E1101
                sparse.CopyFrom(node.dump_proto())
            elif isinstance(node.proto, onnx_pb.ValueInfoProto):
                continue
            else:
                raise TypeError("Unexpected node_proto {}! only [NodeProto, "
                                "TensorProto, SparseTensorProto] are "
                                "supported.".format(type(node.proto)))
        return net

    def remove_initializer(self, delete_node):
        """
        Function: Remove initializer node from graph
        Parameter: delete_node, the node to be removed
        Return: None
        """
        is_removed = False
        for index, node in enumerate(self._nodes):
            if node == delete_node:
                if not isinstance(node.proto, onnx_pb.TensorProto):
                    raise RuntimeError('Cannot only remove initializer node ' \
                        'from graph by this api. "%s" is "%s"' % (node.name, \
                        node.type))
                if delete_node.get_output_anchor(0).get_peer_input_anchor():
                    LOGGER.logd('Node "%s" still connect to other node, cannot'
                                'remove it now.' % (delete_node.name))
                    return
                del self._nodes[index]

                initializer_name = node.ori_name
                for i, del_node in enumerate(self._in_out_nodes):
                    if isinstance(del_node.proto, onnx_pb.ValueInfoProto) and \
                            del_node.ori_name == initializer_name:
                        del self._in_out_nodes[i]

                is_removed = True
                LOGGER.logd('Remove node "%s" from graph success.' %
                            (delete_node.name))
                break
        if not is_removed:
            raise RuntimeError('Remove %s from graph failed, cannot found' %
                               (delete_node.name))

    def deep_copy(self):
        """
        Function: Make a copy of current graph
        Parameter: None
        Return: Graph's copy
        """
        copy_net = self.dump_proto()
        copy_graph = Graph(copy_net, model_path=self.model_path)
        # copy node attrs
        for node in self._nodes:
            copy_node = copy_graph.get_node_by_name(node.name)
            copy_node.set_attrs(node.attrs)
        return copy_graph

    def _init_graph(self):
        output_record = {}
        # 0. init graph input/output nodes
        self._init_graph_input_node(output_record)
        # 1. parse node proto structure
        self._parse_initializer(output_record)
        self._parse_sparse_initializer(output_record)
        self._parse_node_proto(output_record)
        self._init_graph_output_node(output_record)
        # 2. update the name prefix of nodes, only node with parameters has
        #    module_name
        self._update_node_names()

        # delete node and initializer in graph
        self._net.graph.ClearField('node')
        self._net.graph.ClearField('initializer')
        self._net.graph.ClearField('sparse_initializer')
        # delete custom opset domain in graph
        if len(self._net.opset_import) > 1:
            self._net.opset_import.pop()

    def _init_graph_input_node(self, output_record):
        del_input_list = []
        for idx, input_desc in enumerate(self._net.graph.input):
            flag = False
            for initializer_proto in self._net.graph.initializer:
                if input_desc.name == initializer_proto.name:
                    del_input_list.insert(0, idx)
                    flag = True
                    break
            if flag:
                continue

            graph_input_node = Node(self._tail, input_desc, model_path=self.model_path)
            graph_input_node.add_output_anchor(input_desc.name)
            self._in_out_nodes.append(graph_input_node)
            output_record[input_desc.name] = [graph_input_node, 0]
        for idx in del_input_list:
            del self._net.graph.input[idx]

    def _init_graph_output_node(self, output_record):
        for output_desc in self._net.graph.output:
            graph_output_node = Node(self._tail, output_desc, model_path=self.model_path)
            graph_output_node.add_input_anchor(output_desc.name)
            self._in_out_nodes.append(graph_output_node)
            # Add link to output
            if output_desc.name not in output_record:
                raise ReferenceError('Cannot find tensor %s in model.' % (
                    output_desc.name))
            src_node, src_index = output_record[output_desc.name]
            self.add_edge(src_node, src_index, graph_output_node, 0)

    def _parse_initializer(self, output_record):
        for initializer_proto in self._net.graph.initializer:
            node = Node(self._tail, initializer_proto, model_path=self.model_path)
            self._nodes.append(node)
            output_record[node.name] = [node, 0]

    def _parse_sparse_initializer(self, output_record):
        for sparse_initializer_proto in self._net.graph.sparse_initializer:
            node = Node(self._tail, sparse_initializer_proto, model_path=self.model_path)
            self._nodes.append(node)
            output_record[node.name] = [node, 0]

    def _parse_node_proto(self, output_record):
        for index, node_proto in enumerate(self._net.graph.node):
            if not node_proto.HasField('name'):
                node_proto.name = 'node_{}'.format(index)
            node = Node(self._tail, node_proto, model_path=self.model_path)
            self._nodes.append(node)

            for output_anchor in node.output_anchors:
                output_record[output_anchor.name] = [node, output_anchor.index]
        # Add link to output
        for node in self._nodes + self._in_out_nodes:
            for input_anchor in node.input_anchors:
                if input_anchor.name not in output_record:
                    if input_anchor.name == '':
                        continue
                    raise ReferenceError('Cannot find tensor %s in model.'
                                         % (input_anchor.name))
                src_node, src_index = output_record[input_anchor.name]
                self.add_edge(src_node, src_index, node, input_anchor.index)

    def _update_node_names(self):
        def set_name_node(node, new_name):
            """set node's name as new_name. """
            LOGGER.logd('set "%s" to "%s"' % (new_name, node.name), MODULE_NAME)
            try:
                target_node = self.get_node_by_name(new_name)
            except RuntimeError:
                LOGGER.logd('"%s" is unique in graph.' % (new_name), MODULE_NAME)
            else:
                LOGGER.logd('"%s" already exist in graph, may be reused module'
                            '.' % (new_name), MODULE_NAME)
                node.set_attr('is_reuse', True)
                target_node.set_attr('is_reuse', True)

            node.set_name(self._decorate_node_name(new_name))
            node.set_module_name(new_name)

        # get quantize node name from mark node, and set to itself
        quant_mark_nodes = list()
        for node in self._nodes:
            if node.type == 'QuantIdentity':
                attr_helper = AttributeProtoHelper(node.proto)
                layer_name = str(attr_helper.get_attr_value('op_name'),
                                 encoding="utf-8")
                layer_module_type = str(attr_helper.get_attr_value('module_type'),
                                 encoding="utf-8")
                consumers, _ = node.get_consumers(0)
                consumer = consumers[0]
                if consumer.type == 'Pad':
                    set_name_node(consumer, '%s_pad' % (layer_name))
                    consumer_output = consumer.get_output_anchor(0)
                    consumer = consumer_output.get_peer_input_anchor()[0].node

                if consumer.type == 'Transpose':
                    set_name_node(consumer, '%s_transpose' % (layer_name))
                    consumer_output = consumer.get_output_anchor(0)
                    consumer = consumer_output.get_peer_input_anchor()[0].node

                if consumer.type == 'Unsqueeze':
                    consumer = Graph._parse_unsqueeze_nodes(node)

                # handle circular padding for conv2d
                if layer_module_type == 'Conv2d' and consumer.type != 'Conv' and consumer.type != 'Unsqueeze':
                    consumer = Graph._parse_circular_padding_nodes(node)

                set_name_node(consumer, layer_name)
                quant_mark_nodes.append(node)
        # delete quant mark nodes
        for node in quant_mark_nodes:
            self.delete_node(node, 0, 0)
            self.remove_node(node)

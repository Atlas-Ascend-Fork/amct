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
from ...utils.log import LOGGER


class GraphBase():
    """
    Function: Data structure of graph IR
    APIs: init_graph, nodes, topologic_sort, add_edge, remove_edge,
    add_node, add_data_node, remove_node, get_node, dump_proto,
    deep_copy
    """
    def __init__(self, net):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        self._data_nodes = []
        self._in_out_nodes = []
        self._nodes = []
        self._node_ids = []
        self._tail_index = 0
        self._net = net

    @property
    def nodes(self):
        """
        Function: Get all nodes in graph
        Parameter: None
        Return: nodes in graph
        """
        return self._nodes

    @property
    def node_ids(self):
        """
        Function: Get all node_ids in graph
        Parameter: None
        Return: node_ids in graph
        """
        return self._node_ids

    @property
    def _tail(self):
        current_index = self._tail_index
        self._tail_index += 1
        return current_index

    def get_node(self, node_index):
        """
        Function: Get specific node in graph by index
        Parameter: None
        Return: node in graph
        """
        object_node = None
        for node in self._nodes:
            if node.index == node_index:
                object_node = node
                break
        if object_node is None:
            raise RuntimeError("Cannot find node {} in " \
                "graph".format(node_index))
        return object_node

    def check_node_in_graph(self, node_name):
        """
        Function: check whether a specific node in graph
        Parameter: node_name: checked node's name
        Return: whether node in graph
        """
        for node in self._nodes + self._data_nodes:
            if node.name == node_name:
                return True
        return False

    def get_node_by_name(self, node_name):
        """
        Function: Get specific node in graph by index
        Parameter: None
        Return: node in graph
        """
        for node in self._nodes + self._data_nodes:
            if node.name == node_name:
                return node

        raise RuntimeError('Cannot find node "{}" in ' \
            'graph'.format(node_name))

    def get_node_by_module_name(self, module_name):
        """
        Function: Get specific node in graph by module_name
        Parameter: module_name: module name
        Return: nodes in graph
        """
        object_nodes = list()
        for node in self._nodes:
            if node.module_name == module_name:
                object_nodes.append(node)
        if not object_nodes:
            raise RuntimeError('Cannot find node\'s module_name "{}" in ' \
                'graph'.format(module_name))
        return object_nodes

    def topologic_sort(self):
        """
        Function: Do whole grpah topologic sort
        Parameter: None
        Return: None
        """
        # Step1: record all zero indegree node as sorted
        sorted_nodes_index = set()
        sorted_indexes = set()
        sorted_nodes = []
        sorted_node_ids = []
        self._record_zero_indegree_nodes(sorted_nodes_index, sorted_indexes,
                                         sorted_nodes, sorted_node_ids)
        # Step2: record nodes that all input peer_node is in sorted_nodes_index
        record_sorted_num = len(sorted_nodes_index)
        while len(sorted_nodes_index) < len(self._nodes):
            for index, node in enumerate(self._nodes):
                if index in sorted_indexes:
                    continue
                if node.index not in sorted_nodes_index:
                    all_input_ready = True
                    for input_anchor in node.input_anchors:
                        producer = input_anchor.get_peer_output_anchor()
                        if producer is None or producer.node.is_data_node or \
                            producer.node in self._in_out_nodes:
                            continue
                        if producer.node.index not in sorted_nodes_index:
                            all_input_ready = False
                            break
                    if all_input_ready:
                        sorted_nodes_index.add(node.index)
                        sorted_indexes.add(index)
                        sorted_nodes.append(node)
                        sorted_node_ids.append(node.name)
                        break
            if record_sorted_num == len(sorted_nodes_index):
                raise RuntimeError('May exist loop in graph, topological '
                                   'sort failed!')

            record_sorted_num = len(sorted_nodes_index)

        self._nodes = sorted_nodes
        self._node_ids = sorted_node_ids
        self._renumber_node_index()

    def add_edge(self, src_node, src_index, dst_node, dst_index):
        """
        Function: Add edge from src_node[src_index] to dst_node[dst_index]
        Parameter: None
        Return: None
        """
        # Prepare src anchor info
        src_anchor = self._prepare_src_anchor(src_node, src_index)
        # Prepare dst anchor info
        dst_anchor = self._prepare_dst_anchor(dst_node, dst_index)

        if dst_anchor.get_peer_output_anchor() is not None:
            raise RuntimeError("Node:{} input:{} already has peer output " \
                "anchor, disconnect it first".format( \
                dst_node.name, dst_index))
        # add link between src_anchor and dst_anchor
        src_anchor.add_link(dst_anchor)
        dst_anchor.add_link(src_anchor)
        LOGGER.logd("Add edge from {}[{}] to {}[{}] success!".format(
            src_node.name, src_index, dst_node.name, dst_index), 'Graph')

    def remove_edge(self, src_node, src_index, dst_node, dst_index):
        """
        Function: Remove edge from src_node[src_index] to dst_node[dst_index]
        Parameter: None
        Return: None
        """
        # Prepare src anchor info
        src_anchor = self._prepare_src_anchor(src_node, src_index)
        if not src_anchor.get_peer_input_anchor():
            raise RuntimeError("Src node {} output {} have no peer input " \
                "anchor".format(src_node.name, src_index))
        # Prepare dst anchor
        dst_anchor = self._prepare_dst_anchor(dst_node, dst_index)
        if dst_anchor.get_peer_output_anchor() is None:
            raise RuntimeError("Node:{} input:{} have no peer output " \
                "anchor".format(dst_node.name, dst_index))
        # Disconnct link between src anchor and dst anchor
        if src_anchor is not dst_anchor.get_peer_output_anchor() or \
            dst_anchor not in src_anchor.get_peer_input_anchor():
            raise RuntimeError('There is no link from {}[{}] to {}[{}]'.format(
                src_node.name, src_index, dst_node.name, dst_index))
        src_anchor.del_link(dst_anchor)
        dst_anchor.del_link()
        LOGGER.logd("Remove edge from {}[{}] to {}[{}] success!".format(
            src_node.name, src_index, dst_node.name, dst_index), 'Graph')

    def insert_parallel_node(self, node, in_idx,
                             brother_node, brother_in_idx):
        """
        Function: Insert paralel node to brother_node as follows, so
        node[in_idx] and brother_node[brother_in_idx] are both linked
        to producer[*].
                    producer        producer
                       |      -->     / \
                  brother_node    node  brother_node
        Parameter:
            node: Node, node to be insert
            in_idx: int, which input will to be linked producer
            brother_node: Node, node to be parallel
            brother_in_idx: int, which input to insert node
        Return: None
        """
        brother_input_anchor = brother_node.get_input_anchor(brother_in_idx)
        peer_output_anchor = brother_input_anchor.get_peer_output_anchor()
        producer = peer_output_anchor.node
        out_idx = peer_output_anchor.index
        self.add_edge(producer, out_idx, node, in_idx)

    def insert_node_before(self, node, in_idx, out_idx,
                           post_node, post_in_idx):
        """
        Function: Insert node before post_node as follows, so
        node.intput[in_idx] is link to producer[*] and node.output[out_idx]
        is link to post_node.input[post_in_idx]
                    producer        producer
                       |      -->      |
                   post_node          node
                                       |
                                    post_node
        Parameter:
            node: Node, node to be insert
            in_idx: int, which input will to be linked producer
            out_idx: int, which input will to be linked post_node
            post_node: Node, node to be insert before it
            post_in_idx: int, which input to insert node
        Return: None
        """
        producer, producer_out_idx = post_node.get_producer(post_in_idx)
        self.remove_edge(producer, producer_out_idx, post_node, post_in_idx)
        self.add_edge(producer, producer_out_idx, node, in_idx)
        self.add_edge(node, out_idx, post_node, post_in_idx)

    def insert_node_after(self, node, in_idx, out_idx,
                          pre_node, pre_out_idx):
        """
        Function: Insert node after pre_node as follows, so
        node.intput[in_idx] is link to producer[*] and node.output[out_idx]
        is link to post_node.input[post_in_idx]
                    pre_node          pre_node
                      / \\      -->      |
                 node1   node2          node
                                        / \
                                    node1  node2
        Parameter:
            node: Node, node to be insert
            in_idx: int, which input will to be linked producer
            out_idx: int, which input will to be linked post_node
            post_node: Node, node to be insert before it
            post_in_idx: int, which input to insert node
        Return: None
        """
        consumers, consumer_in_idxs = pre_node.get_consumers(pre_out_idx)
        for consumer, consumer_in_idx in zip(consumers, consumer_in_idxs):
            self.remove_edge(pre_node, pre_out_idx, consumer, consumer_in_idx)
            self.add_edge(node, out_idx, consumer, consumer_in_idx)

        self.add_edge(pre_node, pre_out_idx, node, in_idx)

    def delete_node(self, node, in_idx, out_idx):
        """
        Function: Insert node after pre_node as follows, so
        node.intput[in_idx] is link to producer[*] and node.output[out_idx]
        is link to post_node.input[post_in_idx]
                    pre_node           pre_node
                       |                /  \
                     node     -->   node1  node2
                      / \\
                 node1   node2

        Parameter:
            node: Node, node to be insert
            in_idx: int, which input will to be delted
            out_idx: int, which input will to be delted
        Return: None
        """
        producer, producer_out_idx = node.get_producer(in_idx)
        self.remove_edge(producer, producer_out_idx, node, in_idx)

        consumers, in_idx = node.get_consumers(out_idx)
        for consumer, consumer_in_idx in zip(consumers, in_idx):
            self.remove_edge(node, out_idx, consumer, consumer_in_idx)
            self.add_edge(producer, producer_out_idx,
                          consumer, consumer_in_idx)

    def remove_node(self, delete_node):
        """
        Function: Remove node from graph
        Parameter: None
        Return: None
        """
        remove_done = False
        for index, node in enumerate(self._nodes):
            if node == delete_node:
                del self._nodes[index]
                remove_done = True
                break
        if not remove_done:
            raise RuntimeError('Remove %s from graph failed, cannot found' % (
                delete_node.name))

    def _decorate_node_name(self, node_name):
        """decorate node_name to generate unique node_id"""
        node_id = node_name
        dec_index = 1
        while node_id in self._node_ids:
            node_id = '{}{}'.format(node_name, dec_index)
            dec_index += 1
        self._node_ids.append(node_id)
        return node_id

    def _prepare_dst_anchor(self, dst_node, dst_index):
        """
        Function: Find dst anchor according to dst_node name and index
        Parameter: None
        Return: Dst input anchor
        """
        if dst_node not in self._nodes + self._data_nodes + self._in_out_nodes:
            raise RuntimeError('Cannot find node "%s" in graph.' % (
                dst_node.name))
        if dst_index >= len(dst_node.input_anchors):
            raise RuntimeError("Get input of {} from node:{} out of " \
                "range".format(dst_index, dst_node))
        return dst_node.get_input_anchor(dst_index)

    def _record_zero_indegree_nodes(self, sorted_nodes_index, sorted_indexes,
                                    sorted_nodes, sorted_node_ids):
        """
        Function: Record all zero indegree nodes in graph
        Parameter: None
        Return: List of zero indegree nodes' index
        """
        for index, node in enumerate(self._nodes):
            # Check if node's all input is come from 'input' of net
            all_input_ready = True
            for input_anchor in node.input_anchors:
                if input_anchor.get_peer_output_anchor() is None:
                    continue
                peer_node = input_anchor.get_peer_output_anchor().node
                if not peer_node.is_data_node:
                    all_input_ready = False
                    break
            if all_input_ready:
                sorted_nodes_index.add(node.index)
                sorted_indexes.add(index)
                sorted_nodes.append(node)
                sorted_node_ids.append(node.name)
        return sorted_nodes_index, sorted_indexes

    def _renumber_node_index(self):
        """
        Function: Renumber index of nodes
        Parameter: None
        Return: None
        """
        for index, node in enumerate(self._nodes):
            node.set_index(index)
        self._tail_index = len(self._nodes)

    def _prepare_src_anchor(self, src_node, src_index):
        """
        Function: Find src anchor according to src_node name and index
        Parameter: None
        Return: Src output anchor
        """
        if src_node not in self._nodes + self._data_nodes + self._in_out_nodes:
            raise RuntimeError('Cannot find node "%s" in graph.' % (
                src_node.name))
        if src_index >= len(src_node.output_anchors):
            raise RuntimeError("Get output of {} from node:{} out of " \
                "range".format(src_index, src_node))

        return src_node.get_output_anchor(src_index)

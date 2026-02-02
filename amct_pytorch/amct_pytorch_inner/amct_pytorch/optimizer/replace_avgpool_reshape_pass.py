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


from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.optimizer.replace_avgpool_flatten_pass import ReplaceAvgpoolFlattenPass
from ...amct_pytorch.utils.log import LOGGER


class ReplaceAvgpoolReshapePass(BaseFusionPass):
    """
    Function: Replace "GlobalAveragePool + Reshape" node in graph.
    APIs: match_pattern, do_pass
    """
    def __init__(self):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseFusionPass.__init__(self)
        self.structure = {}

    def match_pattern(self, node):
        """
        Function: Match the GlobalAveragePool node to be replaced in graph
        Parameters: node: node in graph to be matched
        Return: True: matched / False: mismatch
        """
        if node.type != 'GlobalAveragePool':
            return False

        consumers, in_idxs = node.get_consumers(0)
        if len(consumers) != 1 or consumers[0].type != 'Reshape':
            LOGGER.logd("node {} match_pattern fail for it must have only one consumer Reshape.".format(node.name),
                        'ReplaceAvgpoolReshapePass')
            return False

        reshape_node = consumers[0]

        producer, out_idx = node.get_producer(0)
        consumers, in_idxs = producer.get_consumers(out_idx)
        for consumer, in_idx in zip(consumers, in_idxs):
            if consumer.type == 'GlobalAveragePool':
                consumers.remove(consumer)
                in_idxs.remove(in_idx)
                break

        search_node = None
        for consumer in consumers:
            if consumer.type == 'Shape':
                search_node = consumer
        if search_node is None:
            return False
        shape_nodes = [search_node]

        for target_type in ['Gather', 'Unsqueeze', 'Concat', 'Reshape']:
            consumers, in_idxs = search_node.get_consumers(0)
            if len(consumers) != 1 or consumers[0].type != target_type:
                return False
            search_node = consumers[0]
            shape_nodes.append(search_node)

        if search_node is not reshape_node:
            return False

        self.structure = {
            node.name: [reshape_node, shape_nodes]
        }
        return True

    def do_pass(self, graph, object_node):
        """
        Function: Do actually replacement.
        Parameters:
        graph: graph structure
        object_node: node to process
        Return: None
        """
        LOGGER.logd("Doing: delete node {} in graph.".format(object_node.name),
                    'ReplaceAvgpoolReshapePass')
        reshape_node, shape_nodes = self.structure[object_node.name]

        proto_p1ton1 = ReplaceAvgpoolFlattenPass.construct_d4tod2('.'.join([object_node.name, reshape_node.name]))
        node_p1ton1 = graph.add_node(proto_p1ton1)
        graph.insert_node_before(node_p1ton1, 0, 0, object_node, 0)

        graph.delete_node(object_node, 0, 0)
        graph.remove_node(object_node)
        graph.delete_node(reshape_node, 0, 0)

        producer, out_idx = shape_nodes[0].get_producer(0)
        graph.remove_edge(producer, out_idx, shape_nodes[0], 0)
        for node in shape_nodes:
            graph.remove_node(node)

        LOGGER.logd("Finished: delete node {} in graph.".format(object_node.name),
                    'ReplaceAvgpoolReshapePass')

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
from ...amct_pytorch.utils.log import LOGGER


class DeleteIsolatedNodePass(BaseFusionPass):
    """
    Function: Delete isolated nodes in graph.
    APIs: match_pattern, do_pass
    """
    def __init__(self):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseFusionPass.__init__(self)

    @staticmethod
    def match_pattern(node):
        """
        Function: Match the node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched / False: mismatch
        """
        if node.type == 'graph_anchor':
            return False

        for idx, _ in enumerate(node.output_anchors):
            consumers, _ = node.get_consumers(idx)
            if consumers:
                return False

        return True

    @staticmethod
    def do_pass(graph, object_node):
        """
        Function: Do actual quantization and node's weight is changed to int8.
        Parameters:
        graph: graph structure
        object_node: node to process
        Return: None
        """
        LOGGER.logd("Doing: delete node {} in graph.".format(object_node.name),
                     'DeleteIsolatedNodePass')

        delete_nodes = []

        def del_node(node):
            for idx, _ in enumerate(node.input_anchors):
                producer, out_idx = node.get_producer(idx)
                if producer is None:
                    continue
                graph.remove_edge(producer, out_idx, node, idx)
                if producer not in delete_nodes:
                    delete_nodes.append(producer)
            graph.remove_node(node)

            while delete_nodes:
                delete_node = delete_nodes.pop()
                is_del = True
                for idx, _ in enumerate(node.output_anchors):
                    consumers, _ = delete_node.get_consumers(idx)
                    if consumers:
                        is_del = False
                        #  add graph inputs
                        break
                if is_del:
                    del_node(delete_node)


        del_node(object_node)

        LOGGER.logd("Finished: delete node {} in graph.".format(object_node.name),
                     'DeleteIsolatedNodePass')

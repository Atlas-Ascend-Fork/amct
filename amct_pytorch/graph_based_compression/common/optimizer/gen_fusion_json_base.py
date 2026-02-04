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
import os
import datetime
from google.protobuf import json_format

from ...common.utils import files as files_util
from ...common.utils.util import FILE_MODE
from ...common.utils.files import check_files_exist
from ...common.utils.attrs_list import ATTR_NODE_EQUIVALENT_OBJECT_LAYER
from ...common.utils.attrs_list import ATTR_NODE_EQUIVALENT_INPUT
from ...common.utils.attrs_list import ATTR_NODE_EQUIVALENT_OUTPUT
from ...common.utils.attrs_list import ATTR_NODE_FUSION_INFO
from ...common.utils.attrs_list import ATTR_NODE_OUTPUT_NODE
from ...utils.log import LOGGER
from .base_pass import BasePass


class GenFusionJsonBasePass(BasePass):
    """
    Function: Generate json file that contains fusion layer infos
    APIs: match_pattern, do_pass
    """
    def __init__(self, fusion_pb2):
        """
        Function: Init object
        Parameters: None
        Return: None
        """
        super().__init__()
        self._ops = []
        self._graph_name = None
        self._dump_file = ''
        self._save_path = None
        self._save_prefix = None
        self.fusion_info = fusion_pb2.FusionInfo()
        self.fusion_info.graph.add()

    def tear_down(self):
        """
        Function: write fusion info to json
        """
        # add name
        if self._save_prefix != '':
            self.fusion_info.name = self._save_prefix
        elif self._graph_name is not None:
            self.fusion_info.name = self._graph_name
        else:
            time = datetime.datetime.utcnow()
            time_str = '{:0>4}{:0>2}{:0>2}{:0>2}{:0>2}{:0>2}{:0>2}'.format( \
                time.year, time.month, time.day, time.hour, time.minute, \
                time.second, time.microsecond // 10000)
            self.fusion_info.name = 'default_{}'.format(time_str)
        # trans proto to json
        json_string = json_format.MessageToJson(
            self.fusion_info, preserving_proto_field_name=True)
        # write to file
        dump_file = os.path.join(self._save_path, \
            '{}_quant.json'.format(self._save_prefix))
        files_util.create_path(self._save_path)
        check_files_exist([dump_file])
        with open(dump_file, 'w') as fid:
            fid.write(json_string)
        os.chmod(dump_file, FILE_MODE)

    def match_pattern(self, node):
        """
        Function: Find node need dump info in graph
        Parameters: node: node in graph
        Return:
        True: node that need to dump info
        False: skip the node
        """
        return True

    def set_dump_file_dir(self, save_path, save_prefix):
        """
        Function: Set file name and path to save dump file
        Parameters: dump_file: file name and path to save dump file
        Return: None
        """
        self._save_path = save_path
        self._save_prefix = save_prefix

    def do_pass(self, graph, object_node):
        """
        Funtion: add fusion info for object_node
        Parameters:
            graph: object_node's graph
            object_node: node to process
        """
        op_info = self.fusion_info.graph[0].op.add()
        op_info.name = object_node.name
        op_info.type = object_node.type
        # Generate dump info of layer to quantize
        is_quant = 0
        is_quant += int(object_node.has_attr(ATTR_NODE_EQUIVALENT_INPUT))
        is_quant += int(object_node.has_attr(ATTR_NODE_EQUIVALENT_OUTPUT))
        is_quant += int(object_node.has_attr(ATTR_NODE_FUSION_INFO))
        if is_quant > 0:
            LOGGER.logd('Find node {} as object node'.format(object_node.name))
            attr = op_info.attr.add()
            attr.key = '_datadump_original_op_names'
            set_attr_ori_op(attr, object_node)

        # Generate Quant/DeQuant layer dump info
        if object_node.has_attr(ATTR_NODE_EQUIVALENT_OBJECT_LAYER):
            LOGGER.logd('Find node {} as add node'.format(object_node.name))
            eq_node_name = object_node.get_attr( \
                ATTR_NODE_EQUIVALENT_OBJECT_LAYER)
            if eq_node_name == '':
                eq_node = None
            else:
                eq_node = graph.get_node_by_name(eq_node_name)
            attr = op_info.attr.add()
            set_attr_ori_op(attr, eq_node)

            if eq_node is None:
                return
            if eq_node.has_attr(ATTR_NODE_EQUIVALENT_OUTPUT) and \
                object_node.name == eq_node.get_attr(
                        ATTR_NODE_EQUIVALENT_OUTPUT):
                if eq_node.has_attr(ATTR_NODE_FUSION_INFO):
                    output_origin_name = eq_node.get_attr(
                        ATTR_NODE_OUTPUT_NODE)
                else:
                    output_origin_name = eq_node.name
                output_desc = op_info.output_desc.add()
                set_output_desc(output_desc, output_origin_name)

    def run(self, graph):
        """
        Function: Generate fusion json for graph
        Parameters:
        graph: Graph
        Return: None
        """
        self.set_up()
        # Step1: match pattern and record first matched node
        matched_nodes = []
        for node in graph.nodes:
            if self.match_pattern(node):
                matched_nodes.append(node)
        # Step2: do each matched node fusion operation
        for node in matched_nodes:
            self.do_pass(graph, node)

        self.tear_down()


def set_attr_ori_op(attr, node):
    """
    Function: set a attr for node, the attr means the node's origin op. The
    layer1.quant origin op is layer1, for example.
    """
    attr.key = '_datadump_original_op_names'
    attr.value.list.val_type = 1
    if node is None:
        return
    if node.has_attr(ATTR_NODE_FUSION_INFO):
        ori_nodes = node.get_attr(ATTR_NODE_FUSION_INFO)
    else:
        ori_nodes = [node.name]
    attr.value.list.s.extend(ori_nodes)


def set_output_desc(output_desc, output_origin_name):
    """
    Function: set a output_desc for node, the output_desc means the node's
    origin output. The layer1 origin op is layer1.bn if layer1+layer1.bn
    is fused, for example.
    """
    attr1 = output_desc.attr.add()
    attr1.key = '_datadump_origin_format'
    attr1.value.s = 'NCHW'
    attr2 = output_desc.attr.add()
    attr2.key = '_datadump_data_type'
    attr2.value.s = 'DT_FLOAT'
    attr3 = output_desc.attr.add()
    attr3.key = '_datadump_origin_output_index'
    attr3.value.i = 0
    attr4 = output_desc.attr.add()
    attr4.key = '_datadump_origin_name'
    attr4.value.s = output_origin_name

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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class IntergerProgramming:
    """ base class of interger programming. """
    def __init__(self, weight_info, value_info, constraint_weight):
        """ Init func.

        Args:
            weight_info (dict): weight info to do search
            value_info (dict): value info to do search
            constraint_weight (float): constraint weight to do search
        """
        self.weight_info = weight_info
        self.value_info = value_info
        self.constraint_weight = constraint_weight

    def search(self, search_range=None, reverse=False):
        """ Do search to find best choice.

        Args:
            search_range (dict, optional): key is valid layer and value is valid range. Defaults to None.
            reverse (bool, optional): if True, find max value and otherwise min value. Defaults to False.

        Returns:
            [type]: [description]
        """
        if reverse:
            return self.search_max(search_range)
        return self.search_min(search_range)

    def search_max(self, search_range=None):
        """ Do search to find best choice aimming to find max value.

        Args:
            search_range ([type], optional): key is valid layer and value is valid range. Defaults to None.
        """
        pass

    def search_min(self, search_range=None):
        """ Do search to find best choice aimming to find min value.

        Args:
            search_range ([type], optional): key is valid layer and value is valid range. Defaults to None.
        """
        pass


class SearchTreeNode:
    """ Class of search tree node. """
    def __init__(self, choice=None, weight=0, value=0, name=''):
        self.choice = choice
        self.weight = weight
        self.value = value
        self.name = name

        self.parent = None
        self.child_num = 0
        self.sum_weight = 0
        self.sum_value = 0
        self.invalid = False

    def __str__(self):
        """ override function """
        return 'name: {} choice:{} weight: {:.4f} value: {:.4f}'.format(self.name, self.choice, self.weight,
                                                                        self.value)

    def __repr__(self):
        """ override function """
        return self.__str__()

    @staticmethod
    def get_path(node):
        """ Get path from node to root.

        Args:
            node (SearchTreeNode): node's path to get.

        Returns:
            dict: the choice path from root to node.
        """
        path = {}
        while node.choice is not None:
            path[node.name] = node.choice
            node = node.parent
        return path

    def add_child(self, child_node):
        """ Add child_node as a child.

        Args:
            child_node (SearchTreeNode): child to add.
        """
        setattr(self, 'branch_{}'.format(self.child_num), child_node)
        self.child_num += 1

    def set_parent(self, parent_node):
        """ Set parent_node as parent.

        Args:
            parent_node (SearchTreeNode): set it as parent.
        """
        self.parent = parent_node
        self.sum_value = parent_node.sum_value + self.value
        self.sum_weight = parent_node.sum_weight + self.weight


class BranchBound(IntergerProgramming):
    """ BranchBound from IntergerProgramming """
    @staticmethod
    def add_layer(current_list, layer, choices, value_info, weight_info):
        """ add one layer in search tree.

        Args:
            current_list (list of SearchTreeNode): last layer
            layer (string): name of new layer
            choices (list): the choice for layer
            value_info (dict): include value for layer in each choice
            weight_info (dict): include weight for layer in each choice

        Returns:
            list of SearchTreeNode: new next layer
        """
        next_list = []
        for node in current_list:
            for choice in choices:
                child_node = SearchTreeNode(choice,
                                            weight_info[layer][choice],
                                            value_info[layer][choice],
                                            name=layer)
                node.add_child(child_node)
                child_node.set_parent(node)
                next_list.append(child_node)
        return next_list

    def search_min(self, search_range=None):
        """ Do search to find best choice aimming to find min value.

        Args:
            search_range ([type], optional): key is valid layer and value is valid range. Defaults to None.
        """
        layer_names = self.value_info.keys() if search_range is None else search_range.keys()
        current_list = [SearchTreeNode(weight=0, value=0)]

        for layer in layer_names:
            # 1. add new node for layer after each node in current_list
            choices = self.value_info[layer].keys() if search_range is None else search_range[layer]
            next_list = self.add_layer(current_list, layer, choices, self.value_info, self.weight_info)
            # 2. sort according to the weight and prune the invalid path
            next_list.sort(key=lambda x: x.sum_weight, reverse=False)
            pruned_list = []
            for node in next_list:
                if (len(pruned_list) == 0
                        or node.sum_value < pruned_list[-1].sum_value) and node.sum_weight <= self.constraint_weight:
                    pruned_list.append(node)
                    continue
                node.invalid = True
            # 3. set pruned_list as current_list for next iter
            current_list = pruned_list
            # cannot find the valid path
            if not current_list:
                return {}, None, None

        node = current_list[-1]
        best_path = SearchTreeNode.get_path(node)
        return best_path, node.sum_weight, node.sum_value

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
import sys
import os
import unittest
from unittest import mock
import json
import numpy as np
import torch
import math


DEVICE = 'cuda:0'
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op import selective_mask_gen

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestSelectiveMaskGen(unittest.TestCase):
    """
    The UT for SelectiveMaskGen
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_selective_mask_gen')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # @unittest.skip('*')
    def test_algo_001(self):
        """ not to refresh mask, return [1, 1, ..., 1] """
        # prepare input
        tensor_shape = [2, 4, 3, 3]
        tensor_input = torch.randn(tensor_shape)
        prune_axis = 0
        algo_param = {
            'group_size': 4,
            'prune_size': 2,
            'prune_axis': prune_axis,
        }
        print('ori_data:', tensor_input.type())
        print(tensor_input[0])
        # call seletive_mask_gen_op
        prune_mask = selective_mask_gen(tensor_input, algo_param['prune_axis'], \
            algo_param['group_size'], algo_param['prune_size'])
        print(prune_mask[0])
        print(prune_mask.shape)
        mask_counter = np.sum(prune_mask.cpu().numpy()).astype(int)
        shape_counter = 1
        for val in tensor_shape:
            shape_counter *= val
        self.assertEqual(mask_counter, shape_counter)


    def test_algo_002(self):
        """ selective prune  4:2,  prun """
        # prepare input
        tensor_shape = [160, 4, 3, 3]
        tensor_input = torch.randn(tensor_shape)
        prune_axis = 1
        algo_param = {
            'group_size': 4,
            'prune_size': 2,
            'prune_axis': prune_axis,
        }
        print('ori_data:', tensor_input.type())
        print(tensor_input[0])
        # call seletive_mask_gen_op
        prune_mask = selective_mask_gen(tensor_input, algo_param['prune_axis'], \
            algo_param['group_size'], algo_param['prune_size'])
        print(prune_mask[0])
        print(prune_mask.shape)

        # call numpy seletive_mask_gen_op
        prune_mask_faker = SelectiveMaskFaker(tensor_input, \
            algo_param['group_size'], algo_param['prune_size'], algo_param['prune_axis']).do()
        print(prune_mask_faker.shape)
        print(prune_mask_faker[0])

        # compare
        is_equal = (prune_mask_faker.cpu().numpy()==prune_mask.cpu().numpy()).all()
        print(is_equal)
        self.assertEqual(is_equal, True)


    def test_algo_003(self):
        """ selective prune  4:2 """
        # prepare input
        tensor_shape = [160, 7, 3, 3]
        tensor_input = torch.randn(tensor_shape)
        prune_axis = 1
        algo_param = {
            'group_size': 4,
            'prune_size': 2,
            'prune_axis': prune_axis,
        }
        print('ori_data:', tensor_input.type())
        print(tensor_input[0])
        # call seletive_mask_gen_op
        prune_mask = selective_mask_gen(tensor_input, algo_param['prune_axis'], \
            algo_param['group_size'], algo_param['prune_size'])
        print(prune_mask[0])
        print(prune_mask.shape)

        # call numpy seletive_mask_gen_op
        prune_mask_faker = SelectiveMaskFaker(tensor_input, \
            algo_param['group_size'], algo_param['prune_size'], algo_param['prune_axis']).do()
        print(prune_mask_faker.shape)
        print(prune_mask_faker[0])

        # compare
        is_equal = (prune_mask_faker.cpu().numpy()==prune_mask.cpu().numpy()).all()
        print(is_equal)
        self.assertEqual(is_equal, True)

    def test_algo_004(self):
        """ selective prune  4:2 """
        # prepare input
        tensor_shape = [160, 3, 3, 3]
        tensor_input = torch.randn(tensor_shape)
        prune_axis = 0
        algo_param = {
            'group_size': 4,
            'prune_size': 2,
            'prune_axis': prune_axis,
        }
        print('ori_data:', tensor_input.type())
        print(tensor_input[0])
        # call seletive_mask_gen_op
        prune_mask = selective_mask_gen(tensor_input, algo_param['prune_axis'], \
            algo_param['group_size'], algo_param['prune_size'])
        print(prune_mask[0])
        print(prune_mask.shape)

        # call numpy seletive_mask_gen_op
        prune_mask_faker = SelectiveMaskFaker(tensor_input, \
            algo_param['group_size'], algo_param['prune_size'], algo_param['prune_axis']).do()
        print(prune_mask_faker.shape)
        print(prune_mask_faker[0])

        # compare
        is_equal = (prune_mask_faker.cpu().numpy()==prune_mask.cpu().numpy()).all()
        print(is_equal)
        self.assertEqual(is_equal, True)


    def test_algo_005(self):
        """ selective prune  4:2 """
        # prepare input
        tensor_shape = [1, 4, 3, 3]
        tensor_input = torch.randn(tensor_shape)
        prune_axis = 1
        algo_param = {
            'group_size': 4,
            'prune_size': 2,
            'prune_axis': prune_axis,
        }
        print('ori_data:', tensor_input.type())
        print(tensor_input[0])
        # call seletive_mask_gen_op
        prune_mask = selective_mask_gen(tensor_input, algo_param['prune_axis'], \
            algo_param['group_size'], algo_param['prune_size'])
        print(prune_mask[0])
        print(prune_mask.shape)

        # call numpy seletive_mask_gen_op
        prune_mask_faker = SelectiveMaskFaker(tensor_input, \
            algo_param['group_size'], algo_param['prune_size'], algo_param['prune_axis']).do()
        print(prune_mask_faker.shape)
        print(prune_mask_faker[0])

        # compare
        is_equal = (prune_mask_faker.cpu().numpy()==prune_mask.cpu().numpy()).all()
        print(is_equal)
        self.assertEqual(is_equal, True)


    def test_algo_006(self):
        """ selective prune  4:2 """
        # prepare input
        tensor_shape = [16, 6, 3, 3]
        tensor_input = torch.randn(tensor_shape)
        prune_axis = 1
        algo_param = {
            'group_size': 4,
            'prune_size': 2,
            'prune_axis': prune_axis,
            'is_refresh': True
        }
        print('ori_data:', tensor_input.type())
        print(tensor_input[0])
        # call seletive_mask_gen_op
        prune_mask = selective_mask_gen(tensor_input, algo_param['prune_axis'], \
            algo_param['group_size'], algo_param['prune_size'])
        print(prune_mask[0])
        print(prune_mask.shape)

        # call numpy seletive_mask_gen_op
        prune_mask_faker = SelectiveMaskFaker(tensor_input, \
            algo_param['group_size'], algo_param['prune_size'], algo_param['prune_axis']).do()
        print(prune_mask_faker.shape)
        print(prune_mask_faker[0])

        # compare
        is_equal = (prune_mask_faker.cpu().numpy()==prune_mask.cpu().numpy()).all()
        print(is_equal)
        self.assertEqual(is_equal, True)

    # @unittest.skip('*')
    def test_algo_007(self):
        # prepare input
        tensor_shape = [1, 5, 3, 3]
        tensor_input = torch.randn(tensor_shape)
        prune_axis = 1
        algo_param = {
            'group_size': 4,
            'prune_size': 2,
            'prune_axis': prune_axis,
        }
        print('ori_data:', tensor_input.type())
        print(tensor_input[0])
        # call seletive_mask_gen_op
        prune_mask = selective_mask_gen(tensor_input, algo_param['prune_axis'], \
            algo_param['group_size'], algo_param['prune_size'])
        print(prune_mask[0])
        print(prune_mask.shape)

        # call numpy seletive_mask_gen_op
        prune_mask_faker = SelectiveMaskFaker(tensor_input, \
            algo_param['group_size'], algo_param['prune_size'], algo_param['prune_axis']).do()
        print(prune_mask_faker.shape)
        print(prune_mask_faker[0])

        # compare
        is_equal = (prune_mask_faker.cpu().numpy()==prune_mask.cpu().numpy()).all()
        print(is_equal)
        self.assertEqual(is_equal, True)


class SelectiveMaskFaker:
    def __init__(self, data_tensor, group_size=4, prune_size=2, prune_axis=1):
        self.data_tensor = data_tensor
        self.group_size = group_size
        self.prune_size = prune_size
        self.prune_axis = prune_axis

    def do(self):
        prune_mask = self.generate_mask(self.data_tensor,
            self.group_size, self.prune_size, self.prune_axis)
        return prune_mask

    def _padding_and_reshape(self, weights, in_channels, group_size, dim):
        num_groups = int(math.ceil(in_channels / group_size))
        in_padding = num_groups * group_size - in_channels
        group_shape = weights.shape[:dim] + (num_groups, group_size) + weights.shape[dim + 1:]

        padding_shape = weights.shape[:dim] + (in_padding,) + weights.shape[dim + 1:]
        weights_padding = torch.cat((weights.abs(), \
            torch.zeros(padding_shape, dtype=weights.dtype, device=weights.device)), dim=dim)
        weights_group = weights_padding.reshape(group_shape)

        return weights_group

    def generate_mask(self, weights, group_size=4, pruned_size=2, dim=1):
        in_channels = weights.shape[dim]
        if in_channels < group_size:
            return torch.ones_like(weights)

        weights_group = self._padding_and_reshape(weights, in_channels, group_size, dim)
        sort_indices = weights_group.topk(k=pruned_size, dim=dim + 1, largest=False)[1]
        mask = torch.ones_like(weights_group)
        mask.scatter_(dim + 1, sort_indices, 0.0)

        ungroup_shape = weights.shape[:dim] + (-1,) + weights.shape[dim + 1:]
        return mask.reshape(ungroup_shape).split(in_channels, dim=dim)[0]

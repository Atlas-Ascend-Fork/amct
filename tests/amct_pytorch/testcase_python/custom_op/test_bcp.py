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
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op import bcp

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestBcp(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_bcp')
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
        """ tensor:1, ascend_optimized:true, prune_group:1"""
        # prepare input
        tensor_list = [torch.randn([160, 4, 3, 3])]
        prune_axises = [0]
        algo_param = {
            'prune_ratio': 0.3,
            'ascend_optimized': True,
            'prune_group': 1
        }
        print('ori_data:', tensor_list[0].type())
        print(tensor_list[0][0])
        # call bcp_op
        prune_mask = bcp(tensor_list, prune_axises, algo_param['prune_ratio'], algo_param['prune_group'], algo_param['ascend_optimized'])
        # call numpy bcp
        prune_mask_list = BcpFaker(tensor_list, prune_axises, algo_param).do()
        # compare
        is_equal = (prune_mask_list==prune_mask.cpu().numpy()).all()
        self.assertEqual(is_equal, True)

    # @unittest.skip('*')
    def test_algo_002(self):
        """ tensor:1, ascend_optimized:true, prune_group:2"""
        # prepare input
        tensor_list = [torch.randn([160, 4, 3, 3])]
        prune_axises = [0]
        algo_param = {
            'prune_ratio': 0.3,
            'ascend_optimized': True,
            'prune_group': 2
        }
        # call bcp_op
        prune_mask = bcp(tensor_list, prune_axises, algo_param['prune_ratio'], algo_param['prune_group'], algo_param['ascend_optimized'])
        # call numpy bcp
        prune_mask_list = BcpFaker(tensor_list, prune_axises, algo_param).do()
        # compare
        is_equal = (prune_mask_list==prune_mask.cpu().numpy()).all()
        self.assertEqual(is_equal, True)

    # @unittest.skip('*')
    def test_algo_003(self):
        """ tensor:1, ascend_optimized:false, prune_group:1"""
        # prepare input
        tensor_list = [torch.randn([160, 4, 3, 3])]
        prune_axises = [0]
        algo_param = {
            'prune_ratio': 0.3,
            'ascend_optimized': False,
            'prune_group': 1
        }
        # call bcp_op
        prune_mask = bcp(tensor_list, prune_axises, algo_param['prune_ratio'], algo_param['prune_group'], algo_param['ascend_optimized'])
        # call numpy bcp
        prune_mask_list = BcpFaker(tensor_list, prune_axises, algo_param).do()
        # compare
        is_equal = (prune_mask_list==prune_mask.cpu().numpy()).all()
        self.assertEqual(is_equal, True)

    # @unittest.skip('*')
    def test_algo_004(self):
        """ tensor:1, ascend_optimized:true, prune_group:1
            num is too little
        """
        # prepare input
        tensor_list = [torch.randn([16, 4, 3, 3])]
        prune_axises = [0]
        algo_param = {
            'prune_ratio': 0.3,
            'ascend_optimized': True,
            'prune_group': 1
        }
        # call bcp_op
        prune_mask = bcp(tensor_list, prune_axises, algo_param['prune_ratio'], algo_param['prune_group'], algo_param['ascend_optimized'])
        # call numpy bcp
        prune_mask_list = BcpFaker(tensor_list, prune_axises, algo_param).do()
        # compare
        is_equal = (prune_mask_list==prune_mask.cpu().numpy()).all()
        self.assertEqual(is_equal, True)

    # @unittest.skip('*')
    def test_algo_005(self):
        """ tensor:1, ascend_optimized:false, prune_group:1
            num is too little
        """
        # prepare input
        tensor_list = [torch.randn([16, 4, 3, 3])]
        prune_axises = [0]
        algo_param = {
            'prune_ratio': 0.3,
            'ascend_optimized': False,
            'prune_group': 1
        }
        # call bcp_op
        prune_mask = bcp(tensor_list, prune_axises, algo_param['prune_ratio'], algo_param['prune_group'], algo_param['ascend_optimized'])
        # call numpy bcp
        prune_mask_list = BcpFaker(tensor_list, prune_axises, algo_param).do()
        # compare
        is_equal = (prune_mask_list==prune_mask.cpu().numpy()).all()
        self.assertEqual(is_equal, True)

    @unittest.skip('*')
    def test_algo_006(self):
        """ tensor:2, ascend_optimized:true, prune_group:2"""
        # prepare input
        tensor_list = [torch.randn([160, 4, 3, 3]), torch.randn([4, 160, 3, 3])]
        prune_axises = [0, 1]
        algo_param = {
            'prune_ratio': 0.3,
            'ascend_optimized': True,
            'prune_group': 2
        }
        # call bcp_op
        prune_mask = bcp(tensor_list, prune_axises, algo_param['prune_ratio'], algo_param['prune_group'], algo_param['ascend_optimized'])
        # call numpy bcp
        prune_mask_list = BcpFaker(tensor_list, prune_axises, algo_param).do()
        # compare
        is_equal = (prune_mask_list==prune_mask.cpu().numpy()).all()
        print('prune_mask_list', prune_mask_list)
        print('prune_mask', prune_mask)
        print('equal:', prune_mask_list==prune_mask.cpu().numpy())
        print('numpy_num', sum(prune_mask_list[0:80]), sum(prune_mask_list[80:160]))
        print('op_num', prune_mask.cpu().numpy()[0:80].sum(), prune_mask.cpu().numpy()[80:160].sum())
        self.assertEqual(is_equal, True)

class BcpFaker:
    def __init__(self, tensor_list, prune_axises, prune_param):
        self.tensor_list = tensor_list
        self.prune_axises = prune_axises
        self.prune_param = prune_param

    def do(self):
        # 计算norm值
        norm_list = list()
        for data_tensor, cout_axis in zip(self.tensor_list, self.prune_axises):
            norm_val = self.cal_norm(data_tensor, cout_axis)
            norm_list.append(norm_val)
        # 计算cout
        prune_mask, prune_index = self.cal_channel_index(norm_list,
            self.prune_param['prune_ratio'], self.prune_param['ascend_optimized'], self.prune_param['prune_group'])

        return prune_mask

    @staticmethod
    def cal_channel_index(norm_list, prune_ratio, ascend_optimized, prune_group):
        '''norm_list 要分组挑选
        '''
        def cal_prune_num(num, prune_ratio, ascend_optimized, prune_group):
            remain_num = num - round(num * prune_ratio)
            if ascend_optimized:
                remain_num = min(math.ceil(remain_num/16)*16, num)
                if remain_num == 0:
                    remain_num = min(num, 16)
            if prune_group > 1:
                remain_num = min(math.floor(remain_num/prune_group)*prune_group, num)
            if remain_num == 0:
                remain_num = prune_group
            prune_num = num - remain_num
            return prune_num

        norm = np.mean(np.array(norm_list), axis=0)
        prune_index = list()
        group_len = norm.shape[0] // prune_group
        prune_nums = cal_prune_num(norm.shape[0], prune_ratio, ascend_optimized, prune_group)
        prune_nums_grp = prune_nums // prune_group
        # no prune
        if prune_nums == 0:
            prune_index = []
            prune_cout = [1 for idx in range(norm.shape[0])]
            return prune_cout, prune_index
        # do prune
        for i in range(prune_group):
            start_idx = i*group_len
            norm_grp = norm[start_idx:(i+1)*group_len]
            sort_index = np.argsort(norm_grp) + start_idx
            prune_index.extend(sort_index[0:prune_nums_grp].tolist())
        prune_index.sort()
        prune_mask = [0 if idx in prune_index else 1 for idx in range(norm.shape[0])]
        return prune_mask, prune_index

    @staticmethod
    def cal_norm(wts_tensor, axis):
        norm_algo = 'l2_norm'
        ord_map = {'l1_norm': 1, 'l2_norm': 2}

        wts_np = wts_tensor.cpu().numpy()
        if axis != 0:
            axis_list = [idx for idx in range(wts_np.ndim)]
            axis_list[0] = axis_list[axis]
            axis_list[axis] = 0
            wts_np = np.transpose(wts_np, axis_list)
        wts_np = wts_np.reshape([wts_np.shape[0], -1])
        norm = np.linalg.norm(wts_np, ord=ord_map[norm_algo], axis=1, keepdims=False)
        norm = norm/(wts_np[0].size)
        norm = (norm - norm.min()) / (norm.max() - norm.min())
        return norm

    def prune_cout_to_prune_index(prune_cout):
        prune_index = [idx for idx,val in enumerate(prune_cout) if val==0]
        return prune_index
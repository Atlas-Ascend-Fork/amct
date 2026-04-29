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
import unittest
import torch
from amct_pytorch.utils.data_utils import float_to_fp4e2m1


def gloden_float_cast_to_float4_e2m1(tensor):
    res = torch.zeros_like(tensor)
    sign = torch.sign(tensor)
    absvalues = torch.abs(tensor)

    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isposinf(tensor)
    neg_inf_mask = torch.isneginf(tensor)

    res[absvalues <= 0.25] = 0
    res[(absvalues > 0.25) & (absvalues < 0.75)] = 0.5
    res[(absvalues >= 0.75) & (absvalues <= 1.25)] = 1.0
    res[(absvalues > 1.25) & (absvalues < 1.75)] = 1.5
    res[(absvalues >= 1.75) & (absvalues <= 2.5)] = 2.0
    res[(absvalues > 2.5) & (absvalues < 3.5)] = 3.0
    res[(absvalues >= 3.5) & (absvalues <= 5.0)] = 4.0
    res[absvalues > 5.0] = 6.0

    res *= sign
    res[nan_mask] = float('nan')
    res[inf_mask] = float('inf')
    res[neg_inf_mask] = float('-inf')

    return res


class Testfloat2fp4e2m1(unittest.TestCase):
    '''
    UT FOR DATA TRANSFORMATION FROM FLOAT 2 FP4E2M1
    '''
    @classmethod
    def setUpClass(cls):
        print('Testfloat2fp4e2m1 START!')

    @classmethod
    def tearDownClass(cls):
        print('Testfloat2fp4e2m1 END!')

    def test_float16_tensor_trans_2_fp4e2m1_success(self):
        for i in range(0, 10):
            torch.manual_seed(i)
            ori_tensor = torch.randn((100, 100), dtype=torch.float16)
            test_fp4 = float_to_fp4e2m1(ori_tensor)
            golden_fp4 = gloden_float_cast_to_float4_e2m1(ori_tensor)

            self.assertEqual(test_fp4.tolist(), golden_fp4.tolist())
            self.assertEqual(test_fp4.dtype, torch.float16)

    def test_bfloat16_tensor_trans_2_fp4e2m1_success(self):
        for i in range(10, 20):
            torch.manual_seed(i)
            ori_tensor = torch.randn((100, 100), dtype=torch.bfloat16)
            test_fp4 = float_to_fp4e2m1(ori_tensor)
            golden_fp4 = gloden_float_cast_to_float4_e2m1(ori_tensor)

            self.assertEqual(test_fp4.tolist(), golden_fp4.tolist())
            self.assertEqual(test_fp4.dtype, torch.bfloat16)

import unittest
import torch

from amct_pytorch.utils.quant_util import quant_tensor, quant_dequant_tensor


class TestQuantTensor(unittest.TestCase):
    '''
    UT FOR QUANT TENSOR FROM FLOAT 2 LOW BIT
    '''
    @classmethod
    def setUpClass(cls):
        print('TestQuantTensor START!')

    @classmethod
    def tearDownClass(cls):
        print('TestQuantTensor END!')

    def test_quant_tensor_float16_2_fp4e2m1_success(self):
        torch.manual_seed(0)
        ori_tensor = torch.tensor((100, 100), dtype=torch.float16)
        scale = torch.tensor([1.0])
        fp4_tensor = quant_tensor(ori_tensor, 'float4_e2m1', scale)
        self.assertEqual(fp4_tensor.dtype, torch.float32)

    def test_quant_tensor_bfloat16_2_fp4e2m1_success(self):
        torch.manual_seed(0)
        ori_tensor = torch.tensor({100, 100}, dtype=torch.bfloat16)
        scale = torch.tensor([1.0])
        fp4_tensor = quant_tensor(ori_tensor, 'float4_e2m1', scale)
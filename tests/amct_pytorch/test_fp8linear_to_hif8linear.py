import copy 
import unittest
import torch.nn as nn
import transformers.models.deepseek_v3.modeling_deepseek_v3 as deepseek_module
from transformers.integrations.finegrained_fp8 import FP8Linear
import torch
import transformers
import numpy as np
from amct_pytorch import quantize, convert

torch.manual_seed(0)


class FP8Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # 第一层 FP8 Linear
        self.layer1 = FP8Linear(in_dim, hidden_dim, has_bias=True)
        # 第二层 FP8 Linear
        self.layer2 = FP8Linear(hidden_dim, out_dim, has_bias=False)

    def forward(self, x):      
        x = self.layer1(x)
        x = torch.relu(x) # 中间加个激活函数演示
        x = self.layer2(x)
        return x


class TestFP8HIF8(unittest.TestCase):
    '''    ST FOR FP8HIF8 ALGORITHM    '''    
    @classmethod
    def setUpClass(cls):        
        input_dim, hidden_dim, output_dim = 128, 256, 64
        batch = 8
        cls.test_model = FP8Model(input_dim, hidden_dim, output_dim).to(torch.bfloat16)
        cls.test_inputs = torch.randn(batch, input_dim).to(torch.bfloat16)        
        cls.ori_out = cls.test_model(cls.test_inputs)       
        print('TestFP8HIF8 START!')   
    
    @classmethod    
    def tearDownClass(cls):        
        print('TestFP8HIF8 END!') 
       
    def test_fp8_hif8_success(self):     
        model = copy.deepcopy(self.test_model)        
        quantize(model)        
        print(model)        
        self.assertEqual(list(model.state_dict().keys()), list(self.test_model.state_dict().keys()))        
        self.assertEqual(type(model.layer1).__name__, 'FP8Linear')        
        self.assertEqual(type(model.layer2).__name__, 'FP8Linear')        
        model = model.npu()        
        convert(model)        
        self.assertEqual(type(model.layer1).__name__, 'NpuHIF8Linear')       
        self.assertEqual(type(model.layer2).__name__, 'NpuHIF8Linear')        
        print(model)        
        quant_out = model(self.test_inputs)        

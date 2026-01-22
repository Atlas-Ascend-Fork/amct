# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import unittest

from amct_pytorch import quantize, convert

# Customize accordingly
LLAMA2_7B_MODEL_PATH = "meta-llama/Llama-2-7b-hf"
RUN_SKIPPED = os.getenv('RUN_SKIPPED_TESTS', 'False').lower() == 'true'


class TestFlatQuant(unittest.TestCase):
    '''
    ST FOR FLATQUANT ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        print('TestFlatQuant START!')

    @classmethod
    def tearDownClass(cls):
        print('TestFlatQuant END!')

    def setUp(self):
        pass
 
    def tearDown(self):
        pass

    @unittest.skipIf(not RUN_SKIPPED, "Skip by default due to requiring the actual Llama model")
    def test_int4_tensor_sym_flatquant_success(self):
        cfg = {
            'batch_num': 4,
            'quant_cfg': {
                'inputs': {
                    'enable_quant': True,
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'token'
                },
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {
                'flatquant_attn': {
                    'use_kcache_quant': False,
                    'k_bits': 16,
                    'use_vcache_quant': False,
                    'v_bits': 16,
                    'use_o_quant': False
                },
                'flatquant_attn_spda': {},
                'flatquant_mlp': {}
            },
            'skip_layers': {'lm_head'}
        }

        import transformers
        config = transformers.LlamaConfig.from_pretrained(LLAMA2_7B_MODEL_PATH, attn_implementation='eager')
        model = transformers.LlamaForCausalLM.from_pretrained(
            LLAMA2_7B_MODEL_PATH, torch_dtype='auto', config=config,
            use_auth_token=None, low_cpu_mem_usage=True)
        model.seqlen = 2048
        print(f'---> Loading {LLAMA2_7B_MODEL_PATH} Model with seq_len: {model.seqlen}')

        quantize(model, cfg)
        self.assertEqual(type(model.model.layers[0].self_attn).__name__, 'FlatQuantAttention')
        self.assertEqual(type(model.model.layers[0].mlp).__name__, 'FlatQuantMLP')
        convert(model)
        self.assertEqual(type(model.model.layers[0].self_attn).__name__, 'NpuFlatQuantAttention')
        self.assertEqual(type(model.model.layers[0].mlp).__name__, 'NpuFlatQuantMLP')
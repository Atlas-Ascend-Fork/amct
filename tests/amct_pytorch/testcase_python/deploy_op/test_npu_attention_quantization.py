#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
import sys
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class MockDynamicCache:
    """Mock DynamicCache for testing KV cache update hook."""
    def __init__(self):
        self._cache = {}
        self._update_call_count = 0
        self._original_update = self.update

    def update(self, key_states, value_states, layer_idx, **kwargs):
        self._update_call_count += 1
        self._cache[layer_idx] = (key_states, value_states)
        return key_states, value_states

    def get_call_count(self):
        return self._update_call_count

    def get_cache(self, layer_idx):
        return self._cache.get(layer_idx, (None, None))


class MockQuantModule:
    """Mock quantization module for testing attention quantization."""
    def __init__(self, scale_k=1.0, scale_v=1.0, dtype=torch.float32, device='cpu'):
        self.scale_k = torch.tensor([scale_k], dtype=dtype, device=device)
        self.scale_v = torch.tensor([scale_v], dtype=dtype, device=device)
        self.q_proj = MagicMock()
        self.q_proj.dtype = dtype
        self.weight = torch.randn(64, 64, dtype=dtype, device=device)
        self.weight.device = torch.device(device)


class MockTorchNPU:
    """Mock torch_npu module for testing without NPU hardware."""
    hifloat8 = torch.float8_e4m3fn

    @staticmethod
    def npu_quantize(input_tensor, scale, dtype=None, div_mode=False, axis=None):
        quantized = input_tensor / scale
        if dtype == 'hifloat8' or dtype == torch.float8_e4m3fn:
            quantized = quantized.to(torch.float8_e4m3fn)
        else:
            quantized = quantized.to(torch.int8)
        return quantized

    @staticmethod
    def npu_anti_quantize(input_tensor, scale, dtype=None, div_mode=False):
        dequantized = input_tensor.to(torch.float32) * scale
        return dequantized


class MockAttentionModule:
    """Mock original attention module for testing forward pass."""
    def __init__(self):
        self.forward_call_count = 0

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        self.forward_call_count += 1
        return torch.randn(hidden_states.shape[0], hidden_states.shape[1], 64)

    def __call__(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        return self.forward(hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values, **kwargs)


with patch.dict('sys.modules', {'torch_npu': MockTorchNPU(), 'transformer.cache_utils': MagicMock()}):
    from amct_pytorch.deploy_op.npu_quantization_deepseekv3_attention import NpuDeepseekV3AttentionQuant
    from amct_pytorch.deploy_op.npu_quantization_longcat_flashmla import NpuLongcatFlashMLA


class TestNpuDeepseekV3AttentionQuantInit(unittest.TestCase):
    """Unit tests for NpuDeepseekV3AttentionQuant initialization."""

    def setUp(self):
        self.quant_module = MockQuantModule(scale_k=0.5, scale_v=0.5)

    def test_init_success_basic(self):
        quant_module = MockQuantModule()
        module = NpuDeepseekV3AttentionQuant(quant_module)
        self.assertEqual(module.output_dtype, torch.float32)
        self.assertEqual(module.device, torch.device('cpu'))
        self.assertTrue(torch.equal(module.scale_k, quant_module.scale_k))
        self.assertTrue(torch.equal(module.scale_v, quant_module.scale_v))

    def test_init_success_custom_dtype(self):
        quant_module = MockQuantModule(dtype=torch.bfloat16)
        module = NpuDeepseekV3AttentionQuant(quant_module)
        self.assertEqual(module.output_dtype, torch.bfloat16)

    def test_init_success_custom_scale(self):
        scale_k = 2.0
        scale_v = 3.0
        quant_module = MockQuantModule(scale_k=scale_k, scale_v=scale_v)
        module = NpuDeepseekV3AttentionQuant(quant_module)
        self.assertEqual(module.scale_k.item(), scale_k)
        self.assertEqual(module.scale_v.item(), scale_v)

    def test_init_preserves_scale_tensors(self):
        quant_module = MockQuantModule(scale_k=torch.tensor([1.5]), scale_v=torch.tensor([2.5]))
        module = NpuDeepseekV3AttentionQuant(quant_module)
        self.assertIsInstance(module.scale_k, torch.Tensor)
        self.assertIsInstance(module.scale_v, torch.Tensor)


class TestNpuDeepseekV3AttentionQuantHookKVStates(unittest.TestCase):
    """Unit tests for _hook_kv_states method."""

    def setUp(self):
        self.quant_module = MockQuantModule(scale_k=1.0, scale_v=1.0)
        self.module = NpuDeepseekV3AttentionQuant(self.quant_module)

    def test_hook_kv_states_modifies_update_function(self):
        cache = MockDynamicCache()
        original_update = cache.update
        self.module._hook_kv_states(cache)
        self.assertNotEqual(cache.update, original_update)

    def test_hook_kv_states_quantizes_key_states(self):
        cache = MockDynamicCache()
        self.module._hook_kv_states(cache)
        key_states = torch.randn(1, 4, 8, dtype=torch.float32)
        value_states = torch.randn(1, 4, 8, dtype=torch.float32)
        result_k, result_v = cache.update(key_states, value_states, layer_idx=0)
        self.assertEqual(result_k.dtype, torch.float32)
        self.assertEqual(result_v.dtype, torch.float32)

    def test_hook_kv_states_returns_dequantized_tensors(self):
        cache = MockDynamicCache()
        self.module._hook_kv_states(cache)
        key_states = torch.randn(1, 4, 8, dtype=torch.float32)
        value_states = torch.randn(1, 4, 8, dtype=torch.float32)
        result_k, result_v = cache.update(key_states, value_states, layer_idx=0)
        self.assertEqual(result_k.shape, key_states.shape)
        self.assertEqual(result_v.shape, value_states.shape)

    def test_hook_kv_states_different_layer_indices(self):
        cache = MockDynamicCache()
        self.module._hook_kv_states(cache)
        for layer_idx in range(3):
            key_states = torch.randn(1, 4, 8)
            value_states = torch.randn(1, 4, 8)
            result_k, result_v = cache.update(key_states, value_states, layer_idx=layer_idx)
            self.assertEqual(cache.get_call_count(), layer_idx + 1)


class TestNpuLongcatFlashMLAInit(unittest.TestCase):
    """Unit tests for NpuLongcatFlashMLA initialization."""

    def setUp(self):
        self.quant_module = MockQuantModule(scale_k=0.5, scale_v=0.5)

    def test_init_success_basic(self):
        quant_module = MockQuantModule()
        module = NpuLongcatFlashMLA(quant_module)
        self.assertEqual(module.output_dtype, torch.float32)
        self.assertEqual(module.device, torch.device('cpu'))
        self.assertTrue(torch.equal(module.scale_k, quant_module.scale_k))
        self.assertTrue(torch.equal(module.scale_v, quant_module.scale_v))

    def test_init_success_custom_dtype(self):
        quant_module = MockQuantModule(dtype=torch.float16)
        module = NpuLongcatFlashMLA(quant_module)
        self.assertEqual(module.output_dtype, torch.float16)

    def test_init_success_custom_scale(self):
        scale_k = 1.5
        scale_v = 2.5
        quant_module = MockQuantModule(scale_k=scale_k, scale_v=scale_v)
        module = NpuLongcatFlashMLA(quant_module)
        self.assertEqual(module.scale_k.item(), scale_k)
        self.assertEqual(module.scale_v.item(), scale_v)


class TestNpuLongcatFlashMLAHookKVStates(unittest.TestCase):
    """Unit tests for NpuLongcatFlashMLA _hook_kv_states method."""

    def setUp(self):
        self.quant_module = MockQuantModule(scale_k=1.0, scale_v=1.0)
        self.module = NpuLongcatFlashMLA(self.quant_module)

    def test_hook_kv_states_modifies_update_function(self):
        cache = MockDynamicCache()
        original_update = cache.update
        self.module._hook_kv_states(cache)
        self.assertNotEqual(cache.update, original_update)

    def test_hook_kv_states_quantizes_and_dequantizes(self):
        cache = MockDynamicCache()
        self.module._hook_kv_states(cache)
        key_states = torch.randn(2, 4, 16, dtype=torch.float32)
        value_states = torch.randn(2, 4, 16, dtype=torch.float32)
        result_k, result_v = cache.update(key_states, value_states, layer_idx=0)
        self.assertEqual(result_k.shape, key_states.shape)
        self.assertEqual(result_v.shape, value_states.shape)


class TestAttentionQuantizationSystem(unittest.TestCase):
    """System tests for attention quantization workflow."""

    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_attention_quant_temp')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.temp_folder):
            os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        self.quant_module = MockQuantModule(scale_k=0.01, scale_v=0.01)
        np.random.seed(42)

    def test_system_deepseekv3_attention_quant_workflow(self):
        module = NpuDeepseekV3AttentionQuant(self.quant_module)
        module.ori_module = MockAttentionModule()
        cache = MockDynamicCache()
        module._hook_kv_states(cache)
        hidden_states = torch.randn(1, 8, 64)
        position_embeddings = torch.randn(1, 8, 64)
        attention_mask = torch.ones(1, 1, 8, 8)
        output = module.forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=cache
        )
        self.assertEqual(output.shape[0], hidden_states.shape[0])
        self.assertEqual(output.shape[1], hidden_states.shape[1])

    def test_system_longcat_flashmla_workflow(self):
        module = NpuLongcatFlashMLA(self.quant_module)
        module.ori_module = MockAttentionModule()
        cache = MockDynamicCache()
        module._hook_kv_states(cache)
        hidden_states = torch.randn(2, 16, 64)
        position_embeddings = torch.randn(2, 16, 64)
        attention_mask = torch.ones(2, 1, 16, 16)
        output = module.forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=cache
        )
        self.assertEqual(output.shape[0], hidden_states.shape[0])

    def test_system_kv_cache_quantization_accuracy(self):
        scale = 0.1
        quant_module = MockQuantModule(scale_k=scale, scale_v=scale)
        module = NpuDeepseekV3AttentionQuant(quant_module)
        cache = MockDynamicCache()
        module._hook_kv_states(cache)
        original_key = torch.randn(1, 4, 32)
        original_value = torch.randn(1, 4, 32)
        result_k, result_v = cache.update(original_key, original_value, layer_idx=0)
        def compute_similarity(a, b):
            a_flat = a.flatten().numpy()
            b_flat = b.flatten().numpy()
            dot = np.dot(a_flat, b_flat)
            norm_a = np.linalg.norm(a_flat)
            norm_b = np.linalg.norm(b_flat)
            return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
        similarity_k = compute_similarity(original_key, result_k)
        similarity_v = compute_similarity(original_value, result_v)
        self.assertGreater(similarity_k, 0.95)
        self.assertGreater(similarity_v, 0.95)

    def test_system_multiple_forward_calls(self):
        module = NpuDeepseekV3AttentionQuant(self.quant_module)
        mock_attention = MockAttentionModule()
        module.ori_module = mock_attention
        cache = MockDynamicCache()
        module._hook_kv_states(cache)
        for i in range(3):
            hidden_states = torch.randn(1, 8, 64)
            position_embeddings = torch.randn(1, 8, 64)
            attention_mask = torch.ones(1, 1, 8, 8)
            module.forward(hidden_states, position_embeddings, attention_mask, past_key_values=cache)
        self.assertEqual(mock_attention.forward_call_count, 3)

    def test_system_null_past_key_values_creates_cache(self):
        module = NpuDeepseekV3AttentionQuant(self.quant_module)
        module.ori_module = MockAttentionModule()
        with patch('transformer.cache_utils.DynamicCache', return_value=MockDynamicCache()):
            hidden_states = torch.randn(1, 8, 64)
            position_embeddings = torch.randn(1, 8, 64)
            attention_mask = torch.ones(1, 1, 8, 8)
            output = module.forward(hidden_states, position_embeddings, attention_mask, past_key_values=None)
            self.assertIsNotNone(output)


class TestAttentionQuantizationEdgeCases(unittest.TestCase):
    """Edge case tests for attention quantization."""

    def setUp(self):
        self.quant_module = MockQuantModule(scale_k=1.0, scale_v=1.0)

    def test_edge_case_zero_scale(self):
        quant_module = MockQuantModule(scale_k=0.0, scale_v=0.0)
        module = NpuDeepseekV3AttentionQuant(quant_module)
        cache = MockDynamicCache()
        module._hook_kv_states(cache)
        key_states = torch.ones(1, 4, 8)
        value_states = torch.ones(1, 4, 8)
        with self.assertRaises((RuntimeError, ZeroDivisionError)):
            cache.update(key_states, value_states, layer_idx=0)

    def test_edge_case_large_tensor(self):
        module = NpuDeepseekV3AttentionQuant(self.quant_module)
        cache = MockDynamicCache()
        module._hook_kv_states(cache)
        key_states = torch.randn(1, 128, 1024)
        value_states = torch.randn(1, 128, 1024)
        result_k, result_v = cache.update(key_states, value_states, layer_idx=0)
        self.assertEqual(result_k.shape, key_states.shape)
        self.assertEqual(result_v.shape, value_states.shape)

    def test_edge_case_negative_values(self):
        module = NpuDeepseekV3AttentionQuant(self.quant_module)
        cache = MockDynamicCache()
        module._hook_kv_states(cache)
        key_states = torch.randn(1, 4, 8) * -10
        value_states = torch.randn(1, 4, 8) * -10
        result_k, result_v = cache.update(key_states, value_states, layer_idx=0)
        self.assertEqual(result_k.shape, key_states.shape)

    def test_edge_case_single_element_tensor(self):
        module = NpuDeepseekV3AttentionQuant(self.quant_module)
        cache = MockDynamicCache()
        module._hook_kv_states(cache)
        key_states = torch.randn(1, 1, 1)
        value_states = torch.randn(1, 1, 1)
        result_k, result_v = cache.update(key_states, value_states, layer_idx=0)
        self.assertEqual(result_k.shape, key_states.shape)

    def test_edge_case_multilayer_cache(self):
        module = NpuDeepseekV3AttentionQuant(self.quant_module)
        cache = MockDynamicCache()
        module._hook_kv_states(cache)
        for layer_idx in range(5):
            key_states = torch.randn(1, 4, 32)
            value_states = torch.randn(1, 4, 32)
            result_k, result_v = cache.update(key_states, value_states, layer_idx=layer_idx)
            self.assertEqual(cache.get_call_count(), layer_idx + 1)


if __name__ == '__main__':
    unittest.main()
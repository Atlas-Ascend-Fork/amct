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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from types import SimpleNamespace

import torch
import torch.nn as nn

from amct_pytorch.algorithms.quant import register_algorithms
from amct_pytorch.common.models.llm.deepseek.deepseek_v4.quant_module import (
    QuantV4Attention,
)
from amct_pytorch.common.optimization.blockwise_solver import BlockwiseSolver
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_matmul import QuantizedMatmul


register_algorithms()
register_dtype()


def _quant_args(algos=None, quant_target=("attn-linear",)):
    return SimpleNamespace(
        algos=list(algos or []),
        quant_dtype="int",
        w_bits=8,
        a_bits=8,
        quant_target=list(quant_target),
        device="cpu",
        base_lr=1e-3,
        optimizer="adamw",
        weight_decay=0.0,
        momentum=0.9,
        lr_scheduler="cosine",
        min_lr=0.0,
        lr_step_size=1,
        lr_gamma=0.1,
        epochs=1,
        nsamples=1,
        cali_bsz=1,
        bit_policy=BitPolicy({
            "attn-linear": {
                "wq_a": {"w_bits": 8, "a_bits": 8},
                "wq_b": {"w_bits": 8, "a_bits": 8},
                "wkv": {"w_bits": 8, "a_bits": 8},
                "wo_a": {"w_bits": 8, "a_bits": 8},
                "wo_b": {"w_bits": 8, "a_bits": 8},
                "comp_wkv": {"w_bits": 8, "a_bits": 8},
                "comp_wgate": {"w_bits": 8, "a_bits": 8},
                "idx_wq_b": {"w_bits": 8, "a_bits": 8},
                "idx_weights_proj": {"w_bits": 8, "a_bits": 8},
            },
            "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
        }),
    )


class _FakeV4Compressor(nn.Module):
    def __init__(self, dim, compress_ratio, head_dim, rope_head_dim, rotate=False):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        coff = 1 + self.overlap
        self.norm = nn.Identity()
        self.ape = nn.Parameter(torch.zeros(compress_ratio, coff * head_dim))
        self.freqs_cis = None
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False)


class _FakeV4Indexer(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.n_local_heads = attention.n_local_heads
        self.head_dim = attention.head_dim
        self.rope_head_dim = attention.rope_head_dim
        self.compress_ratio = attention.compress_ratio
        self.index_topk = 1
        self.softmax_scale = 1.0
        self.freqs_cis = None
        self.wq_b = nn.Linear(
            attention.q_lora_rank,
            self.n_local_heads * self.head_dim,
            bias=False,
        )
        self.weights_proj = nn.Linear(attention.dim, self.n_local_heads, bias=False)
        self.compressor = _FakeV4Compressor(
            attention.dim,
            self.compress_ratio,
            self.head_dim,
            self.rope_head_dim,
            rotate=True,
        )


class _FakeV4Attention(nn.Module):
    def __init__(self, compress_ratio=0, with_indexer=False):
        super().__init__()
        self.layer_id = 0
        self.dim = 16
        self.n_heads = 2
        self.n_local_heads = 2
        self.q_lora_rank = 8
        self.o_lora_rank = 4
        self.head_dim = 4
        self.rope_head_dim = 4
        self.n_local_groups = 2
        self.window_size = 4
        self.compress_ratio = compress_ratio
        self.eps = 1e-6
        self.softmax_scale = 1.0
        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads))
        self.q_norm = nn.Identity()
        self.kv_norm = nn.Identity()
        self.register_buffer("freqs_cis", torch.ones(8, self.rope_head_dim // 2, dtype=torch.complex64))

        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.wq_b = nn.Linear(
            self.q_lora_rank,
            self.n_local_heads * self.head_dim,
            bias=False,
        )
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.wo_a = nn.Linear(
            self.head_dim,
            self.n_local_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_local_groups * self.o_lora_rank, self.dim, bias=False)
        if self.compress_ratio:
            self.compressor = _FakeV4Compressor(
                self.dim,
                self.compress_ratio,
                self.head_dim,
                self.rope_head_dim,
            )
            self.indexer = _FakeV4Indexer(self) if with_indexer else None


class _RecordingMatmul(nn.Module):
    def __init__(self, transpose_right=True):
        super().__init__()
        self.calls = 0
        self.transpose_right = transpose_right

    def forward(self, left, right):
        self.calls += 1
        if not self.transpose_right:
            return torch.matmul(left, right)
        return torch.matmul(left, right.transpose(-2, -1))


def test_quant_v4_attention_flatquant_exposes_trainable_structure_params():
    qattn = QuantV4Attention(_quant_args(algos=["flatquant"]), _FakeV4Attention())
    solver = BlockwiseSolver(_quant_args(algos=["flatquant"]), layer_idx=0, model=qattn)

    param_groups = solver._collect_trainable_param_groups(qattn)

    assert param_groups
    names = {name for name, param in qattn.named_parameters() if param.requires_grad}
    assert any(name.startswith("input_transform.") for name in names)
    assert any(name.startswith("out_transform.") for name in names)


def test_quant_v4_attention_attn_cache_uses_quantized_matmul_path():
    qattn = QuantV4Attention(_quant_args(quant_target=["attn-cache"]), _FakeV4Attention())

    assert isinstance(qattn.qk_matmul, QuantizedMatmul)
    assert isinstance(qattn.pv_matmul, QuantizedMatmul)
    assert isinstance(qattn.q_cache_quantizer, nn.Identity)
    assert isinstance(qattn.k_cache_quantizer, nn.Identity)

    qattn.qk_matmul = _RecordingMatmul()
    qattn.pv_matmul = _RecordingMatmul(transpose_right=False)
    query = torch.randn(1, 2, 2, 4)
    kv = torch.randn(1, 2, 4)
    topk_idxs = torch.tensor([[[0], [1]]], dtype=torch.int32)

    output = qattn.sparse_attn(query, kv, qattn.attn_sink.float(), topk_idxs, qattn.softmax_scale)

    assert output.shape == query.shape
    assert qattn.qk_matmul.calls == 1
    assert qattn.pv_matmul.calls == 1


def test_quant_v4_attention_compressed_indexer_forward_aligns_topk_device():
    qattn = QuantV4Attention(
        _quant_args(quant_target=[]),
        _FakeV4Attention(compress_ratio=4, with_indexer=True),
    )
    qattn.qk_matmul = _RecordingMatmul()
    qattn.pv_matmul = _RecordingMatmul(transpose_right=False)
    x = torch.randn(1, 8, 16)

    output = qattn(x)

    assert qattn.compress_ratio > 0
    assert qattn.indexer is not None
    assert output.shape == x.shape
    assert output.device == x.device
    assert qattn.qk_matmul.calls == 1
    assert qattn.pv_matmul.calls == 1

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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from amct_pytorch.quantization.diagnostics import (
    QuantDiagnosisConfig,
    diagnose_quantization_error,
)
from amct_pytorch.quantization.diagnostics.metrics import (
    cosine_similarity,
    max_error,
    mse,
    nmse,
    qsnr,
    relative_error,
)
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer


QWEN3_DIR = Path("/mnt/workspace/gitCode/yaoguangxiu/models/Qwen3-0.6B")
QWEN36_DIR = Path("/mnt/workspace/gitCode/yaoguangxiu/models/Qwen3.6-35B-A3B")


def _make_args(model_path: Path, quant_target: list[str]):
    return SimpleNamespace(
        model=str(model_path),
        quant_target=quant_target,
        quant_dtype="int",
        device="cpu",
        w_bits=8,
        a_bits=8,
        q_bits=8,
        k_bits=8,
        p_bits=8,
        v_bits=8,
        algos=[],
        data_dir="/tmp",
        output_dir="/tmp",
        attn_linear_param_dir="",
        attn_cache_param_dir="",
        moe_mlp_param_dir="",
    )


def _sample_ids(vocab_size=16):
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.long) % vocab_size


def test_metrics_basic():
    ref = torch.tensor([1.0, 2.0, 3.0])
    cand = torch.tensor([1.0, 2.0, 4.0])
    assert mse(ref, cand) > 0
    assert nmse(ref, cand) > 0
    assert relative_error(ref, cand) > 0
    assert max_error(ref, cand) == 1.0
    assert cosine_similarity(ref, cand) <= 1.0
    assert qsnr(ref, cand) > 0


class _ToyAttn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_quant = ActivationQuantizer(args, bits=8)
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden_states):
        return self.linear(self.input_quant(hidden_states))


class _ToyMlp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_quant = ActivationQuantizer(args, bits=8)
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden_states):
        return self.linear(self.input_quant(hidden_states))


class _ToyBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attn = _ToyAttn(args)
        self.mlp = _ToyMlp(args)

    def forward(self, hidden_states):
        return self.mlp(self.self_attn(hidden_states))


class _ToyAdapter:
    def __init__(self):
        self.args = _make_args(Path("/tmp/toy"), ["attn-linear", "mlp"])
        self.num_layers = 1
        self._float_block = _ToyBlock(self.args)

    def do_embedding_forward(self, samples):
        return samples

    def get_block_forward_kwargs(self):
        return {}

    def block(self, _layer_idx):
        return self._float_block

    def build_quant_block(self, _layer_idx):
        quant_block = _ToyBlock(self.args)
        quant_block.load_state_dict(self._float_block.state_dict())
        return quant_block


def test_diagnosis_captures_activation_quantizer_input_outliers():
    adapter = _ToyAdapter()
    sample = torch.tensor([[[1.0, 1.0, 1.0, 32.0]]])

    report = diagnose_quantization_error(
        adapter,
        [sample],
        QuantDiagnosisConfig(layer_limit=1, sample_limit=1),
    )

    assert report.input_outlier_metrics
    assert all(metric.max_to_p999 >= 1.0 for metric in report.input_outlier_metrics)


@pytest.mark.skipif(not QWEN3_DIR.exists(), reason="local Qwen3-0.6B model missing")
def test_qwen3_dense_diagnosis_smoke():
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    adapter = Qwen3(_make_args(QWEN3_DIR, ["attn-linear", "mlp"]))
    cfg = QuantDiagnosisConfig(layer_limit=1, sample_limit=1)
    report = diagnose_quantization_error(adapter, [_sample_ids(adapter.config.vocab_size)], cfg)

    assert report.block_metrics
    assert report.layer_metrics
    assert report.mixed_precision_plan
    assert "layers.0" in report.mixed_precision_plan


@pytest.mark.skipif(not QWEN36_DIR.exists(), reason="local Qwen3.6-35B-A3B model missing")
def test_qwen36_moe_diagnosis_smoke():
    from amct_pytorch.common.models.llm.qwen.qwen3_6 import Qwen3_6Moe

    adapter = Qwen3_6Moe(_make_args(QWEN36_DIR, ["attn-linear", "moe"]))
    cfg = QuantDiagnosisConfig(layer_limit=1, sample_limit=1)
    report = diagnose_quantization_error(adapter, [_sample_ids(adapter.config.vocab_size)], cfg)

    assert report.block_metrics
    assert report.layer_metrics
    assert report.mixed_precision_plan
    assert any(k.endswith(".moe") for k in report.mixed_precision_plan)
    assert report.summary()

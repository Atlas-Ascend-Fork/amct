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
"""Tests for LongcatLite adapter initialization and basic properties."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
from amct_pytorch.common.models.llm.longcat.longcat_lite.longcat_lite import LongcatLite


def _make_args(**extra):
    base = {
        "model": "/tmp/fake_model",
        "quant_target": ["mlp"],
        "device": "cpu",
        "data_dir": "/tmp/fake_data",
        "output_dir": "/tmp/fake_output",
        "quant_dtype": "int8",
        "w_bits": 8,
        "a_bits": 8,
        "algos": [],
        "cali_bsz": 2,
    }
    base.update(extra)
    return SimpleNamespace(**base)


def _mock_hf_deps(mock_config, mock_tokenizer, mock_from_config, mock_init_empty, num_layers=2):
    mock_config.return_value = MagicMock(num_layers=num_layers)
    mock_tokenizer.return_value = MagicMock()
    mock_from_config.return_value = MagicMock()
    mock_init_empty.return_value.__enter__.return_value = None
    mock_init_empty.return_value.__exit__.return_value = None


class TestLongcatLiteAdapter:
    @patch("amct_pytorch.common.models.llm.common.base.AutoConfig.from_pretrained")
    @patch("amct_pytorch.common.models.llm.common.base.AutoTokenizer.from_pretrained")
    @patch("amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM.from_config")
    @patch("amct_pytorch.common.models.llm.common.base.init_empty_weights")
    def test_longcat_lite_init(
        self, mock_init_empty, mock_from_config, mock_tokenizer, mock_config
    ):
        _mock_hf_deps(mock_config, mock_tokenizer, mock_from_config, mock_init_empty, num_layers=4)
        model = LongcatLite(_make_args())
        assert model.num_layers == 4

    @patch("amct_pytorch.common.models.llm.common.base.AutoConfig.from_pretrained")
    @patch("amct_pytorch.common.models.llm.common.base.AutoTokenizer.from_pretrained")
    @patch("amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM.from_config")
    @patch("amct_pytorch.common.models.llm.common.base.init_empty_weights")
    def test_longcat_lite_get_layer_weight_prefix(
        self, mock_init_empty, mock_from_config, mock_tokenizer, mock_config
    ):
        _mock_hf_deps(mock_config, mock_tokenizer, mock_from_config, mock_init_empty)
        model = LongcatLite(_make_args())
        assert model.get_layer_weight_prefix(0) == "model.layers.0."

    @patch("amct_pytorch.common.models.llm.common.base.AutoConfig.from_pretrained")
    @patch("amct_pytorch.common.models.llm.common.base.AutoTokenizer.from_pretrained")
    @patch("amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM.from_config")
    @patch("amct_pytorch.common.models.llm.common.base.init_empty_weights")
    def test_longcat_lite_set_safe_attn_impl(
        self, mock_init_empty, mock_from_config, mock_tokenizer, mock_config
    ):
        _mock_hf_deps(mock_config, mock_tokenizer, mock_from_config, mock_init_empty)
        model = LongcatLite(_make_args())
        assert model.config._attn_implementation == "eager"

    @patch("amct_pytorch.common.models.llm.common.base.AutoConfig.from_pretrained")
    @patch("amct_pytorch.common.models.llm.common.base.AutoTokenizer.from_pretrained")
    @patch("amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM.from_config")
    @patch("amct_pytorch.common.models.llm.common.base.init_empty_weights")
    def test_longcat_lite_load_unit_inputs_uses_named_cached_input(
        self, mock_init_empty, mock_from_config, mock_tokenizer, mock_config, tmp_path
    ):
        _mock_hf_deps(mock_config, mock_tokenizer, mock_from_config, mock_init_empty)
        model = LongcatLite(_make_args())
        expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        torch.save(expected, tmp_path / "block_1_mlp_0_in.pkl")
        unit = make_ptq_unit(
            "mlp",
            "mlp_0",
            layer_idx=1,
            module=None,
            metadata={"input_name": "mlp_0"},
        )

        with patch.object(BaseModel, "load_unit_inputs", return_value=(None, {"mask": "kept"})):
            cached_inps, kwargs = model.load_unit_inputs(str(tmp_path), unit)

        assert torch.equal(cached_inps, expected)
        assert kwargs == {"mask": "kept"}

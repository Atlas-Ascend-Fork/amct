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

import pytest
import torch

from amct_pytorch.common.evaluate.eval_checkpoint import (
    EvalCheckpointManager,
    build_eval_signature,
)
from amct_pytorch.quantization.bit_policy import BitPolicy


def _make_args(tmp_path, **overrides):
    defaults = {
        "model": "/models/qwen",
        "model_name": "qwen3_5",
        "eval_mode": "quant",
        "quant_dtype": "int",
        "quant_target": ["mlp", "attn-linear"],
        "algos": ["lwc", "lac"],
        "seq_len": 4096,
        "seed": 42,
        "bit_config": None,
        "bit_policy": BitPolicy(
            {
                "w_bits": 4,
                "a_bits": 8,
                "attn-linear": {
                    "o_proj": {"w_bits": 8, "a_bits": 16},
                },
                "mlp": {
                    "down_proj": {"w_bits": 4, "a_bits": 16},
                },
                "attn-cache": {"k": 8, "v": 8},
            }
        ),
        "attn_linear_param_dir": "",
        "attn_cache_param_dir": "",
        "moe_mlp_param_dir": "",
        "output_dir": str(tmp_path),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize(
    "parameter, value",
    [("block_size_col", 1), ("scale_dtype", "e8m0"), ("is_per_tensor", True)],
)
def test_signature_distinguishes_quantizer_options(tmp_path, parameter, value):
    first = _make_args(tmp_path, quant_dtype="fp")
    second = _make_args(tmp_path, quant_dtype="fp", **{parameter: value})
    assert build_eval_signature(first, 4) != build_eval_signature(second, 4)


@pytest.mark.parametrize(
    "eval_mode, quant_dtype, algos", [("bf16", "fp", ["lac"]), ("quant", "int", [])]
)
def test_signature_ignores_inactive_quantizer_options(
    tmp_path, eval_mode, quant_dtype, algos
):
    first = _make_args(
        tmp_path, eval_mode=eval_mode, quant_dtype=quant_dtype, algos=algos
    )
    second = _make_args(
        tmp_path,
        eval_mode=eval_mode,
        quant_dtype=quant_dtype,
        algos=algos,
        block_size_col=1,
        scale_dtype="e8m0",
        is_per_tensor=True,
    )
    assert build_eval_signature(first, 4) == build_eval_signature(second, 4)


def test_signature_normalizes_target_order_and_ignores_irrelevant_config(tmp_path):
    first = _make_args(tmp_path)
    second = _make_args(
        tmp_path,
        quant_target=["attn-linear", "mlp", "mlp"],
        bit_policy=BitPolicy(
            {
                "w_bits": 4,
                "a_bits": 8,
                "attn-linear": {
                    "o_proj": {"w_bits": 8, "a_bits": 16},
                },
                "mlp": {
                    "down_proj": {"w_bits": 4, "a_bits": 16},
                },
                "attn-cache": {"q": 4, "k": 4, "p": 4, "v": 4},
            }
        ),
    )

    assert build_eval_signature(first, num_layers=32) == build_eval_signature(
        second, num_layers=32
    )


def test_signature_changes_when_relevant_nested_config_changes(tmp_path):
    first = _make_args(tmp_path, quant_target=["attn-linear"])
    second = _make_args(
        tmp_path,
        quant_target=["attn-linear"],
        bit_policy=BitPolicy(
            {
                "w_bits": 4,
                "a_bits": 8,
                "attn-linear": {
                    "o_proj": {"w_bits": 4, "a_bits": 16},
                },
            }
        ),
    )

    assert build_eval_signature(first, 32) != build_eval_signature(second, 32)


def test_signature_preserves_algorithm_order(tmp_path):
    first = _make_args(tmp_path, algos=["lwc", "lac"])
    second = _make_args(tmp_path, algos=["lac", "lwc"])

    assert build_eval_signature(first, 32) != build_eval_signature(second, 32)


def test_signature_resolves_missing_cache_bits_to_sixteen(tmp_path):
    omitted = _make_args(
        tmp_path,
        quant_target=["attn-cache"],
        bit_policy=BitPolicy({"attn-cache": {"k": 8, "v": 8}}),
    )
    explicit = _make_args(
        tmp_path,
        quant_target=["attn-cache"],
        bit_policy=BitPolicy({"attn-cache": {"q": 16, "k": 8, "p": 16, "v": 8}}),
    )

    assert build_eval_signature(omitted, 32) == build_eval_signature(explicit, 32)


def test_signature_changes_when_relevant_ptq_parameter_changes(tmp_path):
    param_dir = tmp_path / "ptq_params"
    param_dir.mkdir()
    param_file = param_dir / "layer_0_mlp.pt"
    param_file.write_bytes(b"first")
    args = _make_args(
        tmp_path,
        quant_target=["mlp"],
        moe_mlp_param_dir=str(param_dir),
    )
    first = build_eval_signature(args, 32)

    param_file.write_bytes(b"second")
    second = build_eval_signature(args, 32)

    assert first != second


def test_bf16_signature_ignores_quantization_config(tmp_path):
    first = _make_args(tmp_path, eval_mode="bf16", quant_target=[])
    second = _make_args(
        tmp_path,
        eval_mode="bf16",
        quant_target=[],
        bit_policy=BitPolicy({"w_bits": 16, "a_bits": 16}),
        algos=[],
    )

    first_signature = build_eval_signature(first, 32)
    second_signature = build_eval_signature(second, 32)
    assert first_signature["effective_bit_config"] == {"mode": "bf16"}
    assert second_signature["effective_bit_config"] == {"mode": "bf16"}


def test_checkpoint_replaces_previous_block_and_loads_latest(tmp_path):
    manager = EvalCheckpointManager(_make_args(tmp_path), num_layers=4)
    manager.save(0, [torch.tensor([1.0])])
    manager.save(1, [torch.tensor([2.0])])

    completed_idx, inter_io = manager.load()

    assert completed_idx == 1
    assert torch.equal(inter_io[0], torch.tensor([2.0]))
    assert sorted(path.name for path in manager.run_dir.iterdir()) == ["checkpoint.pt"]


def test_checkpoint_rejects_signature_mismatch(tmp_path):
    manager = EvalCheckpointManager(_make_args(tmp_path), num_layers=4)
    manager.save(0, [torch.tensor([1.0])])
    payload = torch.load(manager.checkpoint_path, weights_only=True)
    payload["signature"]["seq_len"] = 1024
    torch.save(payload, manager.checkpoint_path)

    with pytest.raises(ValueError, match="signature"):
        manager.load()


def test_corrupt_checkpoint_is_removed_and_restarts_from_zero(tmp_path):
    manager = EvalCheckpointManager(_make_args(tmp_path), num_layers=4)
    manager.run_dir.mkdir(parents=True, exist_ok=True)
    manager.checkpoint_path.write_bytes(b"not a torch checkpoint")

    assert manager.load() is None
    assert not manager.checkpoint_path.exists()

    manager.save(0, [torch.tensor([3.0])])
    assert manager.checkpoint_path.exists()

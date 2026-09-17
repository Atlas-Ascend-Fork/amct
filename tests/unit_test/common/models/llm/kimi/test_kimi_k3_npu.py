# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""CPU contract tests; these do not establish NPU kernel correctness."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from amct_pytorch.quantization.bit_policy import BitPolicy


def test_packed_layout_preserves_codes_and_scales():
    from amct_pytorch.quantization.dtypes.mxfp import NpuMxLinear
    from amct_pytorch.quantization.dtypes.mxfp_impl import pack_uint4, unpack_uint4

    codes = torch.randint(0, 16, (8, 96), dtype=torch.uint8)
    scale = torch.arange(24).reshape(8, 3).to(torch.uint8)
    weight, native_scale = NpuMxLinear.checkpoint_layout(pack_uint4(codes), scale, 4)
    assert torch.equal(unpack_uint4(weight), codes.t())
    assert native_scale.shape == (2, 8, 2)
    restored = native_scale.permute(1, 0, 2).reshape(8, 4)
    assert torch.equal(restored[:, :3], scale)
    assert torch.count_nonzero(restored[:, 3]) == 0


def test_kimi_rejects_missing_scale():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3

    with pytest.raises(ValueError, match="scale"):
        KimiK3._validate_mx_checkpoint(
            "expert.w1",
            torch.zeros(8, 32, dtype=torch.uint8),
            None,
            4,
            8,
            64,
        )


@pytest.mark.parametrize("bad", ["dtype", "shape", "nan"])
def test_kimi_rejects_bad_scale(bad):
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3

    scale = torch.full((8, 2), 127, dtype=torch.uint8)
    if bad == "dtype":
        scale = scale.float()
    elif bad == "shape":
        scale = scale[:, :1]
    else:
        scale[0, 0] = 255
    with pytest.raises(ValueError, match="scale"):
        KimiK3._validate_mx_checkpoint(
            "expert.w1",
            torch.zeros(8, 32, dtype=torch.uint8),
            scale,
            4,
            8,
            64,
        )


class _Block(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.block_sparse_moe = torch.nn.Module()
        expert = torch.nn.Module()
        expert.w1 = torch.nn.Linear(64, 8, bias=False)
        self.block_sparse_moe.experts = torch.nn.ModuleList([expert])
        self.block_sparse_moe.shared_experts = torch.nn.Module()
        self.block_sparse_moe.shared_experts.gate_proj = torch.nn.Linear(
            64, 8, bias=False
        )


def _pipeline(tmp_path):
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3

    pipeline = KimiK3.__new__(KimiK3)
    pipeline.args = SimpleNamespace(
        eval_mode="npu",
        quant_dtype="mxfp",
        device="cpu",
        algos=[],
        quant_target=["moe"],
        mxfp_checkpoint_format="e2m1-e8m0",
        bit_policy=BitPolicy(
            {"w_bits": 16, "a_bits": 16, "moe": {"routed": {"w_bits": 4, "a_bits": 4}}}
        ),
    )
    pipeline.quant_target = ["moe"]
    pipeline.config = SimpleNamespace()
    pipeline.cls = _Block
    pipeline.model_path = str(tmp_path)
    pipeline._weight_map = None
    pipeline._build_block_device_map = lambda _: {"": "cpu"}
    return pipeline


def test_stream_mixed_checkpoint_across_files(tmp_path):
    from amct_pytorch.quantization.dtypes.mxfp import NpuMxLinear

    pipeline = _pipeline(tmp_path)
    prefix = pipeline.get_layer_weight_prefix(0) + "block_sparse_moe."
    packed = prefix + "experts.0.w1.weight_packed"
    scale = prefix + "experts.0.w1.weight_scale"
    shared = prefix + "shared_experts.gate_proj.weight"
    shared_weight = torch.randn(8, 64, dtype=torch.bfloat16)
    save_file(
        {packed: torch.full((8, 32), 0x21, dtype=torch.uint8)},
        str(tmp_path / "a.safetensors"),
    )
    save_file(
        {scale: torch.full((8, 2), 127, dtype=torch.uint8), shared: shared_weight},
        str(tmp_path / "b.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    packed: "a.safetensors",
                    scale: "b.safetensors",
                    shared: "b.safetensors",
                }
            }
        )
    )
    block = pipeline.build_quant_block(0)
    routed = block.block_sparse_moe.experts[0].w1
    assert isinstance(routed, NpuMxLinear)
    assert routed.weight is None
    assert routed.qweight.dtype == torch.uint8
    assert routed.w_scale.shape == (1, 8, 2)
    assert torch.equal(
        block.block_sparse_moe.shared_experts.gate_proj.weight, shared_weight
    )


@pytest.mark.parametrize("bits", [4, 8])
def test_forward_quantizes_activation_only_for_loaded_weight(monkeypatch, bits):
    import sys

    from amct_pytorch.quantization.dtypes.mxfp import NpuMxLinear

    calls = []

    def dynamic(x, **kwargs):
        calls.append((x, kwargs))
        return x, torch.ones(x.shape[0], 1, 2, dtype=torch.uint8)

    def matmul(x, weight, scale, **kwargs):
        assert kwargs["group_sizes"] == [1, 1, 32]
        assert kwargs["pertoken_scale"].shape == (6, 1, 2)
        return torch.ones(6, 8, dtype=torch.bfloat16)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu_dynamic_mx_quant=dynamic,
            npu_quant_matmul=matmul,
            float4_e2m1fn_x2=404,
            float8_e8m0fnu=808,
        ),
    )
    module = NpuMxLinear(torch.nn.Linear(64, 8, bias=False), bits, bits)
    module.load_quantized(
        torch.zeros(
            8,
            32 if bits == 4 else 64,
            dtype=torch.uint8 if bits == 4 else torch.float8_e4m3fn,
        ),
        torch.full((8, 2), 127, dtype=torch.uint8),
    )
    x = torch.randn(3, 2, 64).transpose(0, 1)
    assert module(x).shape == (2, 3, 8)
    assert len(calls) == 1
    assert calls[0][0].dtype == torch.bfloat16
    assert calls[0][1]["dst_type"] == (404 if bits == 4 else torch.float8_e4m3fn)


@pytest.mark.parametrize("bits", [4, 8])
def test_float_weight_is_quantized_once(monkeypatch, bits):
    import sys

    from amct_pytorch.quantization.dtypes.mxfp import NpuMxLinear

    calls = []

    def dynamic(x, **kwargs):
        calls.append(kwargs)
        if kwargs["axis"] == 0:
            return torch.zeros(64, 4, dtype=torch.uint8), torch.zeros(
                1, 8, 2, dtype=torch.uint8
            )
        return x, torch.zeros(x.shape[0], 1, 2, dtype=torch.uint8)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu_dynamic_mx_quant=dynamic,
            npu_quant_matmul=lambda *a, **kw: torch.zeros(2, 8, dtype=torch.bfloat16),
            float4_e2m1fn_x2=404,
            float8_e8m0fnu=808,
        ),
    )
    module = NpuMxLinear(torch.nn.Linear(64, 8, bias=False), bits, bits)
    module(torch.ones(2, 64))
    module(torch.ones(2, 64))
    assert len(calls) == 3  # one weight quantization, two activation quantizations
    assert module.weight is None
    assert set(dict(module.named_buffers())) == {"qweight", "w_scale"}
    assert calls[0]["axis"] == (0 if bits == 4 else -1)


def _write_mixed(tmp_path, *, with_scale=True, bits=4):
    prefix = "language_model.model.layers.0.block_sparse_moe."
    weight_key = prefix + "experts.0.w1." + ("weight_packed" if bits == 4 else "weight")
    tensors = {
        weight_key: torch.zeros(
            8,
            32 if bits == 4 else 64,
            dtype=torch.uint8 if bits == 4 else torch.float8_e4m3fn,
        ),
        prefix + "shared_experts.gate_proj.weight": torch.ones(
            8, 64, dtype=torch.bfloat16
        ),
    }
    if with_scale:
        tensors[prefix + "experts.0.w1.weight_scale"] = torch.full(
            (8, 2), 127, dtype=torch.uint8
        )
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {key: "model.safetensors" for key in tensors},
            }
        )
    )


@pytest.mark.parametrize(
    "case,match",
    [
        ("missing_scale", "missing paired"),
        ("bf16_policy", "conflicts"),
        ("mixed_bits", "A8W4"),
        ("unknown_format", "explicitly declare"),
        ("wrong_target", "conflicts"),
    ],
)
def test_loader_rejects_conflicting_checkpoint(tmp_path, case, match):
    pipeline = _pipeline(tmp_path)
    _write_mixed(tmp_path, with_scale=case != "missing_scale")
    if case == "bf16_policy":
        pipeline.args.bit_policy = BitPolicy()
    elif case == "mixed_bits":
        pipeline.args.bit_policy = BitPolicy({"w_bits": 4, "a_bits": 8})
    elif case == "unknown_format":
        pipeline.args.mxfp_checkpoint_format = None
    elif case == "wrong_target":
        pipeline.quant_target = []
    with pytest.raises(ValueError, match=match):
        pipeline.block(0)


def test_fp8_checkpoint_preserves_dtype(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.args.bit_policy = BitPolicy(
        {"w_bits": 16, "a_bits": 16, "moe": {"routed": {"w_bits": 8, "a_bits": 8}}}
    )
    pipeline.args.mxfp_checkpoint_format = "e4m3-e8m0"
    _write_mixed(tmp_path, bits=8)
    module = pipeline.block(0).block_sparse_moe.experts[0].w1
    assert module.qweight.dtype == torch.float8_e4m3fn
    assert module.w_scale.dtype == torch.uint8


def test_float_checkpoint_needs_no_format_declaration(tmp_path):
    from amct_pytorch.quantization.dtypes.mxfp import NpuMxLinear

    pipeline = _pipeline(tmp_path)
    pipeline.args.mxfp_checkpoint_format = None
    prefix = pipeline.get_layer_weight_prefix(0)
    state = {
        prefix + key: value.to(torch.bfloat16)
        for key, value in _Block(None, 0).state_dict().items()
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {key: "model.safetensors" for key in state},
            }
        )
    )
    block = pipeline.block(0)
    assert isinstance(block.block_sparse_moe.experts[0].w1, NpuMxLinear)
    assert block.block_sparse_moe.experts[0].w1.qweight is None
    assert block.block_sparse_moe.experts[0].w1.weight.dtype == torch.bfloat16
    assert isinstance(block.block_sparse_moe.shared_experts.gate_proj, torch.nn.Linear)


def test_cli_npu_mode_and_guards(monkeypatch, tmp_path):
    import sys

    from amct_pytorch.cli.llm.args import parser_gen

    argv = [
        "eval",
        "--model_name",
        "kimi_k3",
        "--granularity",
        "block",
        "--eval_mode",
        "npu",
        "--quant_dtype",
        "mxfp",
        "--quant_target",
        "moe",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert parser_gen(command="eval").eval_mode == "npu"
    monkeypatch.setattr(sys, "argv", argv + ["--granularity", "model"])
    with pytest.raises(ValueError, match="granularity=block"):
        parser_gen(command="eval")
    monkeypatch.setattr(sys, "argv", argv + ["--model_name", "qwen3"])
    with pytest.raises(ValueError, match="model_name=kimi_k3"):
        parser_gen(command="eval")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="only by the eval"):
        parser_gen(command="ptq")


def test_extra_moe_projections_can_remain_bf16(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.args.bit_policy = BitPolicy(
        {
            "w_bits": 16,
            "a_bits": 16,
            "moe": {
                "routed": {
                    "w_bits": 4,
                    "a_bits": 4,
                    "routed_expert_up_proj": {"w_bits": 16, "a_bits": 16},
                }
            },
        }
    )
    block = _Block(None, 0)
    assert pipeline._npu_linear_bits("block_sparse_moe.experts.0.w1", block) == (4, 4)
    assert pipeline._npu_linear_bits(
        "block_sparse_moe.routed_expert_up_proj", block
    ) == (16, 16)
    assert pipeline._npu_linear_bits(
        "block_sparse_moe.shared_experts.gate_proj", block
    ) == (16, 16)


def test_packed_forward_matches_independent_reference(monkeypatch):
    """Verify byte layout, scale grouping and matmul arguments together on CPU."""
    import sys

    from amct_pytorch.quantization.dtypes.mxfp import NpuMxLinear

    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    codes = torch.randint(0, 16, (8, 96), dtype=torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = torch.randint(125, 130, (8, 3), dtype=torch.uint8)
    activation_codes = torch.randint(0, 16, (2, 96), dtype=torch.uint8)
    activation_scales = torch.tensor(
        [[126, 127, 128], [128, 127, 126]], dtype=torch.uint8
    )

    def unpack(p):
        c = torch.stack((p & 15, p >> 4), dim=-1).flatten(-2)
        return lut[c.long()]

    def dynamic(x, **kwargs):
        assert x.shape == (2, 96)
        return (
            activation_codes[:, 0::2] | (activation_codes[:, 1::2] << 4),
            torch.nn.functional.pad(activation_scales, (0, 1)).reshape(2, 2, 2),
        )

    def matmul(x, w, scale, **kwargs):
        assert kwargs["x1_dtype"] == kwargs["x2_dtype"] == 404
        x_scale = kwargs["pertoken_scale"].reshape(2, 4)[:, :3]
        w_scale = scale.permute(1, 0, 2).reshape(8, 4)[:, :3].t()
        x_value = unpack(x) * (2.0 ** (x_scale.float() - 127)).repeat_interleave(
            32, dim=1
        )
        w_value = unpack(w) * (2.0 ** (w_scale.float() - 127)).repeat_interleave(
            32, dim=0
        )
        return (x_value @ w_value + kwargs["bias"]).bfloat16()

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu_dynamic_mx_quant=dynamic,
            npu_quant_matmul=matmul,
            float4_e2m1fn_x2=404,
            float8_e8m0fnu=808,
        ),
    )
    module = NpuMxLinear(torch.nn.Linear(96, 8), 4, 4)
    module.load_quantized(packed, scales)
    reference_w = lut[codes.long()] * (2.0 ** (scales.float() - 127)).repeat_interleave(
        32, dim=1
    )
    reference_x = lut[activation_codes.long()] * (
        2.0 ** (activation_scales.float() - 127)
    ).repeat_interleave(32, dim=1)
    reference = (reference_x @ reference_w.t() + module.bias.float()).bfloat16()
    torch.testing.assert_close(
        module(torch.randn(1, 2, 96)), reference.unsqueeze(0), rtol=0, atol=0
    )

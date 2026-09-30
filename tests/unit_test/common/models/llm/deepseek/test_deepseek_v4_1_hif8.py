# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HiFloat8 codes, and the DeepSeek V4.1 shard conversion that writes them."""

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from amct_pytorch.common.models.llm.common.deploy_selection import DeploySelection
from amct_pytorch.common.models.llm.deepseek.deepseek_v4_1.deepseekv4_1 import (
    DeepseekV41,
)
from amct_pytorch.quantization.dtypes.hifp_impl import (
    HIF8_WEIGHT_SCOPE,
    hif8_decode,
    hif8_encode,
    hif8_pack,
)

SHARD = "model.safetensors"
GROUPS, RANK, COLUMNS = 2, 32, 64
WO_A, PLAIN = "layers.0.attn.wo_a", "layers.0.attn.wq_a"
UNTOUCHED = "layers.0.attn.norm.weight"


def test_hif8_encode_and_pack_follow_the_format():
    codes = hif8_encode(torch.tensor([0.0, 1.0, -1.0, 2.0, float("nan"), float("inf")]))
    assert codes.tolist() == [0x00, 0x08, 0x88, 0x10, 0x80, 0x6E]
    assert torch.equal(hif8_decode(codes[:4]), torch.tensor([0.0, 1.0, -1.0, 2.0]))

    weight = torch.randn(8, 64) * torch.arange(1, 9).view(-1, 1)
    codes, scale = hif8_pack(weight)
    assert codes.dtype == torch.uint8 and codes.shape == weight.shape
    assert scale.dtype == torch.float32 and scale.shape == (8, 1)
    peak = hif8_decode(codes).abs().amax(dim=1)
    assert torch.allclose(peak, torch.full((8,), HIF8_WEIGHT_SCOPE), rtol=0.1)
    assert torch.allclose(hif8_decode(codes) * scale, weight, rtol=0.1, atol=1e-3)
    with pytest.raises(RuntimeError, match="2-D"):
        hif8_pack(torch.ones(4))


def _hif8_model(tmp_path):
    model = DeepseekV41.__new__(DeepseekV41)
    model.args = SimpleNamespace(
        deploy_platform="ascend950",
        quant_dtype="hifp",
        deploy_format="legacy",
        deploy_selection=DeploySelection((PLAIN, WO_A)),
    )
    model.model_path, model.output_dir = tmp_path / "model", tmp_path / "output"
    model.model_path.mkdir()
    model.output_dir.mkdir()
    model.processed_modules = set()
    model.config = {"text_config": {"o_groups": GROUPS, "o_lora_rank": RANK}}

    rows = GROUPS * RANK
    tensors = {UNTOUCHED: torch.ones(COLUMNS, dtype=torch.bfloat16)}
    for name in (PLAIN, WO_A):
        tensors[f"{name}.weight"] = torch.ones((rows, COLUMNS), dtype=torch.float8_e4m3fn)
        tensors[f"{name}.scale"] = torch.full(
            (rows // 32, COLUMNS // 32), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    path = model.model_path / SHARD
    save_file(tensors, str(path), metadata={"format": "pt"})
    model.safetensors_files = frozenset({path.resolve()})
    return model, tensors


def test_convert_hif8_shard_writes_codes_and_leaves_the_rest_alone(tmp_path):
    model, source = _hif8_model(tmp_path)
    routing = model._convert_tensorwise_shard_hif8(
        SHARD, model.model_path, dict.fromkeys(source, SHARD), {PLAIN: 8, WO_A: 8}, {}
    )

    assert set(routing) == set(source)
    assert model.processed_modules == {PLAIN, WO_A}
    with safe_open(str(model.output_dir / SHARD), framework="pt") as shard:
        assert shard.get_slice(f"{PLAIN}.weight").get_dtype() == "U8"
        assert shard.get_slice(f"{PLAIN}.scale").get_dtype() == "F32"
        scale = shard.get_tensor(f"{WO_A}.scale")
        untouched = shard.get_tensor(UNTOUCHED)
    # npu_transpose_quant_batchmatmul takes one rank-long vector per batch.
    per_group = scale.view(GROUPS, RANK)
    assert torch.equal(per_group, per_group[0].expand(GROUPS, RANK))
    assert torch.equal(untouched, source[UNTOUCHED])


def test_validate_platform_accepts_hifp_on_ascend950():
    args = SimpleNamespace(deploy_platform="ascend950", quant_dtype="hifp")
    assert DeepseekV41._validate_platform(args) == "ascend950"
    args.quant_dtype = "int"
    with pytest.raises(ValueError, match="requires --quant_dtype mxfp or hifp"):
        DeepseekV41._validate_platform(args)

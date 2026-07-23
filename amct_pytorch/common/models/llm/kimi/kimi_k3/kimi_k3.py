# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import json
import re
from pathlib import Path

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel


_KIMI_K3_UNAVAILABLE = (
    "Kimi K3 adapter is registered as model_name='kimi_k3', but the model "
    "implementation/checkpoint structure is not available yet. Add the concrete "
    "block, attention, MLP/MoE, quant block, and PtqUnit mapping before running "
    "BF16 eval, PTQ data extraction, PTQ, or deploy."
)


@MODEL_REGISTRY.register(
    name="kimi_k3",
    task="llm",
    family="kimi",
    description="Kimi K3 placeholder adapter",
)
class KimiK3(BaseModel):
    """Minimal tensor-wise deploy adapter for Kimi K3.

    Only the deploy tensor path is enabled. BF16 eval, blockwise PTQ, and
    blockwise deploy still need the concrete model implementation.
    """

    _MOE_WEIGHT_RE = re.compile(
        r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123]\.weight_packed$"
    )

    def __init__(self, args):
        if getattr(args, "granularity", None) != "tensor":
            raise NotImplementedError(_KIMI_K3_UNAVAILABLE)
        self.args = args
        self.model_path = args.model
        self.quant_target = getattr(args, "quant_target", [])
        self.quant_dtype = getattr(args, "quant_dtype", "")
        self._weight_map = None

    @staticmethod
    def block_size(weight):
        return 32

    @staticmethod
    def _is_kimi_moe_weight_packed(weight_name: str) -> bool:
        return bool(KimiK3._MOE_WEIGHT_RE.match(weight_name))

    def _load_weight_map(self):
        if self._weight_map is not None:
            return self._weight_map

        index_path = Path(self.model_path) / "model.safetensors.index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        self._weight_map = dict(index.get("weight_map", {}))
        return self._weight_map

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        raise NotImplementedError(_KIMI_K3_UNAVAILABLE)

    def get_scale_name(self, weight_name):
        if weight_name.endswith(".weight_packed"):
            return ".weight_scale", weight_name.replace(".weight_packed", ".weight_scale")
        if weight_name.endswith(".weight"):
            return ".weight_scale", weight_name.replace(".weight", ".weight_scale")
        return ".weight_scale", f"{weight_name}.weight_scale"

    def tensorwise_output_weight_name(self, weight_name):
        if weight_name.endswith(".weight_packed"):
            return weight_name.replace(".weight_packed", ".weight")
        return weight_name

    def generate_tensorwise_quant_layers(self):
        if self.quant_target and "moe" not in self.quant_target:
            return {}
        quant_layers = {}
        for weight_name in self._load_weight_map():
            if not self._is_kimi_moe_weight_packed(weight_name):
                continue
            output_name = self.tensorwise_output_weight_name(weight_name)
            quant_layers[output_name.rsplit(".", 1)[0]] = 4
        return quant_layers

    def generate_tensorwise_ignore_layers(self):
        return []

    def bits_scheme(self):
        return [
            {"targets": ["Linear"], "w_bits": 8, "a_bits": 8},
            {"targets": ["MoEGMM"], "w_bits": 4, "a_bits": 8},
        ]

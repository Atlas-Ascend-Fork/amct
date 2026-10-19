#!/usr/bin/env python3
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
"""Sample-local wrapper: park routed experts on CPU during DeepSeek-V4 quant eval.

Stock amct_pytorch.eval on a single 64GB NPU places the whole block (all 256
routed experts) on npu:0 via _block_sharded when device_count() <= 1, which
OOMs. Expert-wise .to(device)/.to("cpu") exists only on the PTQ path
(workflows/llm_ptq.py). This monkeypatch is the minimal eval-side equivalent
used to produce the result table; amct_pytorch/ is not modified.

Upstream: see the Issue linked from this sample's README.
"""

from __future__ import annotations

import sys

import torch


def _drop_cached_quant_weight(module: torch.nn.Module) -> None:
    for mod in module.modules():
        if getattr(mod, "cached_eval_weight", None) is not None:
            mod.cached_eval_weight = None


def _place(module: torch.nn.Module, device) -> None:
    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(module, recurse=True)
    except Exception:
        pass
    module.to(device)
    _drop_cached_quant_weight(module)


def _park_routed_experts(block: torch.nn.Module) -> None:
    ffn = getattr(block, "ffn", None)
    if ffn is None:
        return
    for expert in getattr(ffn, "experts", []):
        if expert is None:
            continue
        _place(expert, "cpu")
    if torch.npu.is_available():
        torch.npu.empty_cache()


def install_expert_offload() -> None:
    """Keep routed experts on CPU while Attention runs; move one expert for MoE."""
    from amct_pytorch.common.models.llm.deepseek.deepseek_v4.deepseekv4 import (
        DeepseekV4,
    )
    from amct_pytorch.common.models.llm.deepseek.deepseek_v4.modeling import (
        modeling_deepseek_v4 as v4,
    )

    if not getattr(v4.MoE.forward, "_amct_sample_expert_offload", False):

        def forward(self, x, input_ids):
            shape = x.size()
            x = x.view(-1, self.dim)
            dev = x.device
            for expert in self.experts:
                if expert is None:
                    continue
                _place(expert, "cpu")
            _place(self.gate, dev)
            _place(self.shared_experts, dev)
            if torch.npu.is_available():
                torch.npu.empty_cache()
            weights, indices = self.gate(x, input_ids.flatten())
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(
                indices.flatten(), minlength=self.n_routed_experts
            ).tolist()
            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i] == 0:
                    continue
                expert = self.experts[i]
                _place(expert, dev)
                idx, top = torch.where(indices == i)
                y[idx] += expert(x[idx], weights[idx, top, None])
                _place(expert, "cpu")
            if v4.world_size > 1:
                v4.dist.all_reduce(y)
            y += self.shared_experts(x)
            if torch.npu.is_available():
                torch.npu.empty_cache()
            return y.type_as(x).view(shape)

        forward._amct_sample_expert_offload = True
        v4.MoE.forward = forward

    if getattr(DeepseekV4._dispatch_block, "_amct_sample_park_experts", False):
        return
    orig_dispatch = DeepseekV4._dispatch_block

    def _dispatch_block(self, module):
        module = orig_dispatch(self, module)
        _park_routed_experts(module)
        return module

    _dispatch_block._amct_sample_park_experts = True
    DeepseekV4._dispatch_block = _dispatch_block


def main() -> None:
    install_expert_offload()
    # Same CLI flags as `python3 -m amct_pytorch.eval`.
    sys.argv = ["amct_pytorch.eval", *sys.argv[1:]]
    from amct_pytorch.eval import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()

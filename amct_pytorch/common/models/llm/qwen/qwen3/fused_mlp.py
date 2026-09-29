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
"""Qwen3 MLP sharing one dynamic quantization for up/gate projections."""

from amct_pytorch.classic.deploy_op.local_dynamic import make_local_adapter


class Qwen3FusedMLP:
    @staticmethod
    def from_mlp(mlp, act_type='int8'):
        return make_local_adapter(mlp, _forward, act_type)


def _forward(self, hidden_states):
    quantized, scale = self.shared_quantize(hidden_states)
    gate = self.act_fn(self.gate_proj(quantized, scale))
    up = self.up_proj(quantized, scale)
    return self.down_proj(gate * up)

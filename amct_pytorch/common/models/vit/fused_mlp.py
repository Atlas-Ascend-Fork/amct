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
"""Local ViT MLP/intermediate adapter; residual and output stay in their native layer."""

from amct_pytorch.classic.deploy_op.local_dynamic import make_local_adapter


class ViTFusedMLP:
    @staticmethod
    def from_mlp(mlp, act_type='int8'):
        forward = _forward if hasattr(mlp, 'fc1') else _legacy_forward
        return make_local_adapter(mlp, forward, act_type)


def _forward(self, hidden_states):
    quantized, scale = self.shared_quantize(hidden_states)
    return self.fc2(self.activation_fn(self.fc1(quantized, scale)))


def _legacy_forward(self, hidden_states):
    quantized, scale = self.shared_quantize(hidden_states)
    return self.intermediate_act_fn(self.dense(quantized, scale))

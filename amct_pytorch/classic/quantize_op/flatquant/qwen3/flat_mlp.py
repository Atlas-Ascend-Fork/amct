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
#
# The code snippet comes from FlatQuant (https://github.com/ruikangliu/FlatQuant).
#
# Copyright (c) 2024 ruikangliu. All rights reserved.
#
# This file is part of FlatQuant, which is released under the MIT License.
# See the LICENSE file in the root directory of this source tree
# or at https://opensource.org/licenses/MIT for details.
#
# --------------------------------------------------------------------------------
"""Qwen3 MLP reconstruction for FlatQuant.

up/gate share up_gate_trans on the input side; down_proj gets down_trans.
use_down_quant=False keeps down_proj in float."""

import torch
import torch.nn as nn

from ..flat_base import FlatQuantModuleBase
from ..flat_config import flat_config_parser
from ..flat_linear import FlatQuantizedLinear
from ..trans_utils import (
    SVDDecomposeTransMatrix,
    get_decompose_dim,
    get_init_scale,
)


class FlatQuantQwen3MLP(FlatQuantModuleBase):
    """Reconstructed Qwen3MLP; forward signature stays identical."""

    def __init__(self, module: nn.Module, layer_name: str = '', layer_config=None):
        super().__init__(module, layer_name, layer_config)

        self.flat_config = flat_config_parser(layer_config)
        self.module_config = module.config

        self.hidden_size = module.config.hidden_size
        self.intermediate_size = module.config.intermediate_size
        self.act_fn = module.act_fn if hasattr(module, 'act_fn') else None

        self.up_proj = FlatQuantizedLinear(
            module.up_proj, self.flat_config, self.flat_config.lac
        )
        self.gate_proj = FlatQuantizedLinear(
            module.gate_proj, self.flat_config, self.flat_config.lac
        )
        self.down_proj = FlatQuantizedLinear(
            module.down_proj, self.flat_config, self.flat_config.lac
        )
        self.add_fq_trans()

        self._ori_mode = True
        self._eval_mode = False
        self.up_smax = (
            torch.ones_like(self.up_proj.linear.weight.abs().max(dim=0)[0]) * 1e-5
        )
        self.down_smax = (
            torch.ones_like(self.down_proj.linear.weight.abs().max(dim=0)[0]) * 1e-5
        )

    def add_fq_trans(self):
        if self.flat_config.a_bits < 16 or self.flat_config.w_bits < 16:
            up_dim_left, up_dim_right = get_decompose_dim(
                self.up_proj.linear.weight.shape[1]
            )
            self.up_gate_trans = SVDDecomposeTransMatrix(
                up_dim_left, up_dim_right, add_diag=self.flat_config.add_diag
            )
            if self.flat_config.use_down_quant:
                down_dim_left, down_dim_right = get_decompose_dim(
                    self.down_proj.linear.weight.shape[1]
                )
                self.down_trans = SVDDecomposeTransMatrix(
                    down_dim_left, down_dim_right, add_diag=self.flat_config.add_diag
                )
            else:
                self.down_trans = None
        else:
            self.up_gate_trans, self.down_trans = None, None

    # float passthrough by default
    def _ori_forward(self, x):
        # down_proj = down(act(gate(x)) * up(x)); collect act stats on the way
        if self.up_smax is not None:
            self.up_smax = torch.maximum(
                self.up_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        if self.down_smax is not None:
            self.down_smax = torch.maximum(
                self.down_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        return self.down_proj._ori_forward(x)

    # fake-quant training
    def _trans_forward(self, x):
        x_ts = self.up_gate_trans(x) if self.up_gate_trans is not None else x
        up_states = self.up_proj(x_ts, qa_trans=self.up_gate_trans)
        gate_states = self.gate_proj(x_ts, qa_trans=self.up_gate_trans)
        x_act_fn = self.act_fn(gate_states) * up_states

        if not self.flat_config.use_down_quant:
            return self.down_proj._ori_forward(x_act_fn)

        x_ts_2 = self.down_trans(x_act_fn) if self.down_trans is not None else x_act_fn
        return self.down_proj(x_ts_2, qa_trans=self.down_trans)

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        # training forward and deploy forward share the transform path: the act
        # half of the identity stays after convert
        return self._trans_forward(x)

    def set_fake_quant(self, enabled):
        self._ori_mode = not enabled
        for proj in (self.up_proj, self.gate_proj, self.down_proj):
            proj.set_fake_quant(enabled)

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, 'up_smax') and hasattr(self, 'down_smax')
        upw_smax = (
            torch.cat([self.up_proj.linear.weight, self.gate_proj.linear.weight], dim=0)
            .abs()
            .max(dim=0)[0]
        )
        downw_smax = self.down_proj.linear.weight.abs().max(dim=0)[0]
        if self.up_gate_trans is not None:
            self.up_gate_trans.diag_scale.data = get_init_scale(
                upw_smax, self.up_smax, alpha
            )
        if self.down_trans is not None:
            self.down_trans.diag_scale.data = get_init_scale(
                downw_smax, self.down_smax, alpha
            )
        # stats consumed; guard _ori_forward from touching deleted attrs
        self.up_smax = None
        self.down_smax = None

    def reparameterize(self):
        """Convert-time freeze; the down-side diag merges into up_proj's
        weight output channels (no preceding Norm to host it)."""
        if self._eval_mode:
            return
        # bake the weights while use_diag is still True so the weight side
        # carries the diag inverse (W/d @ M^-T); the Pass then folds the
        # diag into the preceding Norm so the deploy identity holds:
        #   (W/d @ M^-T) @ (M @ (x*c)) == W @ x
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
        if self.down_trans is not None:
            self.down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        if self.flat_config.use_down_quant:
            self.down_proj.reparameterize(qa_trans=self.down_trans)
        if self.up_gate_trans is not None:
            self.up_gate_trans.use_diag = False
        if self.down_trans is not None:
            self.down_trans.use_diag = False
        # merge down-side diag into up_proj weight: silu(gate(x)) * (up(x) * c)
        if self.down_trans is not None and self.down_trans.add_diag:
            up_weight = self.up_proj.linear.weight
            ori_dtype = up_weight.dtype
            up_weight = (
                up_weight.to(torch.float64)
                .T.mul(self.down_trans.diag_scale.to(torch.float64))
                .T
            )
            self.up_proj.linear.weight.data = up_weight.to(ori_dtype)
        self._eval_mode = True

    def get_external_reparam(self, model=None, name=''):
        """up_gate diag folds into post_attention_layernorm."""
        if (
            self.up_gate_trans is None
            or not self.up_gate_trans.add_diag
            or not self.up_gate_trans.use_diag
        ):
            return []
        return [('post_attention_layernorm', self.up_gate_trans)]

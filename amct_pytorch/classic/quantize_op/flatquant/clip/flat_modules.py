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
"""CLIP (vit-l-p14) attention/MLP reconstruction for FlatQuant.

CLIP encoder forward differs from the Qwen3 decoder: no q/k norm, no rope,
plain softmax attention, and the MLP is fc1/fc2 with quick_gelu. Block
declarations for the framework splitter live in blocks.py."""

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


def _clippy_gelu(x):
    """CLIP quick_gelu: x * sigmoid(1.702 * x)."""
    return x * torch.sigmoid(1.702 * x)


class FlatQuantCLIPAttention(FlatQuantModuleBase):
    """Reconstructed CLIPAttention (vision encoder flavor).

    Structure mirrors FlatQuantQwen3Attention:
    q/k/v/o -> FlatQuantizedLinear, ln_trans on the shared input side.
    CLIP heads are small (head_dim=64, 12 heads for vit-l/14); ln_trans
    acts on the concatenated hidden dim as usual."""

    def __init__(self, module: nn.Module, layer_name: str = '', layer_config=None):
        super().__init__(module, layer_name, layer_config)

        self.flat_config = flat_config_parser(layer_config)
        self.module_config = module  # CLIP modules carry no config handle

        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.embed_dim = module.embed_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = FlatQuantizedLinear(
            module.q_proj, self.flat_config, self.flat_config.lac
        )
        self.k_proj = FlatQuantizedLinear(
            module.k_proj, self.flat_config, self.flat_config.lac
        )
        self.v_proj = FlatQuantizedLinear(
            module.v_proj, self.flat_config, self.flat_config.lac
        )
        self.out_proj = FlatQuantizedLinear(module.out_proj, self.flat_config, False)
        self.add_fq_trans()

        self._ori_mode = True
        self._eval_mode = False
        self.ln_smax = (
            torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0]) * 1e-5
        )

    def add_fq_trans(self):
        self.ln_trans = None
        if self.flat_config.a_bits < 16 or self.flat_config.w_bits < 16:
            ln_dim_left, ln_dim_right = get_decompose_dim(
                self.q_proj.linear.weight.shape[1]
            )
            self.ln_trans = SVDDecomposeTransMatrix(
                ln_dim_left, ln_dim_right, add_diag=self.flat_config.add_diag
            )

    def _ori_forward(self, hidden_states):
        if hasattr(self, 'ln_smax'):
            self.ln_smax = torch.maximum(
                self.ln_smax,
                hidden_states.reshape(-1, hidden_states.shape[-1])
                .abs()
                .max(0)[0]
                .clone()
                .detach(),
            )
        q = self.q_proj._ori_forward(hidden_states)
        k = self.k_proj._ori_forward(hidden_states)
        v = self.v_proj._ori_forward(hidden_states)
        return q, k, v

    def _trans_forward(self, hidden_states):
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        q = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        k = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        v = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        return q, k, v

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        bsz, seq, _ = hidden_states.size()
        if self._ori_mode:
            q, k, v = self._ori_forward(hidden_states)
        else:
            q, k, v = self._trans_forward(hidden_states)

        hidden_shape = (bsz, seq, self.num_heads, self.head_dim)
        q = q.view(hidden_shape).transpose(1, 2)
        k = k.view(hidden_shape).transpose(1, 2)
        v = v.view(hidden_shape).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            # transformers 5.x create_causal_mask returns a bool mask
            # (True = visible). Adding it to float logits would promote
            # True to +1.0 — masked positions leak future/pad information
            # and the causal semantics are lost. Convert to the additive
            # -inf form instead (older float additive masks pass through).
            if attention_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attention_mask, float('-inf'))
            else:
                attn = attn + attention_mask
        attn = attn.softmax(dim=-1)
        attn_output = attn @ v
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq, self.embed_dim)
        out = (
            self.out_proj._ori_forward(attn_output)
            if self._ori_mode
            else self.out_proj(attn_output, qa_trans=None)
        )
        return out, None

    def set_fake_quant(self, enabled):
        self._ori_mode = not enabled
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj.set_fake_quant(enabled)

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, 'ln_smax')
        qkvw_smax = (
            torch.cat(
                [
                    self.q_proj.linear.weight,
                    self.k_proj.linear.weight,
                    self.v_proj.linear.weight,
                ],
                dim=0,
            )
            .abs()
            .max(dim=0)[0]
        )
        if self.ln_trans is not None:
            # dtype discipline: stats math in fp32 (fp16 carriers make
            # 1e-5 subnormal and .data assignment would retype the diag
            # Parameter to fp16, corrupting AdamW state), assign back in
            # the Parameter's own dtype.
            init_scale = get_init_scale(qkvw_smax.float(), self.ln_smax.float(), alpha)
            self.ln_trans.diag_scale.data = init_scale.to(
                self.ln_trans.diag_scale.dtype
            )
        del self.ln_smax

    def reparameterize(self):
        if self._eval_mode:
            return
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        self.v_proj.reparameterize(qa_trans=self.ln_trans)
        self.out_proj.reparameterize()
        if self.ln_trans is not None:
            # bake happened with use_diag True (diag inverse folded into the
            # weight); the Pass already folded the diag into the host Norm,
            # so switch it off AFTER baking to avoid double application.
            self.ln_trans.use_diag = False
        self._eval_mode = True

    def get_external_reparam(self, model=None, name=''):
        """ln diag folds into the preceding LN (vision: LayerNorm pre-
        attention, host attribute name provided by the framework)."""
        if (
            self.ln_trans is None
            or not self.ln_trans.add_diag
            or not self.ln_trans.use_diag
        ):
            return []
        return [('layer_norm1', self.ln_trans)]


class FlatQuantCLIPMLP(FlatQuantModuleBase):
    """Reconstructed CLIPMLP: fc1/fc2 with quick_gelu.

    fc1 gets an input-side transform (up_gate_trans analog); fc2 mirrors
    down_proj with use_down_quant."""

    def __init__(self, module: nn.Module, layer_name: str = '', layer_config=None):
        super().__init__(module, layer_name, layer_config)

        self.flat_config = flat_config_parser(layer_config)
        self.module_config = module

        self.activation_fn = module.activation_fn
        self.fc1 = FlatQuantizedLinear(
            module.fc1, self.flat_config, self.flat_config.lac
        )
        self.fc2 = FlatQuantizedLinear(
            module.fc2, self.flat_config, self.flat_config.lac
        )
        self.add_fq_trans()

        self._ori_mode = True
        self._eval_mode = False
        self.fc1_smax = (
            torch.ones_like(self.fc1.linear.weight.abs().max(dim=0)[0]) * 1e-5
        )
        self.fc2_smax = (
            torch.ones_like(self.fc2.linear.weight.abs().max(dim=0)[0]) * 1e-5
        )

    def add_fq_trans(self):
        self.fc1_trans, self.fc2_trans = None, None
        if self.flat_config.a_bits < 16 or self.flat_config.w_bits < 16:
            d1, d2 = get_decompose_dim(self.fc1.linear.weight.shape[1])
            self.fc1_trans = SVDDecomposeTransMatrix(
                d1, d2, add_diag=self.flat_config.add_diag
            )
            if self.flat_config.use_down_quant:
                d1, d2 = get_decompose_dim(self.fc2.linear.weight.shape[1])
                self.fc2_trans = SVDDecomposeTransMatrix(
                    d1, d2, add_diag=self.flat_config.add_diag
                )

    def _ori_forward(self, x):
        # smax lifetime: init_diag_scale deletes the stats after the first
        # train_block, but the float forward may run again later (input
        # capture for other blocks), so keep the hasattr guard.
        if hasattr(self, 'fc1_smax'):
            self.fc1_smax = torch.maximum(
                self.fc1_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        x = self.fc1._ori_forward(x)
        x = self.activation_fn(x)
        if hasattr(self, 'fc2_smax'):
            self.fc2_smax = torch.maximum(
                self.fc2_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        return self.fc2._ori_forward(x)

    def _trans_forward(self, x):
        x_ts = self.fc1_trans(x) if self.fc1_trans is not None else x
        x = self.fc1(x_ts, qa_trans=self.fc1_trans)
        x = self.activation_fn(x)
        if not self.flat_config.use_down_quant or self.fc2_trans is None:
            return self.fc2._ori_forward(x)
        x_ts2 = self.fc2_trans(x)
        return self.fc2(x_ts2, qa_trans=self.fc2_trans)

    def forward(self, x):
        return self._ori_forward(x) if self._ori_mode else self._trans_forward(x)

    def set_fake_quant(self, enabled):
        self._ori_mode = not enabled
        self.fc1.set_fake_quant(enabled)
        self.fc2.set_fake_quant(enabled)

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, 'fc1_smax') and hasattr(self, 'fc2_smax')
        fc1w_smax = self.fc1.linear.weight.abs().max(dim=0)[0]
        fc2w_smax = self.fc2.linear.weight.abs().max(dim=0)[0]
        # dtype discipline: as in FlatQuantCLIPAttention.init_diag_scale —
        # fp32 stats math, assign back in the Parameter's own dtype.
        if self.fc1_trans is not None:
            init_scale = get_init_scale(fc1w_smax.float(), self.fc1_smax.float(), alpha)
            self.fc1_trans.diag_scale.data = init_scale.to(
                self.fc1_trans.diag_scale.dtype
            )
        if self.fc2_trans is not None:
            init_scale = get_init_scale(fc2w_smax.float(), self.fc2_smax.float(), alpha)
            self.fc2_trans.diag_scale.data = init_scale.to(
                self.fc2_trans.diag_scale.dtype
            )
        del self.fc1_smax, self.fc2_smax

    def reparameterize(self):
        """fc2-side diag CANNOT merge into fc1's weight output channels:
        quick_gelu is nonlinear, gelu(z*c) != gelu(z)*c — the diag must stay
        on the activation side (folded into fc2's kron matrices below)."""
        if self._eval_mode:
            return
        # bake FIRST while use_diag is True (weight carries the diag
        # inverse); the Pass folds the diag into the preceding Norm so the
        # deploy identity holds: (W/d @ M^-T) @ (M @ (x*c)) == W @ x
        if self.fc1_trans is not None:
            self.fc1_trans.to_eval_mode()
        if self.fc2_trans is not None:
            self.fc2_trans.to_eval_mode()
        self.fc1.reparameterize(qa_trans=self.fc1_trans)
        if self.flat_config.use_down_quant and self.fc2_trans is not None:
            self.fc2.reparameterize(qa_trans=self.fc2_trans)
        if self.fc1_trans is not None:
            self.fc1_trans.use_diag = False
        if self.fc2_trans is not None:
            # quick_gelu is nonlinear, so the fc2-side diag cannot fold into
            # fc1's weight output channels (gelu(z*c) != gelu(z)*c). The
            # baked fc2 weight keeps the W/d @ M^-T form; the deploy forward
            # compensates by scaling gelu(fc1_out) by diag_scale, so /d and
            # *d cancel inside the matmul.
            self.fc2_trans.use_diag = False
        self._eval_mode = True

    def get_external_reparam(self, model=None, name=''):
        if (
            self.fc1_trans is None
            or not self.fc1_trans.add_diag
            or not self.fc1_trans.use_diag
        ):
            return []
        return [('layer_norm2', self.fc1_trans)]

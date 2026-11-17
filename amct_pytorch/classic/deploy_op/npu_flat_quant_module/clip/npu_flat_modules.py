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
"""CLIP deploy modules for FlatQuant (W4A4).

Same packing/orchestration as the Qwen3 side (npu_flat_utils), different
forward structure: encoder attention (no rope/q_norm, plain softmax
attention math, attention_mask optional) and fc1/fc2 MLP, aligned with
the FlatQuantCLIPAttention/MLP reconstruction modules."""

import torch

from ..npu_flat_base import NpuFlatQuantDeployMixin
from ..npu_flat_utils import (
    pack_int4_weights,
    pad_to_kronecker_multiple,
    w4a4_dynamic_linear,
    w4a4_flatquant_linear,
)


class NpuFlatQuantCLIPAttention(NpuFlatQuantDeployMixin):
    """W4A4 deploy module for FlatQuantCLIPAttention."""

    _quant_module_prefix = 'FlatQuantCLIPAttention'

    def _from_quant_module(self, q):
        self.num_heads = q.num_heads
        self.head_dim = q.head_dim
        self.embed_dim = q.embed_dim
        self.scaling = q.scaling

        for name in ('q', 'k', 'v'):
            proj = getattr(q, f'{name}_proj')
            proj.weight_quantizer.find_params(proj.linear.weight)
            wt, scale = proj.weight_quantizer.quantize(
                proj.linear.weight, quantonly=True
            )
            setattr(
                self, f'wt_{name}_packed', self._pack_param(pack_int4_weights(wt.data))
            )
            setattr(self, f'scale_{name}_proj', self._pack_param(scale))
            setattr(self, f'{name}_lac', proj.lac_ratio.item())
            # CLIP projections carry bias (unlike Qwen3); the quant matmul
            # returns the product only, so keep the bias and add it after.
            bias = proj.linear.bias
            setattr(
                self,
                f'{name}_bias',
                self._pack_param(bias.detach().clone()) if bias is not None else None,
            )
        self.left_trans = self._pack_param(q.ln_trans.matrix_left.T.contiguous())
        self.right_trans = self._pack_param(q.ln_trans.matrix_right.contiguous())

        # CLIP out_proj: no composite variant in the calibration side —
        # the deploy forward routes plain qa_trans=None, so the deploy side carries
        # only the packed weight + dynamic quant path.
        q.out_proj.weight_quantizer.find_params(q.out_proj.linear.weight)
        wt_o, scale_o = q.out_proj.weight_quantizer.quantize(
            q.out_proj.linear.weight, quantonly=True
        )
        self.wt_out_packed = self._pack_param(pack_int4_weights(wt_o.data))
        self.scale_out_proj = self._pack_param(scale_o)
        out_bias = q.out_proj.linear.bias
        self.out_bias = (
            self._pack_param(out_bias.detach().clone())
            if out_bias is not None
            else None
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        bsz, seq, _ = hidden_states.size()

        # CLIP runs full-sequence encoder batches; the kron operator's
        # 16-alignment applies the same way
        hidden_states, pad_after = pad_to_kronecker_multiple(hidden_states)

        query_states = w4a4_flatquant_linear(
            self.left_trans,
            self.right_trans,
            self.wt_q_packed,
            self.scale_q_proj,
            hidden_states,
            self.q_lac,
            bias=self.q_bias,
        )
        key_states = w4a4_flatquant_linear(
            self.left_trans,
            self.right_trans,
            self.wt_k_packed,
            self.scale_k_proj,
            hidden_states,
            self.k_lac,
            bias=self.k_bias,
        )
        value_states = w4a4_flatquant_linear(
            self.left_trans,
            self.right_trans,
            self.wt_v_packed,
            self.scale_v_proj,
            hidden_states,
            self.v_lac,
            bias=self.v_bias,
        )
        if pad_after:
            query_states = query_states[:, :seq]
            key_states = key_states[:, :seq]
            value_states = value_states[:, :seq]

        hidden_shape = (bsz, seq, self.num_heads, self.head_dim)
        q = query_states.view(hidden_shape).transpose(1, 2)
        k = key_states.view(hidden_shape).transpose(1, 2)
        v = value_states.view(hidden_shape).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            # transformers 5.x create_causal_mask returns a bool mask; adding
            # it to float logits promotes True to +1.0 and loses the causal
            # semantics. Convert to the additive -inf form instead (older
            # float additive masks pass through).
            if attention_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attention_mask, float('-inf'))
            else:
                attn = attn + attention_mask
        attn = attn.softmax(dim=-1)
        attn_output = attn @ v
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq, self.embed_dim)
        out = w4a4_dynamic_linear(
            attn_output, self.wt_out_packed, self.scale_out_proj, bias=self.out_bias
        )
        return out, None


class NpuFlatQuantCLIPMLP(NpuFlatQuantDeployMixin):
    """W4A4 deploy module for FlatQuantCLIPMLP."""

    _quant_module_prefix = 'FlatQuantCLIPMLP'

    def _from_quant_module(self, q):
        self.activation_fn = q.activation_fn

        q.fc1.weight_quantizer.find_params(q.fc1.linear.weight)
        wt, scale = q.fc1.weight_quantizer.quantize(q.fc1.linear.weight, quantonly=True)
        self.wt_fc1_packed = self._pack_param(pack_int4_weights(wt.data))
        self.scale_fc1 = self._pack_param(scale)
        self.fc1_lac = q.fc1.lac_ratio.item()
        self.fc1_left_trans = self._pack_param(q.fc1_trans.matrix_left.T.contiguous())
        self.fc1_right_trans = self._pack_param(q.fc1_trans.matrix_right.contiguous())
        fc1_bias = q.fc1.linear.bias
        self.fc1_bias = (
            self._pack_param(fc1_bias.detach().clone())
            if fc1_bias is not None
            else None
        )

        if q.flat_config.use_down_quant and q.fc2_trans is not None:
            q.fc2.weight_quantizer.find_params(q.fc2.linear.weight)
            wt, scale = q.fc2.weight_quantizer.quantize(
                q.fc2.linear.weight, quantonly=True
            )
            self.wt_fc2_packed = self._pack_param(pack_int4_weights(wt.data))
            self.scale_fc2 = self._pack_param(scale)
            self.fc2_lac = q.fc2.lac_ratio.item()
            self.fc2_left_trans = self._pack_param(
                q.fc2_trans.matrix_left.T.contiguous()
            )
            self.fc2_right_trans = self._pack_param(
                q.fc2_trans.matrix_right.contiguous()
            )
            fc2_bias = q.fc2.linear.bias
            self.fc2_bias = (
                self._pack_param(fc2_bias.detach().clone())
                if fc2_bias is not None
                else None
            )
            # quick_gelu is nonlinear so the fc2-side diag cannot fold into
            # fc1's weight output channels (gelu(z*c) != gelu(z)*c); it stays
            # on the activation: scale gelu(fc1_out) by diag_scale before
            # the kron operator.
            if q.fc2_trans.add_diag:
                self.fc2_diag = self._pack_param(
                    q.fc2_trans.diag_scale.detach().clone()
                )
            else:
                self.fc2_diag = None
        else:
            self.fc2 = q.fc2.linear

    def forward(self, x):
        bsz, seq, _ = x.size()

        x, pad_after = pad_to_kronecker_multiple(x)

        x = w4a4_flatquant_linear(
            self.fc1_left_trans,
            self.fc1_right_trans,
            self.wt_fc1_packed,
            self.scale_fc1,
            x,
            self.fc1_lac,
            bias=self.fc1_bias,
        )
        x = self.activation_fn(x)
        if hasattr(self, 'wt_fc2_packed'):
            if self.fc2_diag is not None:
                # keep the fc2-side diag on the activation (quick_gelu is
                # nonlinear — it cannot fold into fc1's weight); the fc2
                # weight bakes to W @ M^-T so this restores W @ gelu(z).
                x = x * self.fc2_diag.to(x.dtype)
            x = w4a4_flatquant_linear(
                self.fc2_left_trans,
                self.fc2_right_trans,
                self.wt_fc2_packed,
                self.scale_fc2,
                x,
                self.fc2_lac,
                bias=self.fc2_bias,
            )
        else:
            x = self.fc2(x)

        if pad_after:
            x = x[:, :seq]
        return x

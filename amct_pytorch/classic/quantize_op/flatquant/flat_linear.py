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
"""FlatQuant fake-quant linear module.

Forward: y = Q(x*M) @ Q(W*M^-T)^T. The invertible transform M flattens
the activation distribution and its inverse-transpose compensates on the
weight side, so the matmul result stays mathematically equivalent.

Mode flags (driven by the parent reconstruction module):
- _ori_mode=True (default): float passthrough for calibration capture
- _ori_mode=False: fake-quant training forward
- _eval_mode=True: transforms baked into weights, activation-only quant"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import ActivationQuantizer, WeightQuantizer
from .trans_utils import kronecker_matmul


class FlatQuantizedLinear(nn.Module):
    """Wraps the original nn.Linear (weight by reference, no copy)."""

    def __init__(self, linear: nn.Linear, flat_config, lac):
        super().__init__()
        self.linear = linear

        quantizer_size = linear.weight.shape[0]
        # weight granularity: per-channel (default) or per-tensor, resolved
        # from weights_cfg.strategy; the deploy-side pack consumes the same
        # scale shape, so 'tensor' yields a broadcast single-scale artifact.
        self._w_perchannel = (
            getattr(flat_config, 'w_granularity', 'channel') != 'tensor'
        )
        scale_size = quantizer_size if self._w_perchannel else 1
        self.weight_quantizer = WeightQuantizer(shape=(scale_size, 1))
        self.weight_quantizer.configure(
            flat_config.w_bits,
            perchannel=self._w_perchannel,
            sym=flat_config.w_sym,
        )
        self.act_quantizer = ActivationQuantizer(
            flat_config.a_bits,
            sym=flat_config.a_sym,
            lac=lac,
            act_dtype=getattr(flat_config, 'act_dtype', 'fp32'),
        )

        self.lwc = flat_config.lwc
        if self.lwc:
            lwc_dim = self.linear.weight.shape[0]
            dev = self.linear.weight.device
            self.clip_factor_w_max = nn.Parameter(
                torch.ones((lwc_dim, 1)).to(dev) * 4.0, requires_grad=True
            )
            self.clip_factor_w_min = nn.Parameter(
                torch.ones((lwc_dim, 1)).to(dev) * 4.0, requires_grad=True
            )
            self.sigmoid = nn.Sigmoid()

        self._eval_mode = False
        # forward-mode flag driven by the parent reconstruction module:
        # True -> float passthrough (default), False -> fake-quant training
        self._fake_quant_enabled = False

    def apply_wclip(self, weight):
        """Learnable weight clip: sigmoid-parameterized per-channel bounds."""
        wmin = weight.min(1, keepdim=True)[0]
        wmax = weight.max(1, keepdim=True)[0]
        wmax = wmax * self.sigmoid(self.clip_factor_w_max)
        wmin = wmin * self.sigmoid(self.clip_factor_w_min)
        return torch.clamp(weight, min=wmin, max=wmax)

    def apply_trans(self, weight, qa_trans):
        """Weight side of the identity: W @ M^-T.

        qa_trans is either a transform module (single matrix) or a
        [left, right] list of already-inverted matrices (the o_proj
        v->o composite."""
        if isinstance(qa_trans, list):
            return kronecker_matmul(
                weight, qa_trans[0].to(weight), qa_trans[1].to(weight)
            )
        return qa_trans(weight, inv_t=True)

    def get_quantized_weight(self, qa_trans=None, out_trans=None, quantonly=False):
        """Fake-quant the (transformed) weight."""
        weight = self.linear.weight.data
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T

        self.weight_quantizer.find_params(weight)
        if quantonly:
            weight, scale = self.weight_quantizer.quantize(weight, quantonly=True)
            return weight, scale
        return self.weight_quantizer.quantize(weight)

    def _ori_forward(self, hidden_states):
        return self.linear(hidden_states)

    def _train_forward(self, hidden_states, qa_trans=None, out_trans=None):
        weight = self.get_quantized_weight(qa_trans=qa_trans, out_trans=out_trans)
        hidden_states = self.act_quantizer(hidden_states)
        if out_trans is not None and self.linear.bias is not None:
            bias = out_trans(self.linear.bias.data)
        else:
            bias = self.linear.bias
        return F.linear(hidden_states, weight, bias)

    def _eval_forward(self, hidden_states):
        """Post-convert: W already carries M^-T, only the act side quantizes."""
        x_dtype = hidden_states.dtype
        hidden_states = self.act_quantizer(hidden_states).to(x_dtype)
        return self.linear(hidden_states)

    def forward(self, hidden_states, qa_trans=None, out_trans=None):
        if self._eval_mode:
            return self._eval_forward(hidden_states)
        if self._fake_quant_enabled:
            return self._train_forward(
                hidden_states, qa_trans=qa_trans, out_trans=out_trans
            )
        return self._ori_forward(hidden_states)

    def set_fake_quant(self, enabled):
        """Enable/disable fake-quant training (calibrator calls this)."""
        self._fake_quant_enabled = enabled

    def reparameterize(self, qa_trans=None, out_trans=None):
        """Bake the weight-side transform into the main weight (float64).

        Called at convert time; _eval_mode makes it idempotent."""
        if self._eval_mode:
            return
        weight = self.linear.weight.data
        ori_dtype = weight.dtype
        weight = weight.to(torch.float64)
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        if out_trans is not None and self.linear.bias is not None:
            self.linear.bias.data = out_trans(self.linear.bias.data)
        # bake the fake-quantized weight so the deploy forward matches the
        # training forward numerically
        self.weight_quantizer.find_params(weight.to(ori_dtype))
        weight = self.weight_quantizer.quantize(weight.to(ori_dtype))
        self.linear.weight.data = weight.to(ori_dtype)
        self.lac_ratio = self.act_quantizer.clip_factor_a.detach()
        self._eval_mode = True

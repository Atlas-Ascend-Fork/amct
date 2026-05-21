# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import torch
import torch.nn as nn
from amct_pytorch.utils.vars import INT8, INT32_MAX, INT32_MIN


class BaseQuantizeModule(nn.Module):
    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__()
        self.ori_module_type = None
        self.act_type = None
        self.wts_type = None
        self.scale_w = None
        self.scale_d = None
        self.offset_w = None
        self.offset_d = None
        self.group_size = None

        self.scale = None
        self.clip_max = None

    def _quantizable_weight(self):
        """Weight tensor to pass into quant_dequant_tensor.
        SmoothQuant pre-multiplies by self.scale (smooth_factor) so the
        fake-quant matches scale_w, which was computed from weight * smooth_factor.
        """
        if self.scale is not None:
            return self.weight.data * self.scale.to(device=self.weight.device,
                                                    dtype=self.weight.dtype)
        return self.weight.data

    def _quantizable_input(self, inputs):
        """Activation to pass into quant_dequant_tensor.
        SmoothQuant divides by self.scale (smooth_factor) so it matches scale_d.
        """
        if self.scale is not None:
            return inputs / self.scale.to(device=inputs.device, dtype=inputs.dtype)
        return inputs

    def _fake_quant_bias(self, deq_scale):
        """Simulate NPU integer bias addition by quant→dequant.
        Only INT8 + per-tensor activation path quantizes bias to INT32 (matching
        the NPU integer accumulator); all other paths return the raw fp bias.
        """
        if self.bias is None:
            return None
        if (deq_scale is None
                or self.act_type != INT8
                or getattr(self, 'act_granularity', None) != 'tensor'):
            return self.bias
        bias_q = (self.bias.data / deq_scale).round().clamp(INT32_MIN, INT32_MAX)
        return bias_q * deq_scale

    def _build_fake_quant_cache(self):
        """Compute weight dequant tensor, deq_scale and quantized bias once.
        These derive from calibration outputs (scale_w/scale_w1/scale_w2/scale_d/
        offset/bias) and do not change across batches once calibration finishes,
        so they are cached on the first _fake_quant_forward call.
        """
        from amct_pytorch.utils.quant_util import quant_dequant_tensor

        if self.scale_w is None and getattr(self, 'scale_w1', None) is not None \
                and getattr(self, 'scale_w2', None) is not None:
            from amct_pytorch.quantize_op.utils import apply_progressive_quant_dequant
            self._cached_dequant_w = apply_progressive_quant_dequant(
                self._quantizable_weight(), self.scale_w1, self.scale_w2)
            self._cached_deq_scale = None
        else:
            self._cached_dequant_w = quant_dequant_tensor(
                self._quantizable_weight(), self.wts_type,
                self.scale_w, self.offset_w, self.group_size)
            if self.scale_d is not None and self.scale_w is not None:
                ref_dtype = self.bias.dtype if self.bias is not None \
                    else self._cached_dequant_w.dtype
                self._cached_deq_scale = (self.scale_d.to(ref_dtype)
                                          * self.scale_w.to(ref_dtype)).reshape(-1)
            else:
                self._cached_deq_scale = None

        self._cached_bias_for_linear = self._fake_quant_bias(self._cached_deq_scale)

    @torch.no_grad()
    def _fake_quant_forward(self, inputs):
        """Fake-quantize inputs and run linear/conv with the cached dequantized
        weight and bias. Activation dequant is recomputed per batch; everything
        else comes from the cache built on the first call.
        """
        import torch.nn.functional as F
        from amct_pytorch.utils.quant_util import quant_dequant_tensor

        if not getattr(self, '_fake_quant_cache_ready', False):
            self._build_fake_quant_cache()
            self._fake_quant_cache_ready = True

        if self.scale_d is not None:
            dequant_x = quant_dequant_tensor(
                self._quantizable_input(inputs), self.act_type,
                self.scale_d, self.offset_d)
        else:
            dequant_x = inputs

        if isinstance(getattr(self, 'ori_module', None), nn.Conv2d):
            conv = self.ori_module
            return F.conv2d(dequant_x, self._cached_dequant_w,
                            self._cached_bias_for_linear,
                            conv.stride, conv.padding, conv.dilation, conv.groups)
        return F.linear(dequant_x, self._cached_dequant_w,
                        self._cached_bias_for_linear)

    def forward(self, inputs):
        pass
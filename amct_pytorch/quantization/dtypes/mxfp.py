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

from __future__ import annotations

import torch
from torch import Tensor
from amct_pytorch.common.utils.quant_util import (
    MXFP4,
    MXFP8,
    quant_format_supported,
)
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY
from amct_pytorch.quantization.dtypes.fp_impl import (
    f32_to_f4_unpacked,
    pack_uint4,
    quantize_elewise,
    shared_exponents,
)


FP4_BITS = 4
FP8_BITS = 8
MX_SUPPORTED_BITS = (FP4_BITS, FP8_BITS)
MX_BLOCK_SIZE = 32
MX_AXIS = -1
MX_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def npu_dynamic_mx_quant_value(tensor: torch.Tensor, bits: int):
    if bits not in MX_SUPPORTED_BITS:
        return None
    if torch.is_grad_enabled() or tensor.dtype not in MX_INPUT_DTYPES:
        return None
    fmt = MXFP8 if bits == FP8_BITS else MXFP4
    if not quant_format_supported(fmt, tensor.device):
        return None
    import torch_npu

    dst_type = (
        torch_npu.float8_e4m3fn if bits == FP8_BITS else torch_npu.float4_e2m1fn_x2
    )
    q, mxscale = torch_npu.npu_dynamic_mx_quant(
        tensor, axis=MX_AXIS, dst_type=dst_type, block_size=MX_BLOCK_SIZE
    )
    return torch_npu.npu_anti_mx_quant(
        q,
        mxscale,
        axis=MX_AXIS,
        dst_type=tensor.dtype,
        src_type=dst_type,
    )


@DTYPE_REGISTRY.register(name="mxfp", description="quant dequant for mxfp")
class QuantDequantMx(torch.nn.Module):
    def __init__(self, bits=8, is_act=False, *args):
        super(QuantDequantMx, self).__init__()
        self.bits = bits
        self.is_act = is_act
        self.is_observe = False
        self._get_format_params()
        self.block_size = 32

    def deploy(self, x: Tensor, qdim: int = -1, v: Tensor | float = 0.0):
        e8m0, ex_mx = self.quant(x, qdim, v=v)
        e8m0 = (e8m0 + 127).to(torch.uint8).squeeze(-1).cpu()
        ex_mx = ex_mx.flatten(qdim - 1, qdim).cpu()
        if self.bits == 8:
            ex_mx = ex_mx.to(torch.float8_e4m3fn)
        else:
            ex_mx = f32_to_f4_unpacked(ex_mx.float().cpu())
            ex_mx = pack_uint4(ex_mx)
        return ex_mx, e8m0

    def export_deploy(self, x: Tensor, v: Tensor | float = 0.0):
        qx, scale = self.deploy(x, v=v)
        return {
            "qweight": qx.detach().cpu(),
            "weight_scale": scale.detach().cpu(),
        }

    def fake_quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        e8m0, ex_mx = self.quant(x, qdim, v=v)
        dx = (2**e8m0) * ex_mx
        dx = dx.flatten(qdim - 1, qdim)
        return dx.to(x.dtype)

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.is_observe or self.bits == 16:
            return x
        if torch.count_nonzero(torch.as_tensor(v)).item() == 0:
            out = npu_dynamic_mx_quant_value(x, self.bits)
            if out is not None:
                return out
        return self.fake_quant(x, v=v)

    def quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        if x.dtype == torch.float16:
            x = x.float()
        x = x.unflatten(qdim, (-1, self.block_size))
        if isinstance(v, torch.Tensor):
            v = v.to(device=x.device, dtype=x.dtype).unflatten(
                qdim, (-1, self.block_size)
            )
        e8m0 = shared_exponents(x, self.emax)
        x = x / (2**e8m0)
        ex_mx = quantize_elewise(x, self.min_exp, self.max_norm, self.shift_val, v=v)
        return e8m0, ex_mx

    def _get_format_params(self):
        if self.bits == 8:
            (
                self.ebits,
                self.mbits,
                self.emax,
                self.max_norm,
                self.shift_val,
                self.min_exp,
            ) = 4, 5, 8, 448.0, 8, -6
        else:
            (
                self.ebits,
                self.mbits,
                self.emax,
                self.max_norm,
                self.shift_val,
                self.min_exp,
            ) = 2, 3, 2, 6.0, 2, 0

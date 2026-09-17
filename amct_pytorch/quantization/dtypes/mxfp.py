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

from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY
from amct_pytorch.quantization.dtypes.mxfp_impl import (
    f32_to_f4_unpacked,
    pack_uint4,
    quantize_elewise,
    shared_exponents,
    unpack_uint4,
)


@DTYPE_REGISTRY.register(name="mxfp", description="quant dequant for mxfp")
class QuantDequantMx(torch.nn.Module):
    def __init__(self, bits=8, is_act=False):
        super(QuantDequantMx, self).__init__()
        self.bits = bits
        self.is_act = is_act
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
        dx = (2 ** e8m0) * ex_mx
        dx = dx.flatten(qdim - 1, qdim)
        return dx

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.bits == 16:
            return x
        return self.fake_quant(x, v=v)

    def quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        x = x.unflatten(qdim, (-1, self.block_size))
        if isinstance(v, torch.Tensor):
            v = v.to(device=x.device, dtype=x.dtype).unflatten(qdim, (-1, self.block_size))
        e8m0 = shared_exponents(x, self.emax)
        x = x / (2 ** e8m0)
        ex_mx = quantize_elewise(x, self.min_exp, self.max_norm, self.shift_val, v=v)
        return e8m0, ex_mx

    def _get_format_params(self):
        if self.bits == 8:
            self.ebits, self.mbits, self.emax, self.max_norm, self.shift_val, self.min_exp = 4, 5, 8, 448.0, 8, -6
        else:
            self.ebits, self.mbits, self.emax, self.max_norm, self.shift_val, self.min_exp = 2, 3, 2, 6.0, 2, 0


class NpuMxLinear(torch.nn.Module):
    """Inference-only MXFP Linear; fake-quantization interfaces remain unchanged.

    Checkpoint input uses [N, K] (FP4 packed along K) and [N, K/32]
    E8M0 bytes. Runtime uses [K, N] (FP4 packed along N) and
    [ceil(K/64), N, 2] interleaved scale bytes. No dequantization or
    rounding is performed when converting checkpoint layouts.
    """

    def __init__(self, linear, w_bits, a_bits):
        super().__init__()
        if (a_bits, w_bits) not in ((4, 4), (8, 8)):
            raise ValueError(f"NPU MXFP requires A4W4 or A8W8, got A{a_bits}W{w_bits}")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.w_bits, self.a_bits = w_bits, a_bits
        if self.in_features % 32 or (w_bits == 4 and self.out_features % 2):
            raise ValueError("NPU MXFP requires K divisible by 32 and FP4 requires even N")
        self.register_parameter("weight", linear.weight)
        self.register_parameter("bias", linear.bias)
        self.register_buffer("qweight", None)
        self.register_buffer("w_scale", None)

    @staticmethod
    def checkpoint_layout(qweight, scale, bits):
        """Convert canonical checkpoint codes to the non-transposed NPU layout."""
        if bits == 4:
            codes = unpack_uint4(qweight.contiguous())
            qweight = pack_uint4(codes.t().contiguous())
        elif bits == 8:
            qweight = qweight.t().contiguous()
        else:
            raise ValueError(f"Unsupported MXFP weight bits: {bits}")
        n, groups = scale.shape
        if groups % 2:
            scale = torch.nn.functional.pad(scale, (0, 1), value=0)
        scale = scale.reshape(n, -1, 2).permute(1, 0, 2).contiguous()
        return qweight, scale

    @torch.no_grad()
    def load_quantized(self, qweight, scale):
        self.qweight, self.w_scale = self.checkpoint_layout(qweight, scale, self.w_bits)
        # Do not retain a full BF16 copy (or a meta placeholder) of packed weights.
        self.weight = None

    @torch.no_grad()
    def prepare_weight(self):
        if self.qweight is not None:
            return
        if self.weight is None or self.weight.is_meta:
            raise RuntimeError("NPU MXFP weight has not been loaded")
        import torch_npu

        weight = self.weight.detach().to(torch.bfloat16)
        if self.w_bits == 8:
            quant, scale = torch_npu.npu_dynamic_mx_quant(
                weight, dst_type=torch.float8_e4m3fn, axis=-1, block_size=32,
            )
            self.qweight = quant.t().contiguous()
            self.w_scale = scale.transpose(0, 1).contiguous()
        else:
            self.qweight, self.w_scale = torch_npu.npu_dynamic_mx_quant(
                weight.t().contiguous(), dst_type=torch_npu.float4_e2m1fn_x2,
                axis=0, block_size=32,
            )
            self.qweight = self.qweight.contiguous()
            self.w_scale = self.w_scale.contiguous()
        self.weight = None

    @torch.no_grad()
    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"NPU MXFP input K={x.shape[-1]}, expected {self.in_features}")
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.out_features), dtype=torch.bfloat16)
        self.prepare_weight()
        if x.device != self.qweight.device or x.device != self.w_scale.device:
            raise ValueError("NPU MXFP activation, weight and scale must share a device")
        if self.a_bits == 8:
            return self.forward_a8w8(x, self.bias)
        return self.forward_a4w4(x, self.bias)

    def forward_a8w8(self, x, bias=None):
        import torch_npu

        return self._quant_matmul(x, bias, torch.float8_e4m3fn, torch_npu)

    def forward_a4w4(self, x, bias=None):
        import torch_npu

        return self._quant_matmul(x, bias, torch_npu.float4_e2m1fn_x2, torch_npu)

    def _quant_matmul(self, x, bias, dtype, torch_npu):
        shape = x.shape[:-1]
        x = x.reshape(-1, self.in_features).to(torch.bfloat16)
        quant, scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=dtype, axis=-1, block_size=32)
        kwargs = {}
        if self.a_bits == 4:
            kwargs.update(x1_dtype=dtype, x2_dtype=dtype)
        output = torch_npu.npu_quant_matmul(
            quant, self.qweight, self.w_scale,
            pertoken_scale=scale,
            bias=None if bias is None else bias.float(),
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            output_dtype=torch.bfloat16, group_sizes=[1, 1, 32], **kwargs,
        )
        return output.reshape(*shape, self.out_features)

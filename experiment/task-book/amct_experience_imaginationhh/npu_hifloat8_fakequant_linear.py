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
"""基于 amct_ops NPU 自定义算子的 HiFloat8 伪量化线性层。

与 amct_pytorch/experimental/hifloat8/hifloat8_fakequant_linear.py（CPU 实现）
的区别：本模块用 amct_ops 的 NPU ascendc 算子 encode_to_hifloat8 /
decode_from_hifloat8 在 NPU 上完成 HiFloat8 编解码，避免逐层逐 batch 的
CPU<->NPU 数据搬运，推理显著更快。

背景：AMCT 内置 HIFP8_CAST_CFG 走 torch_npu.npu_quantize(dtype=hifloat8)，
在当前 CANN 9.1.0 上因 aclnnQuantize 内核未编入 HiFloat8 支持而报错
(DT_HIFLOAT8 not in [INT8,UINT8,INT32])。amct_ops 的 cast 算子是独立的
ascendc kernel（LUT 实现），不依赖该内核，因此可在本环境跑通真·NPU 伪量化。

量化策略（对齐 HIFP8_CAST_CFG 与 CPU 版伪量化实现）：
- 权重：per-tensor 对称量化，scale = max(|W|) / 16（HiFloat8 高精度范围 16）
- 激活：直转（cast），不缩放
"""

import torch
import torch.nn.functional as F

from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule

# HiFloat8 高精度表示范围（E∈[-3,3] 区间，绝对值上界约 16）
HIF8_HIGH_PRECISION_RANGE = 16.0


def _encode_decode(x, encode_fn, decode_fn):
    """用 NPU 算子做一次 HiFloat8 编解码（伪量化往返）。

    amct_ops 算子要求输入为 fp16/bf16 且在 NPU 上。fp32 先降到 fp16 再还原，
    保持与原 dtype 一致。
    """
    ori_dtype = x.dtype
    work_dtype = (
        ori_dtype
        if ori_dtype in (torch.float16, torch.bfloat16)
        else torch.float16
    )
    codes = encode_fn(x.to(work_dtype))
    deq = decode_fn(codes, work_dtype)
    return deq.to(ori_dtype)


class NpuHifloat8FakequantLinear(BaseQuantizeModule):
    """HiFloat8 NPU 伪量化线性层。

    input cast to hifloat8 (NPU op)
    weight quantize to hifloat8, per-tensor (NPU op)
    """

    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__(ori_module, layer_name, quant_config)
        from amct_ops.hifloat8_cast import (
            encode_to_hifloat8,
            decode_from_hifloat8,
        )

        self._encode = encode_to_hifloat8
        self._decode = decode_from_hifloat8
        self.layer_name = layer_name

        weight = ori_module.weight
        device = weight.device
        # per-tensor 权重 scale：缩放到 HiFloat8 高精度范围内再伪量化
        scale_w = weight.abs().max() / HIF8_HIGH_PRECISION_RANGE
        scale_w = torch.clamp(scale_w, min=torch.finfo(torch.float16).tiny)
        scaled = weight / scale_w
        fq_w = (
            _encode_decode(scaled.to(device), self._encode, self._decode)
            * scale_w
        )
        # 用 fakequant 后的权重替换（保持原 dtype/device）
        self.register_buffer("fakequant_weight", fq_w.to(weight.dtype))
        self.scale_w = scale_w
        self.bias = ori_module.bias

    @torch.no_grad()
    def forward(self, x):
        # 激活直转：cast 到 HiFloat8 再还原
        fq_x = _encode_decode(x, self._encode, self._decode)
        return F.linear(fq_x, self.fakequant_weight, self.bias)

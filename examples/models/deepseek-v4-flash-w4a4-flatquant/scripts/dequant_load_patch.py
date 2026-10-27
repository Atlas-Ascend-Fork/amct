# coding=utf-8
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
"""Load-time FP8/MXFP4 dequant for the official DeepSeek-V4-Flash checkpoint.

The official checkpoint stores attention/shared-expert linears as FP8 e4m3
with 128x128 e8m0 scale blocks, and routed experts as int8-packed MXFP4 with
per-32-element e8m0 scales. A full bf16 re-export is ~600 GB and does not fit
next to the 149 GB source on a 300 GB host, so instead of materializing a
converted copy this patch dequants each tensor on CPU as the per-block loader
reads it (~13 GB transient per layer).

The dequant math mirrors amct_pytorch/experimental/deepseek-v4/convert_model.py
(weight_dequant + unpack_mxfloat4_to_fp32). Only the two loader modules that
call `safe_open` are touched via a proxy; amct_pytorch/ is not modified.
"""

import warnings

import torch

_E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)


def _unpack_mxfp4(packed_u8: torch.Tensor) -> torch.Tensor:
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    out = _E2M1[torch.stack([low, high], dim=-1).long()]
    shape = list(packed_u8.shape)
    shape[-1] *= 2
    return out.view(shape)


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype == torch.int8:
        # MXFP4 打包约定：每字节 2 个 4bit、scale 每 32 元素一组——
        # 形状不满足时保持原样返回并告警（对当前 checkpoint 行为不变）。
        if (
            scale.dim() == 0
            or weight.dim() == 0
            or scale.shape[-1] * 32 != weight.shape[-1] * 2
        ):
            warnings.warn(
                "dequant_load_patch: scale/weight 形状不满足 MXFP4 打包约定，跳过反量化保持原样"
            )
            return weight
        w = _unpack_mxfp4(weight.view(torch.uint8)).to(torch.float32)
        s = scale.to(torch.float32).repeat_interleave(32, dim=1)
    else:
        # FP8 e4m3：128x128 分块 scale。形状不符合分块约定（如 1-D 或
        # per-row [M,1]）时保持原样返回并告警——避免静默截断乘出错误权重
        # （当前官方 checkpoint 为 128x128 分块，正常不触发）。
        m_blk = (weight.shape[0] + 127) // 128
        n_blk = (weight.shape[1] + 127) // 128
        if scale.dim() != 2 or tuple(scale.shape) != (m_blk, n_blk):
            warnings.warn(
                "dequant_load_patch: scale 形状 "
                f"{tuple(scale.shape)} 与 128x128 分块约定不符，跳过反量化保持原样"
            )
            return weight
        w = weight.to(torch.float32)
        s = scale.to(torch.float32)
        s = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w * s[: w.shape[0], : w.shape[1]]).to(torch.bfloat16)


class _DequantSafeOpen:
    """Proxy over safetensors.safe_open that dequants `.weight` tensors which
    have a `.scale` sibling, and hides the `.scale` entries from key listing."""

    def __init__(self, factory, path, **kwargs):
        self._f = factory(path, **kwargs)

    def get_tensor(self, name):
        tensor = self._f.get_tensor(name)
        if name.endswith(".scale"):
            return tensor
        # 判定依据与参考实现一致：1 字节存储（FP8 e4m3 / int8 打包 MXFP4）
        # 即视为量化权重，配对的 .scale 缺失时保留原样并告警（不静默失真）。
        if tensor.element_size() == 1:
            try:
                scale = self._f.get_tensor(name + ".scale")
            except Exception:
                warnings.warn(f"dequant_load_patch: no .scale for {name}, keep raw")
                return tensor
            return _dequant(tensor, scale)
        return tensor

    def keys(self):
        return [k for k in self._f.keys() if not k.endswith(".scale")]

    def __getattr__(self, item):
        return getattr(self._f, item)

    def __enter__(self):
        self._f.__enter__()
        return self

    def __exit__(self, *exc):
        return self._f.__exit__(*exc)


def apply_dequant_patch() -> None:
    """Route `safe_open` in the two loader modules through the dequant proxy."""
    import amct_pytorch.common.models.llm.common.base as base_mod
    import amct_pytorch.common.models.llm.deepseek.deepseek_v4.deepseekv4 as dsv4_mod

    real_safe_open = base_mod.safe_open

    def factory(path, **kwargs):
        kwargs.setdefault("framework", "pt")
        return _DequantSafeOpen(real_safe_open, path, **kwargs)

    base_mod.safe_open = factory
    dsv4_mod.safe_open = factory

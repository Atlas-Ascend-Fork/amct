"""
amct_ops — AMCT custom operators for Ascend NPU
===============================================

Available submodules
--------------------
amct_ops.hifloat8_cast
    FP16 / BF16  ↔  HiFloat8 (8-bit float) conversion.

    Functions
    ~~~~~~~~~
    encode_to_hifloat8(x)                    FP16/BF16 → uint8 (HiFloat8)
    decode_from_hifloat8(x, dtype=bfloat16)  uint8 → FP16/BF16

    Also accessible as:
        torch.ops.amct.encode_to_hifloat8
        torch.ops.amct.decode_from_hifloat8

Quick start
-----------
>>> import torch
>>> from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8
>>>
>>> x = torch.randn(1024, dtype=torch.float16, device="npu")
>>> y = encode_to_hifloat8(x)           # → uint8, same shape
>>> z = decode_from_hifloat8(y)         # → bfloat16, same shape
"""

__all__ = ['hifloat8_cast']

from . import hifloat8_cast

"""
npu_ops — AMCT custom operators for Ascend NPU
===============================================

Available submodules
--------------------
npu_ops.hifloat8_cast
    FP16 / BF16  ↔  HiFloat8 (8-bit float) conversion.

    Functions
    ~~~~~~~~~
    encode_to_hifloat8(x)                    FP16/BF16 → uint8 (HiFloat8)
    decode_from_hifloat8(x, dtype=bfloat16)  uint8 → FP16/BF16

    Also accessible as:
        torch.ops.npu.encode_to_hifloat8
        torch.ops.npu.decode_from_hifloat8

Quick start
-----------
>>> import torch
>>> from npu_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8
>>>
>>> x = torch.randn(1024, dtype=torch.float16, device="npu")
>>> y = encode_to_hifloat8(x)           # → uint8, same shape
>>> z = decode_from_hifloat8(y)         # → bfloat16, same shape
"""

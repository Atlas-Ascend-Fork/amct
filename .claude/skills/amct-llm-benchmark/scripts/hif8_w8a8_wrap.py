"""HiFloat8 W8A8 fakequant Linear, NPU-native cast via amct_ops.

This is the runtime equivalent of amct_pytorch/experimental/hifloat8/
hifloat8_fakequant_linear.py (Cast algorithm), but executes the HiFloat8
encode/decode on NPU instead of CPU (~3.5x faster) AND uses per-channel
weight scale (matches AMCT main pipeline HIFP8_CAST_CFG).

Quantization policy (Cast — simplest HiFloat8 W8A8 variant):
- Weight: per-output-channel symmetric, scale_w[i] = |w[i,:]|.amax() / 16,
  normalize so each row's max -> 16 (HiFloat8 high-precision boundary).
  Cast to HiFloat8 once at init.
- Activation: direct cast every forward, no scale (relies on HiFloat8
  dynamic range to cover typical activation magnitudes).
- Skips lm_head and embeddings; only wraps Linear inside model.layers.*.

Per-channel was chosen over per-tensor because LLM weight magnitudes vary
significantly across output channels (10-100x not uncommon); per-tensor's
single global scale squashes the small-magnitude channels into HiFloat8
low-precision / denormal regions. Per-channel gives each channel its own
optimal scale. On Qwen3-8B: per-channel PPL=9.7398 (matches AMCT main
pipeline HIFP8_CAST_CFG exactly); per-tensor degrades to 9.7523.

For TAQ/ORQ/Quantile variants see amct_pytorch/experimental/hifloat8/.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8


def _hif8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Encode to HiFloat8 then decode back to the input's dtype."""
    return decode_from_hifloat8(encode_to_hifloat8(x), x.dtype)


class Hifloat8FakequantLinearNPU(nn.Module):
    HIF8_RANGE = 16  # HiFloat8 high-precision upper bound (|E|<=3 region)

    def __init__(self, ori: nn.Linear):
        super().__init__()
        w = ori.weight.detach()  # shape: [out_features, in_features]
        # Per-output-channel symmetric scale (matches HIFP8_CAST_CFG strategy='channel')
        scale_w = w.abs().amax(dim=1, keepdim=True) / self.HIF8_RANGE  # [out_features, 1]
        fq_w = _hif8_roundtrip(w / scale_w) * scale_w
        self.register_buffer("fakequant_weight", fq_w.to(w.dtype))
        self.register_buffer("scale_w", scale_w.to(w.dtype))
        self.bias = ori.bias

    @torch.no_grad()
    def forward(self, x):
        return F.linear(_hif8_roundtrip(x), self.fakequant_weight, self.bias)


def wrap_model_hif8_w8a8(model: nn.Module, prefix: str = "") -> int:
    """Recursively replace nn.Linear inside model.layers.* with HiFloat8 fakequant.

    Returns the number of modules swapped. Skips lm_head, embedding, and
    any Linear outside the transformer block path to keep the head FP16.
    """
    n = 0
    for name, child in list(model.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and full.startswith("model.layers."):
            setattr(model, name, Hifloat8FakequantLinearNPU(child))
            n += 1
        else:
            n += wrap_model_hif8_w8a8(child, full)
    return n

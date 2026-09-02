# FakeQuant Simulation Toolkit

This directory provides a fake quantization toolkit. It simulates the numerical behavior of quantization formats that the hardware does not yet natively support, so that accuracy can be verified and algorithms can be evaluated in software.

Typical cases include formats that have not yet landed in the NPU operator stack, or environments where the corresponding low-bit compute units cannot be enabled. The fake-quant path still reproduces an approximate quantized numerical behavior.

## Directory Layout

```
fakequant/
├── mxfp4_ascendc/   # MXFP4 Ascend-C fake-quant operator (layout aligned with amct_ops/hifloat8_cast)
│   ├── op_kernel/   # device kernel + tiling
│   ├── op_extension/# Torch host + TORCH_LIBRARY registration
│   ├── python/mxfp4/# Python wrapper
│   ├── reference/   # pure PyTorch reference implementation
│   └── tests/
├── mxfp4_qat/       # MXFP4 quantization-aware training (QAT): STE-based nn.Linear replacement
│   ├── fake_quant.py# MXFP4 QDQ + STE autograd.Function + quantizer module
│   └── linear.py    # MXFP4QATLinear + convert_to_mxfp4_qat
├── README.md
└── README_en.md
```

`mxfp4_ascendc/` targets **inference-side accuracy verification** (producing fake-quant values quickly). `mxfp4_qat/` targets **training** (letting the model adapt to MXFP4 error during training). On NPU, the latter automatically reuses the former's operator for acceleration.

## Notes

- This module is experimental (`experimental`). Interfaces and implementations may change as hardware capability evolves.
- Fake-quant results are for accuracy alignment and scheme validation. They are not equivalent to the performance of real low-bit operators on the target hardware.
- The MXFP4 operator follows the three-layer layout of `amct_ops/hifloat8_cast`, but stays under `amct_pytorch/experimental/` during the experimental stage and is not moved into `amct_ops/` yet.

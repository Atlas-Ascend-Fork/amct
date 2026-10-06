# MXFP4 QAT Linear (experimental)

A **minimal usable** MXFP4 quantization-aware training (QAT) implementation: `MXFP4QATLinear` can replace `torch.nn.Linear` directly. The forward pass fake-quantizes values with MXFP4 numerics; the backward pass uses STE (straight-through estimator) to send gradients back to the high-precision master weights, so the model adapts to MXFP4 quantization error during training.

This directory only provides **training-side quantization operators and layers**. It does not include a training framework. Section 5 describes how to plug them into your own stack (Megatron / DeepSpeed / HuggingFace Trainer, etc.).

## 1. MXFP4 and STE

**MXFP4** = per-element FP4 E2M1 mantissa + per-block E8M0 (power-of-two) shared scale:

- Every `block_size` (default 32) adjacent elements along the last dimension share one scale;
- `scale = 2^round(log2(max_abs / scale_factor))`, with `scale_factor` defaulting to `6.0`;
- The element codebook is `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` (multiplied by scale).

Quantization is a piecewise-constant function whose derivative is zero almost everywhere, so it cannot back-propagate directly. **STE** uses the quantized value in the forward pass and treats the quantizer as identity in the backward pass, so gradients can pass through the quantization point to the high-precision weights:

```
forward :  y = Q(x)
backward:  dL/dx = dL/dy            # plain STE
           dL/dx = dL/dy * (|x| <= 6*scale)   # clipped STE (clip_grad=True)
```

When `clip_grad=True`, gradients of **saturated (clipped)** elements are zeroed. Because scale is forced onto a power of two (and may round down), the block maximum has about a 50% chance of exceeding `6*scale` and being clipped. Gradients at those positions are misleading; masking them usually makes training more stable.

> This is numerically equivalent to MindSpeed-LLM's `x + (x_q - x).detach()` form. This directory uses an explicit
> `torch.autograd.Function` instead, which makes gradient masking in `backward` straightforward and easier to follow.

## 2. Directory Structure

```
mxfp4_qat/
├── fake_quant.py   # MXFP4 QDQ, STE autograd.Function, quantizer module, config
├── linear.py       # MXFP4QATLinear + convert_to_mxfp4_qat
├── README.md
└── README_en.md
```

`fake_quant.py` depends only on `torch` and can be copied as a single file into any training repository.

## 3. Quick Start

```python
import sys
sys.path.insert(0, ".../amct_pytorch/experimental/fakequant")

from mxfp4_qat import MXFP4QATConfig, convert_to_mxfp4_qat

model.load_state_dict(torch.load(ckpt))                       # start from float pretrained weights
convert_to_mxfp4_qat(model, MXFP4QATConfig(quantize_input=True))
# remaining training code is unchanged
```

## 4. API

```python
import sys
sys.path.insert(0, ".../amct_pytorch/experimental/fakequant")

from mxfp4_qat import MXFP4QATConfig, MXFP4QATLinear, convert_to_mxfp4_qat
```

### `MXFP4QATConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `quantize_weight` | `True` | Whether to fake-quantize weights |
| `quantize_input` | `False` | Whether to fake-quantize layer inputs. `False` → W4A16 (recommended starting point), `True` → W4A4 |
| `block_size` | `32` | Number of elements sharing a scale; the Ascend-C operator only supports 32 |
| `scale_factor` | `6.0` | Larger → smaller scale, higher inlier resolution but more clipping; smaller is the opposite |
| `clip_grad` | `False` | `True` uses clipped STE |
| `backend` | `"auto"` | `"auto"` (use the Ascend-C operator on NPU tensors, otherwise pure PyTorch) / `"torch"` / `"npu"` |

### `MXFP4QATLinear`

A subclass of `nn.Linear`. Forward is `F.linear(Q(x), Q(W), b)` (bias is not quantized).

Because the quantizer has no parameters and no buffers, the **`state_dict` matches a float layer exactly**: float weights can be loaded into a QAT model, and QAT-trained weights can be loaded back into a float model or handed to AMCT's deploy flow for export.

```python
layer = MXFP4QATLinear(in_features, out_features, config=MXFP4QATConfig())
layer = MXFP4QATLinear.from_linear(existing_linear, config)   # reuse the original Parameter, no extra GPU memory
```

### `convert_to_mxfp4_qat(module, config=None, skip_names=())`

Recursively replace all `nn.Linear` modules under `module` in place. `skip_names` does **substring matching** on dotted module paths; a hit skips that subtree:

```python
convert_to_mxfp4_qat(
    model,
    MXFP4QATConfig(quantize_input=True, clip_grad=True),
    skip_names=("lm_head", "embed_tokens"),   # keep sensitive layers in float
)
```

### Low-level functions

```python
mxfp4_quant_dequant(x, block_size=32, scale_factor=6.0, backend="auto")  # no grad, bit-exact with mxfp4_ascendc reference
mxfp4_fake_quant(x, block_size=32, scale_factor=6.0, clip_grad=False, backend="auto")  # with STE
mxfp4_saturation_mask(x, block_size=32, scale_factor=6.0)  # saturation-position mask
MXFP4FakeQuantizer(block_size=32, scale_factor=6.0, clip_grad=False, backend="auto")  # nn.Module form
```

## 5. Integrating with Your Training Framework

### Option 1: The model uses standard `nn.Linear`

After the model is built and pretrained weights are loaded, insert one line **before** creating the `optimizer`. The rest of the training code stays unchanged:

```python
model = build_model()
model.load_state_dict(torch.load(ckpt))          # start from float pretrained weights

convert_to_mxfp4_qat(model, MXFP4QATConfig(quantize_input=True))

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# ... normal training loop ...
```

`from_linear` reuses the original `Parameter` objects, so converting after the optimizer is created still works; converting before the optimizer is safer (avoids dangling parameter-group references).

### Option 2: The framework has a custom Linear (e.g. Megatron `ColumnParallelLinear`)

When inheritance-based replacement is not possible, attach `MXFP4FakeQuantizer` as a quantizer on the layer and call it manually in `forward` — this is how MindSpeed-LLM does it:

```python
from mxfp4_qat import MXFP4FakeQuantizer


class FakeQuantColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quantizer = MXFP4FakeQuantizer()
        self.input_quantizer = MXFP4FakeQuantizer()

    def forward(self, input_, weight=None, **kwargs):
        input_ = self.input_quantizer(input_)
        # The parent forward reads self.weight, so temporarily replace .data then restore.
        # Forward already ran on quantized values; STE sends gradients back to the original high-precision weights.
        original = self.weight.data
        self.weight.data = self.weight_quantizer(original)
        try:
            return super().forward(input_, weight=weight, **kwargs)
        finally:
            self.weight.data = original
```

The same pattern applies to MoE GroupedMatmul expert weights: run the quantizer on `w1` / `w2` and on the permuted expert inputs before calling GMM.

### Option 3: Reuse only the quantization operator

Call `mxfp4_fake_quant(x)` directly. It is a differentiable `Tensor -> Tensor` function and can be placed anywhere (KV cache, logits, residual, etc.).

### Training tips

- **Start QAT fine-tuning from float pretrained weights**; do not train from random initialization.
- **W4A16 first, then W4A4**: activation quantization usually drops accuracy more than weight quantization. Confirm `quantize_input=False` converges first.
- **Use about 1/10 of the pretraining learning rate** with cosine decay.
- **Keep sensitive layers in float**: `lm_head`, embeddings, and the first/last layers are usually excluded via `skip_names`.
- **On NPU, use the Ascend-C backend**: the pure PyTorch path launches more than ten elementwise kernels per QDQ, which is not negligible for large-model training. `backend="auto"` switches automatically on NPU tensors.
- After training, export real low-bit weights with the `amct_pytorch` deploy flow. QAT only makes weights "adapt" to MXFP4; export still needs the regular quantization pipeline.

### Role of the Ascend-C operator

When `backend="auto"`, the `mxfp4` package is looked up in this order: environment variable `MXFP4_ASCENDC_PATH` → sibling `../mxfp4_ascendc/python`. Compile the operator first:

```bash
cd ../mxfp4_ascendc && bash build.sh
```

If it is not compiled or the tensor is not on NPU, the code falls back to the pure PyTorch path (bit-exact results, different speed only). Explicitly setting `backend="npu"` when the operator is unavailable raises a `RuntimeError` with fix instructions.

## 6. Limitations

- This is experimental (`experimental`); interfaces may change.
- Only `nn.Linear` is covered; convolution, Embedding, and matmul inside Attention are not handled.
- Scale and clip thresholds are derived statically from data. Learnable scale / clipping (LSQ, PACT, etc.) is not implemented.
- `block_size != 32` is supported only on the pure PyTorch path.
- Fake quant only reproduces MXFP4 numerical behavior. It does not represent the performance of real low-bit operators on the target hardware.

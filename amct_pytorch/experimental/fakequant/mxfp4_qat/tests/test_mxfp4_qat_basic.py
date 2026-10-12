#!/usr/bin/env python3
"""Self-guarded checks for the experimental MXFP4 QAT linear conversion.

Run directly (``python tests/test_mxfp4_qat_basic.py``); not part of the
repository CI testpaths. Covers:
  * empty-batch forward/backward contract of the reshape fix
  * alias detection: the same Linear registered twice raises RuntimeError
  * unhashable ``nn.Linear`` subclasses (``__eq__`` with ``__hash__ = None``)
    convert without hashing errors
"""

import os
import sys

_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402  (path bootstrap above)
from torch import nn  # noqa: E402

from amct_pytorch.experimental.fakequant.mxfp4_qat import (  # noqa: E402
    MXFP4QATConfig,
    MXFP4QATLinear,
    convert_to_mxfp4_qat,
)


class SharedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(32, 4)
        self.second = self.first


class UnhashableLinear(nn.Linear):
    def __eq__(self, other):
        return isinstance(other, UnhashableLinear)

    __hash__ = None  # type: ignore[assignment]


def test_empty_batch_forward_backward():
    layer = MXFP4QATLinear.from_linear(
        nn.Linear(32, 4), MXFP4QATConfig(quantize_input=True, backend="torch")
    )
    x = torch.empty(0, 32, requires_grad=True)
    out = layer(x)
    assert out.shape == (0, 4)
    out.sum().backward()
    assert x.grad.shape == x.shape


def test_empty_fake_quant_preserves_contract():
    from amct_pytorch.experimental.fakequant.mxfp4_qat import (
        mxfp4_fake_quant,
        mxfp4_saturation_mask,
    )

    x = torch.empty(0, 32, requires_grad=True)
    out = mxfp4_fake_quant(x, backend="torch")
    assert out.shape == x.shape and out.dtype == x.dtype
    assert mxfp4_saturation_mask(x).shape == x.shape
    out.sum().backward()
    assert x.grad.numel() == 0


def test_shared_linear_alias_raises():
    model = SharedLinear()
    try:
        convert_to_mxfp4_qat(model, MXFP4QATConfig(backend="torch"))
    except RuntimeError as err:
        assert "same linear registered twice" in str(err)
    else:
        raise AssertionError("a shared Linear alias must raise RuntimeError")


def test_unhashable_linear_subclass_converts():
    shared = UnhashableLinear(32, 4)
    weight = shared.weight
    model = nn.Sequential(shared)
    convert_to_mxfp4_qat(model, MXFP4QATConfig(backend="torch"))
    layer = model[0]
    assert isinstance(layer, MXFP4QATLinear)
    assert layer.weight is weight


def main():
    test_empty_batch_forward_backward()
    test_empty_fake_quant_preserves_contract()
    test_shared_linear_alias_raises()
    test_unhashable_linear_subclass_converts()
    print("all mxfp4 qat basic checks passed")


if __name__ == "__main__":
    main()

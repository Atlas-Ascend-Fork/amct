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
"""A drop-in ``torch.nn.Linear`` replacement performing MXFP4 QAT."""

from __future__ import annotations

__all__ = [
    "MXFP4QATLinear",
    "convert_to_mxfp4_qat",
]

import torch
import torch.nn.functional as F
from torch import nn

from .fake_quant import MXFP4QATConfig


class MXFP4QATLinear(nn.Linear):
    """``nn.Linear`` whose weight (and optionally input) is MXFP4 fake-quantised.

    Master weights stay in high precision and are what the optimiser updates;
    every forward pass quantises a throwaway copy so the loss reflects MXFP4
    numerics, and the STE routes the gradient back to the master weight.

    Because it subclasses ``nn.Linear`` and the quantizers are stateless, the
    ``state_dict`` is identical to the float layer's, so float checkpoints load
    into a QAT model and QAT checkpoints load back into the float model.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: Whether to learn an additive bias. The bias is never quantised.
        device: Device of the created parameters.
        dtype: Dtype of the created parameters.
        config: MXFP4 QAT settings; defaults to ``MXFP4QATConfig()`` (W4A16).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        config: MXFP4QATConfig | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.config = config if config is not None else MXFP4QATConfig()
        self.weight_quantizer = (
            self.config.make_quantizer() if self.config.quantize_weight else None
        )
        self.input_quantizer = (
            self.config.make_quantizer() if self.config.quantize_input else None
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, config: MXFP4QATConfig | None = None
    ) -> MXFP4QATLinear:
        """Wrap an existing ``nn.Linear``, reusing its parameter objects.

        The weight and bias tensors are adopted rather than copied, so the
        conversion costs no extra memory and any optimiser state or parameter
        group already referencing them stays valid.
        """
        qat_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device="meta",
            dtype=linear.weight.dtype,
            config=config,
        )
        qat_linear.weight = linear.weight
        if linear.bias is not None:
            qat_linear.bias = linear.bias
        return qat_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.input_quantizer is not None:
            input = self.input_quantizer(input)
        weight = self.weight
        if self.weight_quantizer is not None:
            weight = self.weight_quantizer(weight)
        return F.linear(input, weight, self.bias)


def convert_to_mxfp4_qat(
    module: nn.Module,
    config: MXFP4QATConfig | None = None,
    skip_names: tuple[str, ...] = (),
) -> nn.Module:
    """Recursively replace every ``nn.Linear`` in *module* with an MXFP4 QAT one.

    The replacement happens in place; the returned reference is the same object,
    provided only for chaining. Subclasses of ``nn.Linear`` are included so
    framework wrappers (Megatron-style Linear subclasses, etc.) are converted
    rather than left in float. Already converted ``MXFP4QATLinear`` layers are
    skipped, so calling this twice is a no-op.

    Modules are tracked by identity, so linear layers registered under several
    aliases (including ``nn.Linear`` subclasses that define ``__eq__`` without
    ``__hash__``) are converted once and every alias sees the same replacement.

    Args:
        module: Root module to rewrite.
        config: Settings applied to every replaced layer.
        skip_names: Substrings matched against the dotted module path. A match
            skips that submodule and everything below it, which is the usual way
            to keep sensitive layers (``"lm_head"``, ``"embed"``) in float. A
            module reached through *any* matching path stays float through all
            of its aliases, and so do modules inside a skipped container.

    Returns:
        The same *module*, with its linear layers replaced.
    """
    skipped_ids = _collect_skipped_ids(module, skip_names, prefix="", skipped=set())
    return _convert(module, config, prefix="", replacements={}, skipped_ids=skipped_ids)


def _collect_skipped_ids(
    module: nn.Module,
    skip_names: tuple[str, ...],
    prefix: str,
    skipped: set[int],
) -> set[int]:
    """Collect ids of modules reached through any path matching *skip_names*.

    The walk has no side effects. A matched path marks the submodule itself and,
    because the whole subtree is meant to stay float, also marks everything
    below it; aliases elsewhere then skip the very same objects.
    """
    for name, child in module._modules.items():
        if child is None:
            continue
        path = f"{prefix}{name}"
        matched = any(pattern in path for pattern in skip_names)
        if matched:
            skipped.add(id(child))
        if isinstance(child, MXFP4QATLinear):
            continue
        if matched or not isinstance(child, nn.Linear):
            _collect_skipped_ids(child, skip_names, prefix=f"{path}.", skipped=skipped)
    return skipped


def _convert(
    module: nn.Module,
    config: MXFP4QATConfig | None,
    prefix: str,
    replacements: dict[int, MXFP4QATLinear],
    skipped_ids: set[int],
) -> nn.Module:
    # named_children() removes duplicate modules, so an alias could stay in float.
    for name, child in list(module._modules.items()):
        if child is None or id(child) in skipped_ids:
            continue
        if isinstance(child, MXFP4QATLinear):
            continue
        if isinstance(child, nn.Linear):
            # Key by identity: hashing *child* would break ``nn.Linear``
            # subclasses that define ``__eq__`` with ``__hash__ = None``.
            if id(child) not in replacements:
                replacements[id(child)] = MXFP4QATLinear.from_linear(child, config)
            setattr(module, name, replacements[id(child)])
        else:
            _convert(
                child,
                config,
                prefix=f"{prefix}{name}.",
                replacements=replacements,
                skipped_ids=skipped_ids,
            )
    return module

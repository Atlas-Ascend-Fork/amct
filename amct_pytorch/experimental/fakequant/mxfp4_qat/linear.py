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

    Args:
        module: Root module to rewrite.
        config: Settings applied to every replaced layer.
        skip_names: Substrings matched against the dotted module path. A match
            skips that submodule and everything below it, which is the usual way
            to keep sensitive layers (``"lm_head"``, ``"embed"``) in float.

    Raises:
        RuntimeError: If one ``nn.Linear`` object is registered under more than
            one module path. ``skip_names`` is matched per path, so a shared
            layer that only some of its paths match would be left in float on
            one path and quantised on another; the aliasing is reported instead
            of silently splitting the layer.

    Returns:
        The same *module*, with its linear layers replaced.
    """
    plan = _Plan()
    _scan(module, skip_names, "", plan)
    for parent, name, child, path in plan.places:
        # Decided only after the whole tree has been walked, so that the outcome
        # does not depend on the order the paths happen to be visited in.
        if plan.hits[id(child)] > 1:
            raise RuntimeError(
                "convert_to_mxfp4_qat found one linear registered under more "
                f"than one path ({path}); skip_names is matched per path, so a "
                "shared nn.Linear would be quantised on one path and left in "
                "float on another. Register it under a single path"
            )
        setattr(parent, name, MXFP4QATLinear.from_linear(child, config))
    return module


class _Plan:
    """Bookkeeping for one ``convert_to_mxfp4_qat`` call.

    ``hits`` counts how many registration paths reach each linear, and is keyed
    by ``id()`` rather than by the module itself so subclasses that define
    ``__eq__`` with ``__hash__ = None`` never need hashing.
    """

    def __init__(self) -> None:
        self.hits: dict[int, int] = {}
        self.places: list[tuple[nn.Module, str, nn.Linear, str]] = []
        self.scanned: set[int] = set()


def _scan(
    module: nn.Module,
    skip_names: tuple[str, ...],
    prefix: str,
    plan: _Plan,
) -> None:
    """Collect the linears of *module* without replacing any of them.

    ``_modules`` is walked instead of ``named_children()`` because the latter
    drops repeated modules. Every linear is counted, including the ones on a
    path that ``skip_names`` matches: replacing a layer before knowing that
    another path keeps it in float would split the shared object, so the
    decision is deferred until the whole tree is known.
    """
    if id(module) in plan.scanned:
        return
    plan.scanned.add(id(module))
    for name, child in list(module._modules.items()):
        if child is None:
            continue
        path = f"{prefix}{name}"
        if isinstance(child, MXFP4QATLinear):
            continue
        if isinstance(child, nn.Linear):
            plan.hits[id(child)] = plan.hits.get(id(child), 0) + 1
            if any(pattern in path for pattern in skip_names):
                continue
            plan.places.append((module, name, child, path))
            continue
        if any(pattern in path for pattern in skip_names):
            continue
        _scan(child, skip_names, f"{path}.", plan)
    return

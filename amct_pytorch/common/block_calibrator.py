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
"""
Block calibrator registry used by `sequential_calibrate`.

`sequential_calibrate` is a generic per-block orchestration entry: it only
partitions the model into blocks, captures each block's input and iterates the
calibration data. The algorithm-specific "how to calibrate one block" is
delegated to a registered calibrator.

Calibrators are NOT registered per algorithm. They are classified by whether
the per-block optimization needs gradients:

- ``forward``:  gradient-free per-block optimization (e.g. grid search / MSE
  reconstruction without backprop).
- ``gradient``: gradient-based per-block optimization (e.g. learnable
  transforms tuned by reconstruction loss + backprop).

This keeps `sequential_calibrate` algorithm-agnostic: it only routes a block to
the ``forward`` or ``gradient`` calibrator and never inspects algorithm names.

Calibrator contract
-------------------
A calibrator is a callable with the following signature::

    calibrator(model, block_name, block, quant_modules, block_inputs, **kwargs)

- model:         the full model (torch.nn.Module).
- block_name:    the block module name, e.g. "layers.0".
- block:         the block submodule (torch.nn.Module).
- quant_modules: list of the quantized modules inside this block.
- block_inputs:  a list of the block input tensors captured during the
                 calibration pass (one entry per calibration sample). Algorithms
                 that only need a single representative input (e.g. grid search)
                 may take `block_inputs[-1]`; algorithms that need the full
                 batch (e.g. gradient descent) may stack the whole list.
- **kwargs:      passthrough params from `sequential_calibrate` (e.g. args /
                 data_loader / forward_kwargs).

Module matcher
--------------
Because dispatch no longer depends on the algorithm name, each calibrator must
declare how to recognize the quantized modules it handles by passing a
``matcher`` callable ``matcher(module) -> bool`` when registering. For example
the ``forward`` calibrator matches gradient-free quantized modules and the
``gradient`` calibrator matches modules exposing trainable parameters.

A calibrator is responsible for running the per-block search / optimization and
persisting the resulting quantization parameters back onto its modules.
"""

# Calibrator kinds: split by whether the per-block optimization needs gradients.
FORWARD = 'forward'
GRADIENT = 'gradient'
_CALIBRATOR_KINDS = (FORWARD, GRADIENT)

_BLOCK_CALIBRATORS = {}
_BLOCK_MATCHERS = {}


def register_block_calibrator(kind, matcher=None):
    """
    Decorator registering a block calibrator for a calibrator kind.
    Args:
        kind: str, calibrator kind, one of 'forward' (gradient-free) or
            'gradient' (gradient-based).
        matcher: optional callable `matcher(module) -> bool` used to detect the
            quantized modules this calibrator handles. Since dispatch is
            algorithm-agnostic, a matcher is required for a calibrator to be
            discovered automatically.
    """
    if kind not in _CALIBRATOR_KINDS:
        raise ValueError(
            f"Invalid block calibrator kind '{kind}', expected one of {_CALIBRATOR_KINDS}."
        )

    def decorator(func):
        _BLOCK_CALIBRATORS[kind] = func
        if matcher is not None:
            _BLOCK_MATCHERS[kind] = matcher
        return func

    return decorator


def get_block_calibrator(kind):
    return _BLOCK_CALIBRATORS.get(kind)


def get_block_matcher(kind):
    return _BLOCK_MATCHERS.get(kind)


def iter_registered_matchers():
    return tuple(_BLOCK_MATCHERS.items())


def registered_kinds():
    return tuple(_BLOCK_CALIBRATORS.keys())

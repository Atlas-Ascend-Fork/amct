#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Fixed point(fp16fixed) shift value calibration kernel plug-in point.

This module is the single plug-in point for the fp16fixed conv / deconv / matmul
reference implementations used by ``ShiftCalibrationModule`` to search the
per-layer ``fixed_shift_value``.

Algorithm contract (inlined below; the ONNX twin is the ``ShiftCalibration``
custom op of the amct-graph repo, ``amct_onnx/amct_onnx_op/src/shift_calibration_kernel.cpp``):

- block split: activation and second operand are split along the reduction
  axis into fp16 blocks of group size 16, zero padded to a multiple of 16;
- per-block fixed point pipeline: take the max exponent of the block
  (``input_max_exp`` for the activation, ``weight_max_exp`` for the second
  operand, INT8 weights do not join the exponent search), derive the runtime
  ``block_shift``:
    - fp16_fp16: ``clamp(fixed_shift_value - input_max_exp - weight_max_exp,
      -26, 36)``
    - fp16_int8: ``fixed_shift_value - input_max_exp``
  then fp16 -> int16 mantissa alignment, 16-element integer MAC and an
  arithmetic right shift by ``block_shift`` onto the global fixed point scale
  ``2 ** (58 - fixed_shift_value)`` (fp16_fp16) or
  ``2 ** (29 - fixed_shift_value)`` (fp16_int8);
- candidate search: fp16_fp16 over ``{34, ..., 43}``, fp16_int8 over
  ``{5, ..., 14}``; pick the candidate whose operator output has the
  minimum MSE against the FP32 reference. The bias is part of the
  search on both sides (candidate output and reference) in this PyTorch
  implementation -- a deliberate deviation from the ONNX kernel spec,
  consistent with the shipped F16xF16 classes.

Device policy: the search runs on the module device --
``compute_shift_value_sse`` moves the input to ``module.weight.device``
where the fp16fixed operator class is built (``from_module``). The fp16_int8
int8 rebuild goes through a numpy (CPU-only) chain and the rebuilt
operands move back with the operator class (``int8_weight`` /
``weight_scale`` are registered buffers), so both regimes stay
device-consistent end to end.

The fp16fixed implementations live in ``nn/module/fp16_fixed_point`` (F16xF16 / F16xS8
fixed point simulation classes). ``compute_shift_value_sse`` dispatches on
the module type in the fp16_fp16 regime and rebuilds the deployed int8
weight from the record factors in the fp16_int8 regime (the weight
calibration writes the fake-quant weight and the scale_w / offset_w /
wts_type entries into the record file before the shift calibration
modules are inserted, so re-quantizing the module weight with the very
same ``weight_quant_np`` as the deploy path recovers the deployed int8
weight bit-exactly for both arq_quantize and ada_quantize).

The fp16_int8 regime falls back to the regime default value (via
``NotImplementedError``) when the weight quant factors are missing or
malformed in the record file, the weight is not INT8, the weight
quantization is asymmetric (``offset_w != 0``: the F16xS8 MAC has no
offset path) or the module type has no fp16fixed implementation.
"""

import torch

from ....custom_op.arq.arq import weight_quant_np
from ....nn.module import fp16_fixed_point as fp16fixed_pkg
from ....utils.weight_quant_api import adjust_deconv_weight_shape
from ....utils.vars import FIXED_SHIFT_VALUE_FP16_FP16_CANDIDATES
from ....utils.vars import FIXED_SHIFT_VALUE_FP16_FP16_DEFAULT
from ....utils.vars import FIXED_SHIFT_VALUE_FP16_INT8_CANDIDATES
from ....utils.vars import FIXED_SHIFT_VALUE_FP16_INT8_DEFAULT

# fp16fixed operator classes (module level: the full import graph including
# these edges is acyclic, verified by Tarjan SCC over the package)
_FP16FIXED_OPERATORS = {
    'Linear': 'Fp16FixedLinear',
    'Conv2d': 'Fp16FixedConv2d',
    'ConvTranspose2d': 'Fp16FixedConvTranspose2d',
}

_FP16FIXED_S8_OPERATORS = {
    'Linear': 'Fp16FixedLinearS8',
    'Conv2d': 'Fp16FixedConv2dS8',
    'ConvTranspose2d': 'Fp16FixedConvTranspose2dS8',
}

# the F16xS8 path only models an INT8 weight
WEIGHT_INT8_NUM_BITS = 8


def compute_shift_value_sse(
    input_tensor, weight, quant_regime, module, weight_quant_info=None
):
    """Compute per-candidate sum of squared error of one calibration batch.

    Args:
        input_tensor: torch.Tensor, the activation fed into ``module``.
            2D ``[m, k]`` for Linear, 4D NCHW for Conv2d / ConvTranspose2d.
        weight: torch.Tensor, the second operand of the operator. Kept for
            interface compatibility; the fp16fixed operator classes take the
            weight from the module itself.
        quant_regime: str, 'fp16_fp16' or 'fp16_int8'.
        module: the wrapped torch.nn.Module (Linear / Conv2d /
            ConvTranspose2d), providing stride / padding / dilation /
            groups attributes for the im2col style expansion.
        weight_quant_info: dict, the weight quant factors of the layer read
            from the record file (``{'scale_w', 'offset_w', 'wts_type'}``),
            only used by the fp16_int8 regime to rebuild the deployed
            int8 weight.
    Returns:
        torch.Tensor of shape ``[num_candidates]`` holding the SSE of every
        candidate ``fixed_shift_value`` on this batch, ordered as
        ``FIXED_SHIFT_VALUE_FP16_FP16_CANDIDATES`` (fp16_fp16) or
        ``FIXED_SHIFT_VALUE_FP16_INT8_CANDIDATES`` (fp16_int8).

    Raises:
        NotImplementedError: the quant regime or the module type has no
            fp16fixed implementation plugged in, or the fp16_int8 weight quant
            factors are missing / malformed / not INT8 / asymmetric.
    """
    # the fp16fixed operator is built on the module device (from_module keeps
    # it there); move the input to the same device so the whole search
    # closes on one device. CPU modules keep the previous behaviour
    # (the move is a no-op). The fp16_int8 int8 rebuild runs through a
    # numpy chain on CPU and the rebuilt operands move back with the
    # operator class (int8_weight / weight_scale are registered
    # buffers), so both regimes are device-consistent end to end.
    input_tensor = input_tensor.detach().to(module.weight.device)
    if quant_regime == 'fp16_int8':
        return _compute_int8_regime_sse(input_tensor, module, weight_quant_info)
    if quant_regime != 'fp16_fp16':
        raise NotImplementedError(
            "the fp16fixed implementation of the '{}' regime is not plugged "
            'in yet, fixed_shift_value falls back to the regime default '
            'value.'.format(quant_regime)
        )
    module_type = type(module).__name__
    class_name = _FP16FIXED_OPERATORS.get(module_type)
    if class_name is None:
        raise NotImplementedError(
            'no fp16fixed implementation for module type {}, fixed_shift_value '
            'falls back to the regime default value.'.format(module_type)
        )
    operator_cls = getattr(fp16fixed_pkg, class_name)
    fp16fixed_operator = operator_cls.from_module(module)
    return fp16fixed_operator.compute_candidate_sse(input_tensor)


def _compute_int8_regime_sse(input_tensor, module, weight_quant_info):
    """fp16_int8 regime: rebuild the deployed int8 weight and search."""
    module_type = type(module).__name__
    class_name = _FP16FIXED_S8_OPERATORS.get(module_type)
    if class_name is None:
        raise NotImplementedError(
            'no fp16fixed implementation for module type {}, fixed_shift_value '
            'falls back to the regime default value.'.format(module_type)
        )
    if weight_quant_info is None:
        raise NotImplementedError(
            'no weight quant factors of layer type {} found in the record '
            'file, fixed_shift_value falls back to the regime default '
            'value.'.format(module_type)
        )
    int8_weight, weight_scale = _rebuild_int8_weight(module, weight_quant_info)
    operator_cls = getattr(fp16fixed_pkg, class_name)
    fp16fixed_operator = operator_cls.from_module(module, int8_weight, weight_scale)
    return fp16fixed_operator.compute_candidate_sse(input_tensor)


def _rebuild_int8_weight(module, weight_quant_info):
    """Rebuild the deployed int8 weight from the (fake-quant) module weight.

    Uses the very same ``weight_quant_np`` as the deploy path
    (``insert_weight_quant_pass`` quantizes the graph weights with it at
    save time). The module weight already carries the fake-quant weight
    of ``arq_quantize`` / ``ada_quantize`` (both are written back before
    the shift calibration modules are inserted), so the reconstruction
    is bit-exact with the deployed weight. The ConvTranspose weight is
    quantized in the transposed (graph) layout, mirroring the deploy
    path.

    Returns:
        (int8_weight, weight_scale): the int8 weight with the torch
        layout of ``module.weight`` and the per-output-channel weight
        scale.
    """
    if weight_quant_info.get('wts_type') != 'INT8':
        raise NotImplementedError(
            "the F16xS8 path needs an INT8 weight, got wts_type '{}', "
            'fixed_shift_value falls back to the regime default '
            'value.'.format(weight_quant_info.get('wts_type'))
        )
    offset_w = weight_quant_info.get('offset_w') or []
    scale_w = weight_quant_info.get('scale_w') or []
    if not scale_w or len(offset_w) != len(scale_w):
        raise NotImplementedError(
            'incomplete weight quant factors (scale_w / offset_w), '
            'fixed_shift_value falls back to the regime default value.'
        )
    if any(offset != 0 for offset in offset_w):
        raise NotImplementedError(
            'the F16xS8 MAC has no offset path, asymmetric weight '
            'quantization (offset_w != 0) falls back to the regime '
            'default value.'
        )
    # the deploy path quantizes the float32 weight; cast accordingly
    weight = module.weight.detach().cpu().to(torch.float32).numpy()
    module_type = type(module).__name__
    if module_type == 'ConvTranspose2d':
        # the deploy path quantizes the transposed (graph) layout
        weight = adjust_deconv_weight_shape(module.groups, weight)
    int8_weight = weight_quant_np(weight, scale_w, offset_w, WEIGHT_INT8_NUM_BITS)
    if module_type == 'ConvTranspose2d':
        int8_weight = adjust_deconv_weight_shape(module.groups, int8_weight)
    # the rebuild itself is numpy (CPU-only); from_module moves the
    # rebuilt operands back to the module device (registered buffers)
    return (
        torch.from_numpy(int8_weight),
        torch.tensor(scale_w, dtype=torch.float32),
    )


def get_candidates(quant_regime):
    """Return the candidate list of the quant regime."""
    if quant_regime == 'fp16_int8':
        return list(FIXED_SHIFT_VALUE_FP16_INT8_CANDIDATES)
    return list(FIXED_SHIFT_VALUE_FP16_FP16_CANDIDATES)


def get_default_value(quant_regime):
    """Return the default fixed_shift_value of the quant regime."""
    if quant_regime == 'fp16_int8':
        return FIXED_SHIFT_VALUE_FP16_INT8_DEFAULT
    return FIXED_SHIFT_VALUE_FP16_FP16_DEFAULT

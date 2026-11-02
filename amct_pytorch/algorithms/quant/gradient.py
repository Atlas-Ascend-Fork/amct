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

from amct_pytorch.common.block_calibrator import GRADIENT, register_block_calibrator


def _is_gradient_module(module):
    """
    Matcher for the 'gradient' calibrator: recognize gradient-based quantized
    modules by their non-empty `trainable_params()` (e.g. FlatQuant / AutoRound
    learnable terms). Gradient-free algorithms (e.g. GPTQ) return an empty list.

    Mirrors `_is_forward_module`'s guards: the module must carry a resolved
    `quant_config` (non-quantized wrappers holding nn.Parameter must not be
    picked up) and must not be a shared-input layer (`shared_param_mode` is
    None); only then is `trainable_params()` consulted.
    """
    quant_config = getattr(module, 'quant_config', None)
    if not quant_config:
        return False
    if getattr(module, 'shared_param_mode', None) is not None:
        return False
    params_fn = getattr(module, 'trainable_params', None)
    if not callable(params_fn):
        return False
    return len(list(params_fn())) > 0


@register_block_calibrator(GRADIENT, matcher=_is_gradient_module)
def calibrate_gradient(model, block_name, block, quant_modules, block_inputs, **kwargs):
    """
    Block calibrator (kind='gradient'): placeholder until FlatQuant is integrated.

    The real gradient-based algorithm (FlatQuant) currently runs its own
    calibration entry under `experimental/flatquant` and does not go through
    `sequential_calibrate`. This calibrator is therefore intentionally left
    unimplemented; it will be filled in once FlatQuant's per-block calibration
    can reuse this dispatch path.
    """
    raise NotImplementedError(
        "gradient block calibrator is not implemented yet; it will be provided "
        "when the FlatQuant algorithm is integrated."
    )

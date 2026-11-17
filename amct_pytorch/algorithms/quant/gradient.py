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

import torch

from amct_pytorch.common.block_calibrator import (
    GRADIENT,
    collect_fp_reference,
    register_block_calibrator,
)
from amct_pytorch.common.utils.log import LOGGER


def _is_gradient_module(module):
    """Matcher for the 'gradient' calibrator: recognize gradient-based
    quantized modules by their non-empty `trainable_params()`. Only called
    on modules with a resolved `quant_config` that are not shared-input
    layers (same guards as `_is_forward_module`).
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


def _collect_trainable(quant_modules):
    """De-duplicated trainable params owned by the modules of this block."""
    seen = set()
    params = []
    for module in quant_modules:
        for param in module.trainable_params():
            if id(param) not in seen:
                seen.add(id(param))
                params.append(param)
    return params


@register_block_calibrator(GRADIENT, matcher=_is_gradient_module)
def calibrate_gradient(model, block_name, block, quant_modules, block_inputs, **kwargs):
    """
    Block calibrator (kind='gradient'): gradient-based per-block calibration.

    sequential_calibrate handles block splitting and input capture; this
    kernel implements the per-block training:

    1. replay every captured (args, kwargs) sample through the block's
       float forward to build the fp reference (framework-side
       ``collect_fp_reference``);
    2. enable fake quant via each module's ``set_fake_quant(True)``, then
       call the block's single declared ``block_trainer`` (see the
       contract in amct_pytorch.common.block_calibrator);
    3. disable fake quant and mark the modules ``_calibrated``.

    A trainer that hits a non-finite loss raises, aborting the whole
    calibration run — a NaN-contaminated block must never reach convert.

    Modules need the contract (quant_config / trainable_params /
    set_fake_quant / block_trainer); the kernel itself is algorithm-agnostic
    and imports nothing from any algorithm package.
    """
    verbose = kwargs.get('_verbose', True)

    if not _collect_trainable(quant_modules):
        LOGGER.logw(
            f"gradient block '{block_name}': matched modules report no "
            'trainable params, skip.',
            'BlockCalibrator',
        )
        return

    # block_trainer contract: every dispatchable module declares its training
    # loop as a class attribute; a block must carry exactly one distinct
    # implementation (one algorithm's trainer). Mixed or missing trainers
    # fail fast instead of silently training nothing / the wrong thing.
    trainers = {getattr(module, 'block_trainer', None) for module in quant_modules}
    trainers.discard(None)
    if len(trainers) != 1:
        if not trainers:
            LOGGER.logw(
                f"gradient block '{block_name}': quant modules declare no "
                'block_trainer (see the contract in '
                'amct_pytorch.common.block_calibrator); skip.',
                'BlockCalibrator',
            )
            return
        raise ValueError(
            f"gradient block '{block_name}' mixes {len(trainers)} distinct "
            'block_trainer implementations across its quant modules; a block '
            'carries one trainer. Split the block or share one trainer '
            'implementation.'
        )
    train = trainers.pop()

    forward_kwargs = kwargs.get('forward_kwargs') or {}

    fp_inps, fp_outs, call_kwargs = collect_fp_reference(
        block, block_inputs, forward_kwargs
    )
    n_samples = fp_inps.shape[0]

    # NaN/Inf guard, mirroring the 'forward' calibrator's contract
    if not torch.isfinite(fp_inps).all():
        raise RuntimeError(
            "Run gradient block calibration error! Invalid value(nan or inf) "
            'in activation!'
        )
    if not torch.isfinite(fp_outs).all():
        raise RuntimeError(
            "Run gradient block calibration error! Invalid value(nan or inf) "
            'in fp reference output!'
        )

    # 2. fake-quant training (module-side loop: init hooks + param
    #    grouping + AdamW/cosine), then back to the float forward
    for module in quant_modules:
        module.set_fake_quant(True)
    try:
        train(block, fp_inps, fp_outs, call_kwargs, quant_modules)
    finally:
        for module in quant_modules:
            module.set_fake_quant(False)

    # Release the per-block training working set so N blocks of
    # calibration do not accumulate in the caching allocator's pool.
    # Numerically inert: frees cached memory only.
    if getattr(torch, 'npu', None) is not None and torch.npu.is_available():
        torch.npu.empty_cache()

    # 3. mark calibrated so convert knows the transforms are trained
    for module in quant_modules:
        module._calibrated = True

    if verbose:
        LOGGER.logi(
            f"gradient block '{block_name}' done: modules={len(quant_modules)} "
            f'samples={n_samples}',
            'BlockCalibrator',
        )

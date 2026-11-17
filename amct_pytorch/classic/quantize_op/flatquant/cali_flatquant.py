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
#
# The code snippet comes from FlatQuant (https://github.com/ruikangliu/FlatQuant).
#
# Copyright (c) 2024 ruikangliu. All rights reserved.
#
# This file is part of FlatQuant, which is released under the MIT License.
# See the LICENSE file in the root directory of this source tree
# or at https://opensource.org/licenses/MIT for details.
#
# --------------------------------------------------------------------------------
"""FlatQuant per-block training loop, called by the 'gradient' block
calibrator of sequential_calibrate.

Steps per block:
1. collect float reference outputs via the default float forward
   (framework-side helper, amct_pytorch.common.block_calibrator)
2. train_block: gather learnable params by switch groups, AdamW + cosine,
   minimizing MSE(quant_out, fp_outs) per batch
3. restore the float forward and mark the block calibrated

Block inputs are captured as (args, kwargs) pairs, so positional inputs
(CLIP's attention_mask) and keyword inputs (Qwen3's position_embeddings)
both replay correctly."""

import torch

from amct_pytorch.common.block_calibrator import (
    _expand_kwargs_for_batch,
    collect_fp_reference,
)
from amct_pytorch.common.utils.log import LOGGER

# re-export: these moved to the framework-side block_calibrator so every
# gradient-based algorithm shares one fp-reference collector; kept
# importable from here for the existing callers (flatquant package exports
# and the verification scripts import them from this module's path).


def _get_n_set_parameters_byname(module, required_names):
    """Collect params whose name contains any pattern, set requires_grad."""
    params = []
    for r_name in required_names:
        for name, param in module.named_parameters():
            if name.find(r_name) > -1:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return iter(params)


def _flatquant_modules(block):
    """Reconstruction modules in the block (module contract: flat_config)."""
    return [
        mod
        for mod in block.modules()
        if hasattr(mod, 'flat_config') and hasattr(mod, 'set_fake_quant')
    ]


def _set_fake_quant(block, enabled):
    """Enable fake-quant training on every flatquant module in the block."""
    for mod in _flatquant_modules(block):
        mod.set_fake_quant(enabled)


def train_block(block, fp_inps, fp_outs, layer_kwargs, quant_modules=None):
    """Train one block's learnable transforms: MSE to the fp reference.

    Param groups follow: trans/diag at flat_lr, clip factors at
    10x (experimental tuning result). Raises RuntimeError on a NaN loss —
    a diverged block must abort the calibration instead of shipping
    contaminated transforms to convert.
    """
    quant_modules = quant_modules or _flatquant_modules(block)
    if not quant_modules:
        LOGGER.logw('no learnable flatquant module found, skip block training')
        return
    flat_cfg = quant_modules[0].flat_config
    n_samples = len(fp_inps)

    loss_func = torch.nn.MSELoss()

    # init diag scale from weight/act statistics for faster convergence
    for mod in quant_modules:
        init_fn = getattr(mod, 'init_diag_scale', None)
        if callable(init_fn):
            init_fn(alpha=flat_cfg.diag_alpha)

    for _, param in block.named_parameters():
        param.requires_grad = False

    trained_params = []
    if flat_cfg.cali_trans:
        trained_params.append(
            {
                'params': _get_n_set_parameters_byname(
                    block, ['trans.linear', 'trans.diag_scale']
                ),
                'lr': flat_cfg.flat_lr,
            }
        )
    if flat_cfg.lwc:
        trained_params.append(
            {
                'params': _get_n_set_parameters_byname(block, ['clip_factor_w']),
                'lr': flat_cfg.flat_lr * 10,
            }
        )
    if flat_cfg.lac:
        trained_params.append(
            {
                'params': _get_n_set_parameters_byname(block, ['clip_factor_a']),
                'lr': flat_cfg.flat_lr * 10,
            }
        )
    if not trained_params:
        LOGGER.logw('no learnable params found for flatquant calibration')
        return

    bsz = max(flat_cfg.cali_bsz, 1)
    extra_keys = sorted(k for k in layer_kwargs if k.startswith('_pos_'))

    # ceil: the last partial batch (n_samples % bsz) trains with its own
    # smaller size so every calibration sample participates and the flattened
    # inputs always agree with the batch-expanded kwargs
    n_batches = max(-(-n_samples // bsz), 1)
    optimizer = torch.optim.AdamW(trained_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(flat_cfg.epochs * n_batches, 1),
        eta_min=flat_cfg.flat_lr * 1e-3,
    )

    for epoch in range(flat_cfg.epochs):
        mse = 0.0
        for j in range(n_batches):
            index = j * bsz
            cur_bsz = min(bsz, n_samples - index)
            if cur_bsz <= 0:
                break
            batch_slice = slice(index, index + cur_bsz)
            batch_kwargs = _expand_kwargs_for_batch(layer_kwargs, cur_bsz)
            extra_args = tuple(batch_kwargs.pop(key) for key in extra_keys)
            # fp_inps/fp_outs are (n, 1, seq, hidden): flatten the sample dim
            # into batch so the block sees (bsz, seq, hidden)
            quant_out = block(
                fp_inps[batch_slice].flatten(0, 1), *extra_args, **batch_kwargs
            )
            quant_out = quant_out[0] if isinstance(quant_out, tuple) else quant_out
            target = fp_outs[batch_slice].flatten(0, 1)
            loss = loss_func(target.float(), quant_out.float())
            if not torch.isfinite(loss):
                # hard fail: NaN and Inf alike. An Inf loss passes an
                # isnan-only guard and lets the optimizer step on inf/nan
                # gradients, so by the next epoch the on-module transforms
                # are already contaminated — no partially-trained state is
                # trustworthy, and a run that keeps going would hand a
                # silently broken deploy artifact to convert.
                raise RuntimeError(
                    f'flatquant calibration hit non-finite (NaN/Inf) loss at '
                    f'epoch {epoch}, batch {j}. The block cannot be trusted '
                    'anymore and the calibration run is aborted. Common '
                    'causes: diverging flat_lr, corrupted calibration data, '
                    'or non-finite module inputs; lower flat_lr or re-check '
                    'the calibration set.'
                )
            mse += loss.detach().cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        LOGGER.logd(
            f'flatquant block calib epoch {epoch}, mse: {mse:.8f}',
            'calibrate_flatquant',
        )


def calibrate_flatquant(
    model, block_name, block, quant_modules, block_inputs, **kwargs
):
    """FlatQuant entry for the 'gradient' block calibrator.

    Called once per block by calibrate_gradient (framework dispatch), or
    directly by verification tooling with the same contract. block_inputs
    is the list of captured (args, kwargs) pairs from the calibration
    forward. Returns nothing — transforms are updated in place; the block
    is left in float forward so captured inputs replay consistently."""
    forward_kwargs = kwargs.get('forward_kwargs', {}) or {}
    quant_modules = quant_modules or _flatquant_modules(block)
    if not quant_modules:
        LOGGER.logw(f'block {block_name} has no flatquant module, skip')
        return

    # 1. fp reference outputs, sample by sample, no grad (float forward by default)
    fp_inps, fp_outs, call_kwargs = collect_fp_reference(
        block, block_inputs, forward_kwargs
    )

    # NaN/Inf guard, same contract as the 'forward' calibrator entry
    if not torch.isfinite(fp_inps).all():
        raise RuntimeError(
            "Run flatquant block calibration error! Invalid value(nan or "
            'inf) in activation!'
        )
    if not torch.isfinite(fp_outs).all():
        raise RuntimeError(
            "Run flatquant block calibration error! Invalid value(nan or "
            'inf) in fp reference output!'
        )

    # 2. fake-quant training, then back to float forward
    _set_fake_quant(block, True)
    try:
        train_block(block, fp_inps, fp_outs, call_kwargs, quant_modules)
    finally:
        _set_fake_quant(block, False)
    # mark as calibrated so the deploy construction skips the
    # "transforms are at init values" warning
    for mod in quant_modules:
        mod._calibrated = True

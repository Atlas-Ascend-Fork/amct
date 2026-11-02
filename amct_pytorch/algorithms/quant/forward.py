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
import torch.nn.functional as F

from amct_pytorch.common.block_calibrator import FORWARD, register_block_calibrator
from amct_pytorch.common.utils.log import LOGGER


def _is_forward_module(module):
    """
    Matcher for the 'forward' calibrator: recognize gradient-free quantized
    modules. A quantized module is gradient-free if it carries a resolved
    `quant_config` and exposes no trainable parameters.

    Shared-input modules (owned by a LayerNormQuant pre_op) are excluded: their
    activation quantization is handled by the pre_op, not by the block-level
    grid search.
    """
    quant_config = getattr(module, 'quant_config', None)
    if not quant_config:
        return False
    if getattr(module, 'shared_param_mode', None) is not None:
        return False
    params_fn = getattr(module, 'trainable_params', None)
    if callable(params_fn):
        return len(list(params_fn())) == 0
    return True


def _has_search_contract(module):
    """Return True if the module exposes the block-wise grid-search contract."""
    return (
        callable(getattr(module, 'calib_param_grid', None))
        and callable(getattr(module, 'apply_calib_param', None))
        and callable(getattr(module, 'reset_calib', None))
    )


def _block_output(block, block_input, forward_kwargs):
    if (
        isinstance(block_input, tuple)
        and len(block_input) == 2
        and isinstance(block_input[1], dict)
    ):
        # Captured as an (args, kwargs) pair: replay both positional and keyword
        # inputs (e.g. Qwen3 position_embeddings passed as keyword).
        args, blk_kwargs = block_input
        merged = dict(blk_kwargs)
        if forward_kwargs:
            merged.update(forward_kwargs)
        out = block(*args, **merged)
    elif isinstance(block_input, (tuple, list)):
        out = block(*block_input, **(forward_kwargs or {}))
    else:
        out = block(block_input, **(forward_kwargs or {}))
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _iter_tensors(structure):
    """Recursively yield tensors nested inside args / kwargs structures."""
    if torch.is_tensor(structure):
        yield structure
    elif isinstance(structure, dict):
        for value in structure.values():
            yield from _iter_tensors(value)
    elif isinstance(structure, (tuple, list)):
        for item in structure:
            yield from _iter_tensors(item)


def _summarize_quant_params(quant_modules):
    """Build a compact per-module summary of the selected quant params."""

    def _stat(tensor, name):
        if tensor is None:
            return None
        tensor = tensor.detach().float()
        return f"{name}[min={tensor.min().item():.4g},mean={tensor.mean().item():.4g},max={tensor.max().item():.4g}]"

    summaries = []
    for qm in quant_modules:
        parts = [
            part
            for part in (
                _stat(getattr(qm, 'scale', None), 'smooth'),
                _stat(getattr(qm, 'scale_w', None), 'scale_w'),
                _stat(getattr(qm, 'offset_w', None), 'offset_w'),
                _stat(getattr(qm, 'scale_d', None), 'scale_d'),
                _stat(getattr(qm, 'offset_d', None), 'offset_d'),
            )
            if part is not None
        ]
        summaries.append(f"forward layer[{qm.layer_name}] " + " ".join(parts))
    return summaries


@register_block_calibrator(FORWARD, matcher=_is_forward_module)
def calibrate_forward(model, block_name, block, quant_modules, block_inputs, **kwargs):
    """
    Block calibrator (kind='forward'): gradient-free per-block calibration.

    Only auto_smooth modules are grid-searched: they expose the block-wise
    search contract (`calib_param_grid` / `apply_calib_param` / `reset_calib`)
    and their smoothing strength is not fixed, so it is determined by the
    grid search that minimizes the reconstruction MSE. Other gradient-free
    modules mixed into the same block (e.g. fixed smoothquant / GPTQ / AWQ /
    minmax) have already been finalized by the calibration forward and only
    receive a one-time dump, when they support persistence.
    """
    searchable = [
        qm
        for qm in quant_modules
        if _has_search_contract(qm) and getattr(qm, 'is_auto_smooth', False)
    ]
    if not searchable:
        LOGGER.logi(
            f"forward block '{block_name}': no block-wise search contract, "
            "calibration already finished by the calibration forward.",
            'BlockCalibrator',
        )
        return

    verbose = kwargs.get('_verbose', True)
    forward_kwargs = kwargs.get('forward_kwargs')
    block_input = block_inputs[-1]

    if (
        isinstance(block_input, tuple)
        and len(block_input) == 2
        and isinstance(block_input[1], dict)
    ):
        args, blk_kwargs = block_input
        finite_candidates = list(args) + list(blk_kwargs.values())
    else:
        finite_candidates = (
            list(block_input)
            if isinstance(block_input, (tuple, list))
            else [block_input]
        )
    for inp in _iter_tensors(finite_candidates):
        if not torch.isfinite(inp).all():
            raise RuntimeError(
                "Run forward block calibration error! Invalid value(nan or inf) "
                "in activation!"
            )
    for qm in quant_modules:
        weight = getattr(qm, 'weight', None)
        if weight is not None and not torch.isfinite(weight.data).all():
            raise RuntimeError(
                "Run forward block calibration error! Invalid value(nan or inf) "
                "in weight!"
            )

    param_grid = searchable[0].calib_param_grid()
    mismatched = [
        getattr(qm, 'layer_name', type(qm).__name__)
        for qm in searchable[1:]
        if qm.calib_param_grid() != param_grid
    ]
    if mismatched:
        raise ValueError(
            "auto_smooth grid_num / smooth_range must be consistent within a "
            f"block: block '{block_name}' has a different candidate grid for "
            f"layer(s) {', '.join(repr(name) for name in mismatched)} than "
            f"'{getattr(searchable[0], 'layer_name', type(searchable[0]).__name__)}'. "
            "The block grid search applies a single smoothing strength to all "
            "searchable layers, so configure the same grid_num and smooth_range "
            "for every auto_smooth layer in one block."
        )

    with torch.no_grad():
        for qm in searchable:
            qm.reset_calib()
        ori_out = _block_output(block, block_input, forward_kwargs)

        best_param = None
        min_loss = float('inf')
        candidates = []

        if verbose:
            LOGGER.logi(
                f"forward block calibration start: block={block_name}, grid_num={len(param_grid)}",
                'BlockCalibrator',
            )

        for idx, param in enumerate(param_grid, start=1):
            for qm in searchable:
                qm.apply_calib_param(param)

            quant_out = _block_output(block, block_input, forward_kwargs)
            loss = F.mse_loss(ori_out.float(), quant_out.float())
            loss_val = float(loss)
            candidates.append((param, loss_val))
            if verbose:
                LOGGER.logi(
                    f"forward [{idx}/{len(param_grid)}] param={param} loss={loss_val:.8f}",
                    'BlockCalibrator',
                )
            if torch.isfinite(loss) and loss_val < min_loss:
                min_loss = loss_val
                best_param = param

            for qm in searchable:
                qm.reset_calib()

    if best_param is None:
        raise RuntimeError("forward block calibration error! Got invalid loss.")

    for qm in searchable:
        qm.apply_calib_param(best_param)
        # The final apply is the only true finalization point of the search:
        # dump the selected params here (the module itself does not dump on
        # every candidate apply).
        dump_fn = getattr(qm, 'maybe_dump_quant_params', None)
        if callable(dump_fn):
            dump_fn()

    # Non-searchable modules (e.g. minmax / gptq mixed into the same block) were
    # already finalized by the calibration forward; only dump them once, if they
    # support persistence.
    for qm in quant_modules:
        if qm in searchable:
            continue
        dump_fn = getattr(qm, 'maybe_dump_quant_params', None)
        if callable(dump_fn):
            dump_fn()

    if verbose:
        candidate_str = ", ".join(
            f"param={param}:loss={loss_val:.8f}" for param, loss_val in candidates
        )
        LOGGER.logi(f"forward all candidates -> {candidate_str}", 'BlockCalibrator')
        LOGGER.logi(
            f"forward selected best_param={best_param} min_loss={min_loss:.8f}",
            'BlockCalibrator',
        )
        for summary in _summarize_quant_params(quant_modules):
            LOGGER.logi(summary, 'BlockCalibrator')
    else:
        LOGGER.logi(
            f"forward block '{block_name}' done: best_param={best_param} "
            f"min_loss={min_loss:.8f}",
            'BlockCalibrator',
        )
    return best_param

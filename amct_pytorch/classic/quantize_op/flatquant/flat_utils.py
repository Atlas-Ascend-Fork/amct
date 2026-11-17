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
"""Save/load FlatQuant transform matrices and clip factors.

Loading restores the calibration result so convert can run without
re-calibration."""

from collections import OrderedDict

import torch

from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.common.utils.safe_load import safe_torch_load


def _get_paras_dict_by_name(model, required_names, destination=None, prefix=''):
    if destination is None:
        destination = OrderedDict()
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                destination[prefix + name] = param.detach()
    return destination


def _is_flatquant_module(mod):
    """FlatQuant reconstruction modules carry flat_config + set_fake_quant."""
    return hasattr(mod, 'flat_config') and hasattr(mod, 'set_fake_quant')


#: dump sidecar key carrying {flat index: layer path}; entries other than
#: plain ints (the legacy layout) are skipped by the loader, keeping old
#: dumps loadable. Present dumps always carry it so the loader can verify
#: layer identity instead of trusting the flat index alone.
_LAYER_PATHS_KEY = '_layer_paths'


def save_flat_matrices(model, matrices_path):
    """Dump learned transforms per layer (trans matrices, diag, clip factors).

    Besides the per-index parameter dicts, a '_layer_paths' sidecar maps each
    flat index to the model-side layer path so load_flat_matrices can verify
    the dump really belongs to this model's layer layout."""
    flat_matrices = {}
    layer_paths = {}
    layer_iter = _iter_quant_layers(model, with_names=True)
    for i, (layer_path, layer) in enumerate(layer_iter):
        for mod in layer.modules():
            if _is_flatquant_module(mod):
                # freeze factorized form into matrices first (rep_matrix_only
                # semantics from experimental)
                for trans in _trans_of(mod):
                    if trans is not None:
                        trans.to_eval_mode()
        paras_name = [
            'trans.matrix',
            'trans.diag_scale',
            'clip_factor_w',
            'clip_factor_a',
        ]
        flat_matrices[i] = _get_paras_dict_by_name(layer, required_names=paras_name)
        layer_paths[i] = layer_path
    flat_matrices[_LAYER_PATHS_KEY] = layer_paths
    torch.save(flat_matrices, matrices_path)
    LOGGER.logi('saved parameters at {}'.format(matrices_path))
    return model


def load_flat_matrices(model, matrix_path):
    """Restore dumped transforms; convertible without re-calibration.

    The dump stores eval-mode tensors (matrix_left/right, diag_scale,
    clip factors). Loading therefore must first switch every target trans
    to eval mode so the matrix_* parameter slots exist, then load per
    layer, and finally assert the tensors actually landed (strict=False
    would otherwise silently drop mismatched keys).
    """
    flat_parameters = safe_torch_load(matrix_path)
    layer_paths = (
        flat_parameters.get(_LAYER_PATHS_KEY)
        if isinstance(flat_parameters, dict)
        else None
    )
    layers = list(_iter_quant_layers(model, with_names=True))
    n_loaded = 0
    for i in sorted(k for k in flat_parameters.keys() if isinstance(k, int)):
        if i >= len(layers):
            break
        layer_path, layer = layers[i]
        if layer_paths is not None and layer_paths.get(i) != layer_path:
            # a flat index is only stable while the two models agree on
            # which layers are flatquant; a mismatch here means the dump
            # would silently write another layer's transforms (review).
            raise RuntimeError(
                f'load_flat_matrices: dump layer {i} was saved from '
                f'\'{layer_paths.get(i)}\' but this model\'s layer {i} is '
                f'\'{layer_path}\'; the model layout does not match the '
                f'dump - refusing to restore into the wrong layers.'
            )
        has_flat = any(_is_flatquant_module(m) for m in layer.modules())
        if not has_flat:
            LOGGER.logi(f'not flatquant layer {i}')
            continue
        # eval mode FIRST: creates the matrix_left/right parameter slots
        # the dump refers to. to_eval_mode is idempotent-safe here because
        # a freshly quantized module is never in eval mode yet; calling it
        # freezes the current (init) factorization into matrix form.
        for mod in layer.modules():
            if _is_flatquant_module(mod):
                for trans in _trans_of(mod):
                    if trans is not None:
                        trans.to_eval_mode()
        missing, unexpected = layer.load_state_dict(flat_parameters[i], strict=False)
        # clip_factor/quantizer buffers are intentionally not all in the
        # dump, so only complain about flatquant-core keys that failed.
        core = (
            'matrix_left',
            'matrix_right',
            'matrix_left_inv',
            'matrix_right_inv',
            'diag_scale',
            'clip_factor_a',
            'clip_factor_w',
        )
        dropped = [k for k in unexpected if any(c in k for c in core)]
        if dropped:
            raise RuntimeError(
                f'load_flat_matrices: layer {i} dropped {len(dropped)} '
                f'flatquant key(s) (e.g. {dropped[:3]}); the module layout '
                f'does not match the dump - refusing to silently skip.'
            )
        # the mirror image of 'dropped': a dump entry the module never
        # consumed leaves the transform at its init value while the caller
        # still sees 'restored 1/1' — a silent accuracy trap.
        unfilled = [k for k in missing if any(c in k for c in core)]
        if unfilled:
            raise RuntimeError(
                f'load_flat_matrices: layer {i} missed {len(unfilled)} '
                f'flatquant core key(s) (e.g. {unfilled[:3]}); the dump '
                f'does not match the module - refusing to keep init '
                f'transforms silently.'
            )
        n_loaded += 1
    n_dumped = sum(1 for k in flat_parameters.keys() if isinstance(k, int))
    if layer_paths is None:
        LOGGER.logw(
            'load_flat_matrices: dump carries no layer paths (legacy '
            'format); layer identity is not verified - make sure the load '
            'model matches the save model layout exactly.'
        )
    LOGGER.logi(
        f'load_flat_matrices: restored {n_loaded}/{n_dumped} '
        f'layer(s) from {matrix_path}'
    )
    return model


def _iter_quant_layers(model, with_names=False):
    """Yield decoder/encoder layers that contain flatquant modules.

    Dual-tower models (CLIP): BOTH towers are iterated, vision first then
    text (deterministic order). Single-tower/decoder models resolve their
    one path as before.

    with_names=True yields (layer_path, layer) pairs where layer_path is
    the model-side module path (e.g. 'model.layers.3' or
    'vision_model.encoder.layers.7') — the stable identity used to match
    dump/load layers across models (review: the flat index alone silently
    misaligns when the two models disagree on which layers are flatquant).
    """
    resolved = []
    matched_any = False
    for path in (
        'model.layers',
        'vision_model.encoder.layers',
        'text_model.encoder.layers',
    ):
        obj = model
        ok = True
        for part in path.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            resolved.extend((f'{path}.{idx}', lyr) for idx, lyr in enumerate(obj))
            matched_any = True
            # 'model.layers' is a single-tower/decoder layout: the later
            # CLIP paths never coexist with it, so break for compat.
            if path == 'model.layers':
                break
    if not matched_any:
        return []
    kept = [
        (lp, lyr)
        for lp, lyr in resolved
        if any(_is_flatquant_module(m) for m in lyr.modules())
    ]
    if with_names:
        return kept
    return [lyr for _, lyr in kept]


def _trans_of(mod):
    """All transform objects attached to a reconstruction module."""
    return [
        getattr(mod, n, None)
        for n in (
            'ln_trans',
            'up_gate_trans',
            'down_trans',
            'o_trans',
            'vcache_trans',
            'kcache_trans',
            'fc1_trans',
            'fc2_trans',
        )
    ]

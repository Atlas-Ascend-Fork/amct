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

Gradient-kernel building blocks (framework-side, algorithm-agnostic)
-------------------------------------------------------------------
The 'gradient' calibrator kernel orchestrates the per-block training; the
fp-reference collection lives here (below) so every gradient-based algorithm
shares it, while the actual training loop stays with each algorithm (see the
``block_trainer`` contract in ``trainable_params``-style contracts below):

- ``collect_fp_reference(block, block_inputs, forward_kwargs)``: replay the
  captured (args, kwargs) samples through the float forward, returning
  ``(fp_inps, fp_outs, call_kwargs)`` stacked as ``(n, 1, seq, hidden)``.

block_trainer contract (for the 'gradient' calibrator)
------------------------------------------------------
A module dispatchable to the 'gradient' kernel declares its per-block training
loop as a ``block_trainer`` class attribute::

    block_trainer = staticmethod(their_train_block)

with the signature
``(block, fp_inps, fp_outs, call_kwargs, quant_modules) -> None``. The kernel
collects the set of ``block_trainer`` objects across the block's quant modules
and requires exactly one distinct implementation (a block mixing two
gradient algorithms with different trainers must be split, or share one
trainer); a missing declaration fails fast at calibration time instead of
silently training nothing.
"""

import torch

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


# --------------------------------------------------------------------------
# fp-reference collection shared by gradient-based calibrators (moved from
# the flatquant calibration module: no algorithm knowledge below)
# --------------------------------------------------------------------------


def _replay_args_kwargs(sample):
    """Normalize one captured block input to (args, kwargs).

    The framework's pre-hook captures ``(tuple(args), dict(kwargs))`` pairs;
    older direct callers may pass a bare tensor or a ``(tensor, kwargs)``
    pair — tolerate both so direct calibration entries stay importable and
    callable as-is (used by the verification scripts).
    """
    if isinstance(sample, tuple) and len(sample) == 2 and isinstance(sample[1], dict):
        first = sample[0]
        return (tuple(first) if isinstance(first, (tuple, list)) else (first,)), sample[
            1
        ]
    if isinstance(sample, (tuple, list)):
        return tuple(sample), {}
    return (sample,), {}


def _same_value(prev, cur):
    """Equality that recurses into tuples/lists so nested tensors compare
    element-wise (review).

    Qwen3's ``position_embeddings`` is a ``(cos, sin)`` tuple: a plain
    ``prev == cur`` on tuples of tensors returns a multi-element tensor and
    ``if not same`` raises "Boolean value of Tensor is ambiguous" — even for
    two separately-generated but numerically equal rope tables, which is the
    normal case for same-seqlen calibration samples."""
    if torch.is_tensor(prev) and torch.is_tensor(cur):
        return torch.equal(prev, cur)
    if (
        isinstance(prev, (tuple, list))
        and isinstance(cur, (tuple, list))
        and len(prev) == len(cur)
    ):
        return all(_same_value(p, c) for p, c in zip(prev, cur))
    return prev == cur


def _first_tensor(args):
    """The block's leading input tensor (the one stacked for training)."""
    for item in args:
        if torch.is_tensor(item):
            return item
    raise ValueError(
        'block input has no leading tensor to train against '
        f'(got args of types {[type(a).__name__ for a in args]})'
    )


#: kwargs that must never ride the training forward: a cache object holds
#: per-sample state (batch 1) that a flattened (bsz, ...) training batch
#: cannot replay against, and kv-cache generation is meaningless in
#: block-local MSE training anyway.
_NON_TRAINABLE_KWARGS = ('past_key_values', 'use_cache', 'cache_position')


def _expand_kwargs_for_batch(kwargs, bsz):
    """Expand per-sample forward kwargs onto a flattened batch axis.

    fp reference inputs are stacked sample-by-sample into (bsz, seq, hidden)
    (batch dim flattened), so every batch-shaped kwarg must repeat along
    dim 0. Known shapes: 4-D causal attention masks (bsz, 1, seq, seq) and
    2-D position ids (bsz, seq). Nested position_embeddings tuples carry
    their own leading batch dim and repeat the same way. None stays None.
    Cache kwargs are dropped (see _NON_TRAINABLE_KWARGS).
    """
    expanded = {}
    for key, value in kwargs.items():
        if key in _NON_TRAINABLE_KWARGS:
            continue
        if value is None:
            expanded[key] = None
        elif torch.is_tensor(value):
            expanded[key] = (
                value.repeat(bsz, *([1] * (value.dim() - 1)))
                if value.dim() >= 2 and value.shape[0] == 1
                else value
            )
        elif isinstance(value, (tuple, list)):
            expanded[key] = type(value)(
                _expand_kwargs_for_batch({'_': item}, bsz)['_'] for item in value
            )
        else:
            expanded[key] = value
    return expanded


def collect_fp_reference(block, block_inputs, forward_kwargs=None):
    """Replay every captured input through the float forward.

    Returns (fp_inps, fp_outs, call_kwargs): inputs normalized to a leading
    batch dim of 1, stacked into (n, 1, seq, hidden); outputs stacked the
    same way; kwargs merged from the first sample plus forward_kwargs.
    Positional args after the leading tensor (e.g. CLIP's attention_mask)
    are folded into call_kwargs under their positional name '_pos_1', ...,
    so the training loop replays them without knowing the block signature.
    """
    fp_outs, fp_inps = [], []
    pos_kwargs = {}
    ref_kwargs = None
    with torch.no_grad():
        for sample_idx, sample in enumerate(block_inputs):
            args, kwargs = _replay_args_kwargs(sample)
            merged = dict(kwargs)
            if forward_kwargs:
                merged.update(forward_kwargs)
            inp = _first_tensor(args)
            if inp.dim() == 3 and inp.shape[0] != 1:
                # the training loop flattens the stacked sample dim into the
                # batch dim and replays per-batch kwargs, so every sample must
                # carry batch dim 1 — a (bsz>1, seq, hidden) sample would be
                # misinterpreted by the block's own view() reshapes. Fail fast
                # instead of producing a confusing downstream shape error.
                raise ValueError(
                    f'gradient calibration expects one sample per captured '
                    f'input with batch dim 1, got batch dim {inp.shape[0]}. '
                    'Split multi-sample batches into individual samples '
                    '(e.g. iterate the DataLoader without batching, or pass '
                    'per-sample tensors).'
                )
            # kwargs cross-sample consistency: the training loop replays the
            # FIRST sample's kwargs for the whole batch (_expand_kwargs_for_
            # batch), while fp_outs below are computed with each sample's own
            # kwargs — a per-sample difference would train the transforms
            # against a mixed reference (review). Cache kwargs are dropped by
            # the training loop, so they are exempt from the check.
            trainable_kwargs = {
                k: v for k, v in kwargs.items() if k not in _NON_TRAINABLE_KWARGS
            }
            if ref_kwargs is None:
                ref_kwargs = trainable_kwargs
            else:
                for key in sorted(set(ref_kwargs) | set(trainable_kwargs)):
                    prev, cur = ref_kwargs.get(key), trainable_kwargs.get(key)
                    same = _same_value(prev, cur)
                    if not same:
                        raise ValueError(
                            f'gradient calibration requires identical kwarg '
                            f'\'{key}\' across calibration samples (batch '
                            'training replays the first sample\'s kwargs for '
                            f'the whole batch), but it varies between sample '
                            f'0 and sample {sample_idx}: {prev!r} vs {cur!r}.'
                        )
            out = block(inp, *args[1:], **merged)
            fp_outs.append(out[0] if isinstance(out, (tuple, list)) else out)
            fp_inps.append(inp)
            for idx, extra in enumerate(args[1:], start=1):
                key = f'_pos_{idx}'
                # the training loop flattens all samples into one batch, so
                # every sample must carry the SAME positional args (e.g.
                # attention_mask); silently keeping the last one would pair
                # the fp reference of sample i with the mask of sample n-1.
                if key in pos_kwargs:
                    prev = pos_kwargs[key]
                    same = _same_value(prev, extra)
                    if not same:
                        raise ValueError(
                            f'gradient calibration requires identical '
                            f'positional argument #{idx} across calibration '
                            f'samples (batch training replays one value for '
                            f'the whole batch), but it varies between samples.'
                        )
                else:
                    pos_kwargs[key] = extra
    fp_outs = torch.stack(fp_outs)  # (n, 1, seq, hidden)
    fp_inps = torch.stack(fp_inps)  # (n, 1, seq, hidden)
    _, kwargs = _replay_args_kwargs(block_inputs[0])
    call_kwargs = dict(kwargs)
    if forward_kwargs:
        call_kwargs.update(forward_kwargs)
    # trailing positional args replay through the same batch-expansion path;
    # `_pos_*` keys never collide with real kwargs
    for key, value in pos_kwargs.items():
        call_kwargs.setdefault(key, value)
    return fp_inps, fp_outs, call_kwargs

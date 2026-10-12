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
"""Expert CPU offload for the blockwise eval / extract_ptq_data paths.

A single large MoE decoder layer does not always fit on one accelerator. Under
``eval_mode=quant`` the pressure roughly doubles: every :class:`QuantLinear`
keeps ``cached_eval_weight``, a dequantized copy of its weight that is the same
size as the original, so a layer with 256 routed experts needs ~2x its bf16
footprint on device.

``workflows/llm_ptq.py`` already handles this by moving one expert at a time
(``.to(device)`` / ``.cpu()``). The eval-side paths instead pin every tensor of
a layer through a device map, which has no per-expert streaming equivalent --
this module provides it: routed experts stay resident in host memory and are
streamed onto the execution device one at a time via an
:class:`~accelerate.hooks.AlignDevicesHook` subclass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from accelerate.hooks import AlignDevicesHook
from loguru import logger

# QuantLinear stores its eval-time dequantized weight as a *plain attribute*,
# not as a parameter or buffer. nn.Module.to() therefore neither moves nor frees
# it: an offloaded expert would keep one device-sized copy alive per expert,
# silently defeating the offload. Any offload implementation must drop it
# explicitly.
_CACHED_WEIGHT_ATTRS = ("cached_eval_weight",)
_CACHED_KEY_ATTRS = ("_cached_transform_key",)

# `auto` turns offload on only when a single layer (with the eval-time weight
# duplication accounted for) is expected to fill most of the device memory.
AUTO_OFFLOAD_MEMORY_FRACTION = 0.75

EXPERT_OFFLOAD_MODES = ("auto", "on", "off")


def clear_quant_weight_cache(module: nn.Module) -> int:
    """Drop QuantLinear eval weight caches held under ``module``.

    Returns the number of cached weights released. Safe to call on modules that
    hold no quantized linear (returns 0).
    """
    cleared = 0
    for mod in module.modules():
        for attr in _CACHED_WEIGHT_ATTRS:
            if getattr(mod, attr, None) is not None:
                setattr(mod, attr, None)
                cleared += 1
        for attr in _CACHED_KEY_ATTRS:
            if getattr(mod, attr, None) is not None:
                setattr(mod, attr, None)
    return cleared


class ExpertOffloadHook(AlignDevicesHook):
    """``AlignDevicesHook(offload=True)`` that also frees quant weight caches.

    ``init_hook`` snapshots the expert's tensors into a host-side ``weights_map``
    and turns the module's parameters into ``meta`` placeholders; ``pre_forward``
    copies them back onto ``execution_device``; ``post_forward`` returns them to
    ``meta``. Because the dequantized eval weight lives outside the parameter
    tree, it is cleared here as well.
    """

    def post_forward(self, module, output):
        output = super().post_forward(module, output)
        clear_quant_weight_cache(module)
        return output

    def detach_hook(self, module):
        module = super().detach_hook(module)
        clear_quant_weight_cache(module)
        return module


def estimate_module_bytes(module: nn.Module) -> int:
    """Sum the storage of every parameter and buffer under ``module``.

    Works on ``meta`` tensors (shapes and dtypes are known there), so it can be
    called on an ``init_empty_weights`` block before anything is materialized.
    """
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor is None:
            continue
        total += tensor.numel() * tensor.element_size()
    return total


def _device_index(device) -> int | None:
    if device is None:
        return None
    text = str(device)
    if ":" in text:
        try:
            return int(text.rsplit(":", 1)[1])
        except ValueError:
            return None
    return None


def get_accelerator_device_count(device) -> int | None:
    """Number of visible accelerators, or ``None`` when not on an accelerator."""
    npu = getattr(torch, "npu", None)
    if npu is not None and str(device).startswith("npu"):
        try:
            return int(npu.device_count())
        except Exception:  # pragma: no cover - runtime/driver dependent
            return None
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.device_count())
    return None


def get_accelerator_total_memory(device) -> int | None:
    """Total memory in bytes of ``device``, or ``None`` when unavailable."""
    npu = getattr(torch, "npu", None)
    try:
        if npu is not None and str(device).startswith("npu"):
            _, total = npu.mem_get_info(device)
            return int(total)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = _device_index(device)
            _, total = torch.cuda.mem_get_info(index if index is not None else device)
            return int(total)
    except Exception:  # pragma: no cover - runtime/driver dependent
        return None
    return None


def should_offload_experts(
    mode,
    *,
    quantized: bool,
    block_bytes: int | None = None,
    device=None,
) -> bool:
    """Resolve the ``--expert_offload`` policy for one decoder block.

    - ``off``: never offload (historical behavior).
    - ``on``: always offload.
    - ``auto`` (default): offload only when the block runs on a single
      accelerator *and* its weights -- doubled when quantized, because
      ``cached_eval_weight`` mirrors every weight -- are expected to exceed
      :data:`AUTO_OFFLOAD_MEMORY_FRACTION` of that accelerator's memory.

    ``auto`` degrades to "no offload" whenever the block size or the device
    memory cannot be determined, so models that used to fit keep their old
    (fast, fully resident) behavior.
    """
    normalized = str(mode or "auto").strip().lower()
    if normalized not in EXPERT_OFFLOAD_MODES:
        raise ValueError(
            f"Unsupported expert_offload mode '{mode}'; "
            f"expected one of {list(EXPERT_OFFLOAD_MODES)}."
        )
    if normalized == "off":
        return False
    if normalized == "on":
        return True

    device_count = get_accelerator_device_count(device)
    if device_count is None:
        logger.warning(
            "expert_offload=auto: no accelerator device count available for "
            "'{}'; keeping every expert resident. Pass --expert_offload on to "
            "force CPU offload.",
            device,
        )
        return False
    if device_count > 1:
        return False
    total_memory = get_accelerator_total_memory(device)
    if not total_memory or not block_bytes:
        logger.warning(
            "expert_offload=auto: cannot estimate memory (block_bytes={}, "
            "device_total={}); keeping every expert resident. Pass "
            "--expert_offload on to force CPU offload.",
            block_bytes,
            total_memory,
        )
        return False
    # Eval-mode quantization keeps a dequantized twin of every weight.
    scale = 2 if quantized else 1
    needed = block_bytes * scale
    if needed > AUTO_OFFLOAD_MEMORY_FRACTION * total_memory:
        logger.info(
            "expert_offload=auto: one decoder block needs ~{:.1f} GiB on {} "
            "(device total {:.1f} GiB, quantized={}); parking routed experts on "
            "CPU and streaming them per forward.",
            needed / 2**30,
            device,
            total_memory / 2**30,
            quantized,
        )
        return True
    return False

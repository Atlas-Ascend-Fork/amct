# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from ..common.utils.log import LOGGER


@dataclass(slots=True)
class ModelBackend:
    name: str


def detect_backend(model: nn.Module) -> ModelBackend:
    module_name = model.__class__.__module__.lower()
    has_save_pretrained = hasattr(model, "save_pretrained") and callable(
        getattr(model, "save_pretrained")
    )
    has_config = hasattr(model, "config")

    if "transformers" in module_name:
        name = "huggingface"
    elif "modelscope" in module_name:
        name = "modelscope"
    elif has_save_pretrained and has_config:
        name = "pretrained-module"
    else:
        name = "torch"
    return ModelBackend(name=name)


def _try_set_int_attr(obj: Any, name: str, value: Any) -> None:
    if not hasattr(obj, name):
        return
    try:
        setattr(obj, name, int(value))
    except (TypeError, ValueError) as exc:
        # Do not silently leave a stale config dim: pruned weights with an
        # unchanged config size cause a shape mismatch on save/reload.
        LOGGER.logw(
            f"[amct.prune] could not update config attribute '{name}' to int({value!r}) "
            f"({type(exc).__name__}: {exc}); the config dim may be stale relative to the "
            f"pruned weights (save/reload could mismatch).",
            "amct_prune",
        )


def patch_common_config(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return
    meta = getattr(model, "_amct_prune_meta", None)
    if not isinstance(meta, dict):
        return

    # A scalar config field can only describe uniform-width pruning; with per-layer
    # widths (allocation/skip_layers) a scalar would cause a shape mismatch on reload.
    dense_hidden = meta.get("dense_hidden_size")
    dense_widths = meta.get("_dense_widths")
    if dense_widths and len(dense_widths) > 1:
        LOGGER.logw(
            f"[amct.prune] per-layer FFN widths differ {sorted(dense_widths)}; not modifying config.intermediate_size"
            f" (a single scalar cannot describe non-uniform-width pruning;"
            f" save/reload needs per-layer sizes).", "amct_prune")
    elif dense_hidden is not None:
        for name in ("intermediate_size", "ffn_hidden_size", "n_inner"):
            _try_set_int_attr(config, name, dense_hidden)

    moe_experts = meta.get("moe_num_experts")
    if moe_experts is not None:
        for name in ("num_local_experts", "num_experts"):
            _try_set_int_attr(config, name, moe_experts)

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
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch.nn as nn

from .. import simulate
from ..config import PruneConfig
from ..utils import (
    expand_flatten_channel_indices,
    get_submodule,
    iter_direct_named_modules,
    prune_batchnorm2d,
    prune_conv2d_in_channels,
    prune_conv2d_out_channels,
    prune_linear_in_features,
    replace_submodule,
)
from .base import BasePruningDomain


_PASSTHROUGH_MODULES = (
    nn.ReLU,
    nn.GELU,
    nn.SiLU,
    nn.Identity,
    nn.Dropout,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveMaxPool2d,
)


@dataclass
class CNNChannelTarget:
    producer_path: str
    bn_path: Optional[str]
    consumer_path: str


def _find_consumer(
    parent: nn.Module, items: List[Tuple[str, nn.Module]], start_idx: int
) -> Tuple[Optional[str], Optional[nn.Module], Optional[str], Optional[nn.Module]]:
    bn_name: Optional[str] = None
    bn_module: Optional[nn.Module] = None
    consumer_name: Optional[str] = None
    consumer_module: Optional[nn.Module] = None

    for j in range(start_idx + 1, len(items)):
        name, mod = items[j]
        if isinstance(mod, nn.BatchNorm2d) and bn_module is None:
            bn_name, bn_module = name, mod
            continue
        if isinstance(mod, _PASSTHROUGH_MODULES):
            continue
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            consumer_name, consumer_module = name, mod
            break
        break
    return bn_name, bn_module, consumer_name, consumer_module


def _find_root_linear_consumer(
    model: nn.Module, conv: nn.Conv2d, out_channels: int
) -> Tuple[Optional[str], Optional[nn.Linear]]:
    """The head Linear, but only for the conv that actually feeds it.

    Divisibility alone is not evidence of a producer->consumer link: an earlier conv whose
    real consumer is the first conv of the next block would match just as well, and pruning
    on that guess rewires the head while leaving the true consumer untouched.
    """
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs or convs[-1] is not conv:
        return None, None
    candidates: List[Tuple[str, nn.Linear]] = []
    for name, child in iter_direct_named_modules(model):
        if isinstance(child, nn.Linear) and child.in_features % out_channels == 0:
            candidates.append((name, child))
    if len(candidates) == 1:
        return candidates[0]
    return None, None


def _is_consumer_compat(
    conv: nn.Conv2d, bn_module: Optional[nn.Module], consumer_module: nn.Module
) -> bool:
    if isinstance(consumer_module, nn.Conv2d):
        if consumer_module.groups != 1:
            return False
        if consumer_module.in_channels != conv.out_channels:
            return False
    elif isinstance(consumer_module, nn.Linear):
        if consumer_module.in_features % conv.out_channels != 0:
            return False
    if (
        isinstance(bn_module, nn.BatchNorm2d)
        and bn_module.num_features != conv.out_channels
    ):
        return False
    return True


def _make_target(
    model: nn.Module,
    parent_name: str,
    name: str,
    bn_name: Optional[str],
    consumer_name: str,
    consumer_from_root: bool = False,
) -> CNNChannelTarget:
    producer_path = f"{parent_name}.{name}" if parent_name else name
    bn_path = f"{parent_name}.{bn_name}" if parent_name and bn_name else bn_name
    # Resolve against the root only when the consumer actually came from there. Testing the
    # name against the root's children instead would resolve a sibling to a same-named
    # root module, pruning something that was never validated as the consumer.
    if consumer_from_root:
        consumer_path = consumer_name
    else:
        consumer_path = (
            f"{parent_name}.{consumer_name}" if parent_name else consumer_name
        )
    return CNNChannelTarget(
        producer_path=producer_path,
        bn_path=bn_path,
        consumer_path=consumer_path,
    )


def _build_target_for_conv(
    model: nn.Module,
    parent_name: str,
    parent: nn.Module,
    items: List[Tuple[str, nn.Module]],
    idx: int,
    name: str,
    mod: nn.Module,
    config: PruneConfig,
) -> Optional[CNNChannelTarget]:
    if (
        not isinstance(mod, nn.Conv2d)
        or mod.groups != 1
        or mod.out_channels <= config.min_channels
    ):
        return None
    bn_name, bn_module, consumer_name, consumer_module = _find_consumer(
        parent, items, idx
    )
    from_root = False
    if consumer_module is None or consumer_name is None:
        root_name, root_module = _find_root_linear_consumer(
            model, mod, mod.out_channels
        )
        if root_module is None or root_name is None:
            return None
        consumer_name, consumer_module = root_name, root_module
        from_root = True
    if not _is_consumer_compat(mod, bn_module, consumer_module):
        return None
    return _make_target(model, parent_name, name, bn_name, consumer_name, from_root)


class CNNPruningDomain(BasePruningDomain):
    name = "cnn"

    def find_targets(
        self, model: nn.Module, config: PruneConfig
    ) -> List[CNNChannelTarget]:
        targets: List[CNNChannelTarget] = []
        for parent_name, parent in list(model.named_modules()):
            items = iter_direct_named_modules(parent)
            if not items:
                continue
            for i, (name, mod) in enumerate(items):
                target = _build_target_for_conv(
                    model, parent_name, parent, items, i, name, mod, config
                )
                if target is not None:
                    targets.append(target)
        return targets

    def apply_keep_indices(
        self, model: nn.Module, target: CNNChannelTarget, keep_idx: List[int]
    ) -> None:
        session = simulate.active()
        if session is not None:
            session.record_cnn(model, target, keep_idx)
            return
        producer = get_submodule(model, target.producer_path)
        consumer = get_submodule(model, target.consumer_path)
        if not isinstance(producer, nn.Conv2d):
            raise TypeError(f"Target producer '{target.producer_path}' is not Conv2d.")
        if not isinstance(consumer, (nn.Conv2d, nn.Linear)):
            raise TypeError(
                f"Target consumer '{target.consumer_path}' must be Conv2d or Linear."
            )

        replace_submodule(
            model, target.producer_path, prune_conv2d_out_channels(producer, keep_idx)
        )

        if target.bn_path is not None:
            bn_module = get_submodule(model, target.bn_path)
            if isinstance(bn_module, nn.BatchNorm2d):
                replace_submodule(
                    model, target.bn_path, prune_batchnorm2d(bn_module, keep_idx)
                )

        if isinstance(consumer, nn.Conv2d):
            replace_submodule(
                model,
                target.consumer_path,
                prune_conv2d_in_channels(consumer, keep_idx),
            )
        else:
            flat_keep = expand_flatten_channel_indices(
                keep_idx,
                old_channels=producer.out_channels,
                linear_in_features=consumer.in_features,
            )
            replace_submodule(
                model,
                target.consumer_path,
                prune_linear_in_features(consumer, flat_keep),
            )

    def channel_count(self, model: nn.Module, target: CNNChannelTarget) -> int:
        producer = get_submodule(model, target.producer_path)
        if not isinstance(producer, nn.Conv2d):
            raise TypeError(f"Target producer '{target.producer_path}' is not Conv2d.")
        return int(producer.out_channels)

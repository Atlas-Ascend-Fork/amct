from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..report import PruneReport
from ..utils import (
    expand_flatten_channel_indices,
    iter_direct_named_modules,
    prune_batchnorm2d,
    prune_conv2d_in_channels,
    prune_conv2d_out_channels,
    prune_linear_in_features,
    replace_submodule,
    topk_keep_indices,
)
from .base import BasePruningStage


class CNNPruningStage(BasePruningStage):
    stage_name = "cnn"

    def _conv_score(self, conv: nn.Conv2d) -> torch.Tensor:
        return conv.weight.detach().abs().mean(dim=(1, 2, 3)).cpu()

    def _find_consumer(
        self, parent: nn.Module, items: List[Tuple[str, nn.Module]], start_idx: int
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
            if isinstance(mod, (nn.ReLU, nn.GELU, nn.SiLU, nn.Identity, nn.Dropout, nn.MaxPool2d, nn.AvgPool2d)):
                continue
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                consumer_name, consumer_module = name, mod
                break
            break
        return bn_name, bn_module, consumer_name, consumer_module

    def apply(self, model: nn.Module, report: PruneReport) -> None:
        ratio = self.config.cnn_ratio
        if ratio <= 0:
            return

        for parent_name, parent in list(model.named_modules()):
            items = iter_direct_named_modules(parent)
            if not items:
                continue

            for i, (name, mod) in enumerate(items):
                if not isinstance(mod, nn.Conv2d):
                    continue
                if mod.groups != 1:
                    continue
                if mod.out_channels <= self.config.min_channels:
                    continue

                bn_name, bn_module, consumer_name, consumer_module = self._find_consumer(parent, items, i)
                if consumer_module is None:
                    continue
                if isinstance(consumer_module, nn.Conv2d) and consumer_module.groups != 1:
                    continue
                if isinstance(consumer_module, nn.Conv2d) and consumer_module.in_channels != mod.out_channels:
                    continue
                if isinstance(consumer_module, nn.Linear) and consumer_module.in_features % mod.out_channels != 0:
                    continue
                if bn_module is not None and isinstance(bn_module, nn.BatchNorm2d) and bn_module.num_features != mod.out_channels:
                    continue

                keep_idx = topk_keep_indices(
                    self._conv_score(mod),
                    prune_ratio=ratio,
                    min_keep=self.config.min_channels,
                )
                if len(keep_idx) == mod.out_channels:
                    continue

                conv_path = f"{parent_name}.{name}" if parent_name else name
                new_conv = prune_conv2d_out_channels(mod, keep_idx)
                replace_submodule(model, conv_path, new_conv)

                if bn_name and isinstance(bn_module, nn.BatchNorm2d):
                    bn_path = f"{parent_name}.{bn_name}" if parent_name else bn_name
                    new_bn = prune_batchnorm2d(bn_module, keep_idx)
                    replace_submodule(model, bn_path, new_bn)

                consumer_path = f"{parent_name}.{consumer_name}" if parent_name else consumer_name
                if isinstance(consumer_module, nn.Conv2d):
                    new_consumer = prune_conv2d_in_channels(consumer_module, keep_idx)
                else:
                    flat_keep = expand_flatten_channel_indices(
                        keep_idx,
                        old_channels=mod.out_channels,
                        linear_in_features=consumer_module.in_features,
                    )
                    new_consumer = prune_linear_in_features(consumer_module, flat_keep)
                replace_submodule(model, consumer_path, new_consumer)

                report.add(
                    self.stage_name,
                    conv_path,
                    f"Conv out_channels {mod.out_channels} -> {len(keep_idx)}",
                )

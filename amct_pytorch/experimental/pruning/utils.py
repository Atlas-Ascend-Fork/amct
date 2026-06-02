from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


class RunningVariance:
    def __init__(self) -> None:
        self.count: int = 0
        self.sum: torch.Tensor | None = None
        self.sum_sq: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().reshape(-1, values.shape[-1]).to(dtype=torch.float32, device="cpu")
        if flat.numel() == 0:
            return
        current_sum = flat.sum(dim=0)
        current_sum_sq = (flat * flat).sum(dim=0)
        if self.sum is None:
            self.sum = current_sum
            self.sum_sq = current_sum_sq
        else:
            self.sum += current_sum
            self.sum_sq += current_sum_sq
        self.count += int(flat.shape[0])

    def variance(self) -> torch.Tensor:
        if self.count == 0 or self.sum is None or self.sum_sq is None:
            raise RuntimeError("RunningVariance has no samples.")
        mean = self.sum / float(self.count)
        var = self.sum_sq / float(self.count) - mean * mean
        return torch.clamp(var, min=0.0)


def clone_model_if_needed(model: nn.Module, copy_model: bool) -> nn.Module:
    return copy.deepcopy(model) if copy_model else model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def topk_keep_indices(scores: torch.Tensor, prune_ratio: float, min_keep: int) -> List[int]:
    if scores.ndim != 1:
        raise ValueError("scores must be 1D")
    total = scores.numel()
    keep = max(min_keep, int(round(total * (1.0 - prune_ratio))))
    keep = min(max(keep, 1), total)
    values, indices = torch.topk(scores, k=keep, largest=True, sorted=False)
    _ = values
    return sorted(indices.cpu().tolist())


def random_keep_indices(total: int, prune_ratio: float, min_keep: int, seed: int) -> List[int]:
    if total < 1:
        raise ValueError("total must be >= 1")
    keep = max(min_keep, int(round(total * (1.0 - prune_ratio))))
    keep = min(max(keep, 1), total)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(total, generator=generator)
    return sorted(perm[:keep].tolist())


def get_parent_and_name(model: nn.Module, module_path: str) -> Tuple[nn.Module, str]:
    parent_path, _, child_name = module_path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    return parent, child_name


def get_submodule(model: nn.Module, module_path: str) -> nn.Module:
    return model.get_submodule(module_path) if module_path else model


def replace_submodule(model: nn.Module, module_path: str, new_module: nn.Module) -> None:
    parent, child_name = get_parent_and_name(model, module_path)
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


def iter_direct_named_modules(module: nn.Module) -> List[Tuple[str, nn.Module]]:
    return list(module.named_children())


def is_activation_like(module: nn.Module) -> bool:
    activation_types = (
        nn.ReLU,
        nn.GELU,
        nn.SiLU,
        nn.Sigmoid,
        nn.Tanh,
        nn.Dropout,
        nn.Identity,
        nn.LeakyReLU,
    )
    return isinstance(module, activation_types)


def is_norm_like(module: nn.Module) -> bool:
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.Identity,
    )
    return isinstance(module, norm_types)


def prune_conv2d_out_channels(module: nn.Conv2d, keep_idx: Sequence[int]) -> nn.Conv2d:
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=module.weight.device)
    new = nn.Conv2d(
        in_channels=module.in_channels,
        out_channels=len(keep_idx),
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(module.weight.index_select(0, keep))
        if module.bias is not None:
            new.bias.copy_(module.bias.index_select(0, keep))
    return new


def prune_conv2d_in_channels(module: nn.Conv2d, keep_idx: Sequence[int]) -> nn.Conv2d:
    if module.groups != 1:
        raise ValueError("Grouped/depthwise convolution input pruning is not supported in this baseline.")
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=module.weight.device)
    new = nn.Conv2d(
        in_channels=len(keep_idx),
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(module.weight.index_select(1, keep))
        if module.bias is not None:
            new.bias.copy_(module.bias)
    return new


def prune_batchnorm2d(module: nn.BatchNorm2d, keep_idx: Sequence[int]) -> nn.BatchNorm2d:
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=module.weight.device)
    new = nn.BatchNorm2d(
        num_features=len(keep_idx),
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        if module.affine:
            new.weight.copy_(module.weight.index_select(0, keep))
            new.bias.copy_(module.bias.index_select(0, keep))
        if module.track_running_stats:
            new.running_mean.copy_(module.running_mean.index_select(0, keep))
            new.running_var.copy_(module.running_var.index_select(0, keep))
            new.num_batches_tracked.copy_(module.num_batches_tracked)
    return new


def prune_linear_out_features(module: nn.Linear, keep_idx: Sequence[int]) -> nn.Linear:
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=module.weight.device)
    new = nn.Linear(
        in_features=module.in_features,
        out_features=len(keep_idx),
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(module.weight.index_select(0, keep))
        if module.bias is not None:
            new.bias.copy_(module.bias.index_select(0, keep))
    return new


def prune_linear_in_features(module: nn.Linear, keep_idx: Sequence[int]) -> nn.Linear:
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=module.weight.device)
    new = nn.Linear(
        in_features=len(keep_idx),
        out_features=module.out_features,
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(module.weight.index_select(1, keep))
        if module.bias is not None:
            new.bias.copy_(module.bias)
    return new


def expand_flatten_channel_indices(channel_idx: Sequence[int], old_channels: int, linear_in_features: int) -> List[int]:
    if linear_in_features % old_channels != 0:
        raise ValueError(
            f"Linear input features ({linear_in_features}) are not divisible by old_channels ({old_channels})."
        )
    span = linear_in_features // old_channels
    expanded: List[int] = []
    for idx in channel_idx:
        start = idx * span
        expanded.extend(range(start, start + span))
    return expanded


def maybe_to_dict(obj: object) -> object:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def infer_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_batch_to_device(args: Tuple[object, ...], kwargs: Dict[str, object], device: torch.device) -> Tuple[Tuple[object, ...], Dict[str, object]]:
    def move(value: object) -> object:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(move(x) for x in value)
        if isinstance(value, list):
            return [move(x) for x in value]
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value

    return tuple(move(arg) for arg in args), {k: move(v) for k, v in kwargs.items()}


def merge_module_parameters_(target: nn.Module, source: nn.Module, target_weight: float, source_weight: float) -> None:
    if target.state_dict().keys() != source.state_dict().keys():
        raise ValueError("Cannot merge experts with different state dict keys.")
    total = float(target_weight + source_weight)
    if total <= 0:
        total = 1.0
    target_state = target.state_dict()
    source_state = source.state_dict()
    merged: Dict[str, torch.Tensor] = {}
    for key, target_tensor in target_state.items():
        source_tensor = source_state[key]
        if target_tensor.shape != source_tensor.shape:
            raise ValueError(f"Cannot merge experts with mismatched parameter shapes for key '{key}'.")
        if torch.is_floating_point(target_tensor):
            merged[key] = (
                target_tensor.to(dtype=torch.float32) * float(target_weight)
                + source_tensor.to(dtype=torch.float32) * float(source_weight)
            ) / total
            merged[key] = merged[key].to(dtype=target_tensor.dtype)
        else:
            merged[key] = target_tensor.clone()
    target.load_state_dict(merged)

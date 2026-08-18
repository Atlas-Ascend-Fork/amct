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

import copy
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from ..common.utils.model_util import ModuleHelper


class RunningVariance:
    def __init__(self) -> None:
        self.count: int = 0
        self.sum: torch.Tensor | None = None
        self.sum_sq: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        flat = (
            values.detach()
            .reshape(-1, values.shape[-1])
            .to(dtype=torch.float32, device="cpu")
        )
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


def topk_keep_indices(
    scores: torch.Tensor, prune_ratio: float, min_keep: int
) -> List[int]:
    if scores.ndim != 1:
        raise ValueError("scores must be 1D")
    total = scores.numel()
    keep = max(min_keep, int(round(total * (1.0 - prune_ratio))))
    keep = min(max(keep, 1), total)
    values, indices = torch.topk(scores, k=keep, largest=True, sorted=False)
    _ = values
    return sorted(indices.cpu().tolist())


def get_submodule(model: nn.Module, module_path: str) -> nn.Module:
    if not module_path:
        return model
    return ModuleHelper(model).named_module_dict[module_path]


def replace_submodule(
    model: nn.Module, module_path: str, new_module: nn.Module
) -> None:
    ModuleHelper.replace_module_by_name(model, module_path, new_module)


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


def _carry_requires_grad(new: nn.Module, old: nn.Module) -> nn.Module:
    """Copy each parameter's requires_grad from the module being replaced.

    Fresh parameters are born requires_grad=True and ``copy_`` moves values only, so
    rebuilding a frozen layer silently unfroze it -- a user pruning mid-training would
    find their frozen layers training again.
    """
    old_flags = {name: param.requires_grad for name, param in old.named_parameters()}
    for name, param in new.named_parameters():
        if name in old_flags:
            param.requires_grad_(old_flags[name])
    return new


def _built_without_rng(factory, device, dtype) -> nn.Module:
    """Construct a module whose init never touches an RNG.

    Every constructor here immediately overwrites its weights with slices of the old
    module, but a plain construction still runs the default init, which draws from the
    global generator -- so pruning perturbed the caller's random stream by an amount
    that depended on layer sizes. Building on the meta device skips init entirely.
    """
    with torch.device("meta"):
        module = factory()
    if any(True for _ in module.buffers()):
        # to_empty would leave every buffer as uninitialized garbage and the callers
        # only copy weight/bias. Rebuild for real, shielding the caller's RNG stream
        # from the constructor's init draws instead.
        with torch.random.fork_rng(devices=[]):
            module = factory()
        return module.to(device=device, dtype=dtype)
    module = module.to_empty(device=device)
    if dtype is not None:
        module = module.to(dtype)
    return module


def prune_conv2d_out_channels(module: nn.Conv2d, keep_idx: Sequence[int]) -> nn.Conv2d:
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=module.weight.device)
    new = _built_without_rng(
        lambda: nn.Conv2d(
            in_channels=module.in_channels,
            out_channels=len(keep_idx),
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        ),
        module.weight.device,
        module.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(module.weight.index_select(0, keep))
        if module.bias is not None:
            new.bias.copy_(module.bias.index_select(0, keep))
    return _carry_requires_grad(new, module)


def prune_conv2d_in_channels(module: nn.Conv2d, keep_idx: Sequence[int]) -> nn.Conv2d:
    if module.groups != 1:
        raise ValueError(
            "Grouped/depthwise convolution input pruning is not supported in this baseline."
        )
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=module.weight.device)
    new = _built_without_rng(
        lambda: nn.Conv2d(
            in_channels=len(keep_idx),
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        ),
        module.weight.device,
        module.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(module.weight.index_select(1, keep))
        if module.bias is not None:
            new.bias.copy_(module.bias)
    return _carry_requires_grad(new, module)


def prune_batchnorm2d(
    module: nn.BatchNorm2d, keep_idx: Sequence[int]
) -> nn.BatchNorm2d:
    # weight is None when affine=False, so take device/dtype from whatever tensor exists.
    ref = module.weight if module.weight is not None else module.running_mean
    if ref is None:
        ref = torch.empty(0)
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=ref.device)
    new = nn.BatchNorm2d(
        num_features=len(keep_idx),
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats,
        device=ref.device,
        dtype=ref.dtype,
    )
    with torch.no_grad():
        if module.affine:
            new.weight.copy_(module.weight.index_select(0, keep))
            new.bias.copy_(module.bias.index_select(0, keep))
        if module.track_running_stats:
            new.running_mean.copy_(module.running_mean.index_select(0, keep))
            new.running_var.copy_(module.running_var.index_select(0, keep))
            new.num_batches_tracked.copy_(module.num_batches_tracked)
    # A fresh BatchNorm2d starts in training mode. Without this an eval model comes back
    # from pruning normalising by batch statistics, which changes its outputs and quietly
    # overwrites the running stats we just copied.
    new.train(module.training)
    return _carry_requires_grad(new, module)


def prune_linear_out_features(module: nn.Linear, keep_idx: Sequence[int]) -> nn.Linear:
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=module.weight.device)
    new = _built_without_rng(
        lambda: nn.Linear(
            in_features=module.in_features,
            out_features=len(keep_idx),
            bias=module.bias is not None,
        ),
        module.weight.device,
        module.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(module.weight.index_select(0, keep))
        if module.bias is not None:
            new.bias.copy_(module.bias.index_select(0, keep))
    return _carry_requires_grad(new, module)


def prune_linear_in_features(module: nn.Linear, keep_idx: Sequence[int]) -> nn.Linear:
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=module.weight.device)
    new = _built_without_rng(
        lambda: nn.Linear(
            in_features=len(keep_idx),
            out_features=module.out_features,
            bias=module.bias is not None,
        ),
        module.weight.device,
        module.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(module.weight.index_select(1, keep))
        if module.bias is not None:
            new.bias.copy_(module.bias)
    return _carry_requires_grad(new, module)


def is_conv1d_like(module: nn.Module) -> bool:
    """Duck-typed detection of GPT-2 ``Conv1D`` (a transposed-weight linear layer; ``nf`` holds the output dim)."""
    return (
        type(module).__name__ == "Conv1D"
        and hasattr(module, "nf")
        and hasattr(module, "weight")
        and getattr(module.weight, "dim", lambda: 0)() == 2
    )


def is_linear_like(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or is_conv1d_like(module)


def linear_like_in_features(module: nn.Module) -> int:
    return (
        int(module.in_features)
        if isinstance(module, nn.Linear)
        else int(module.weight.shape[0])
    )


def linear_like_out_features(module: nn.Module) -> int:
    return int(module.out_features) if isinstance(module, nn.Linear) else int(module.nf)


def linear_like_weight_oi(module: nn.Module):
    """Weight view in [out, in] layout (Linear as-is, Conv1D transposed)."""
    return module.weight if isinstance(module, nn.Linear) else module.weight.t()


def prune_conv1d_out_features(module: nn.Module, keep_idx: Sequence[int]) -> nn.Module:
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=module.weight.device)
    new = _built_without_rng(
        lambda: type(module)(len(keep_idx), int(module.weight.shape[0])),
        module.weight.device,
        module.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(module.weight.index_select(1, keep))
        new.bias.copy_(module.bias.index_select(0, keep))
    return _carry_requires_grad(new, module)


def prune_conv1d_in_features(module: nn.Module, keep_idx: Sequence[int]) -> nn.Module:
    keep = torch.tensor(list(keep_idx), dtype=torch.long, device=module.weight.device)
    new = _built_without_rng(
        lambda: type(module)(int(module.nf), len(keep_idx)),
        module.weight.device,
        module.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(module.weight.index_select(0, keep))
        new.bias.copy_(module.bias)
    return _carry_requires_grad(new, module)


def prune_linear_like_out_features(
    module: nn.Module, keep_idx: Sequence[int]
) -> nn.Module:
    if isinstance(module, nn.Linear):
        return prune_linear_out_features(module, keep_idx)
    return prune_conv1d_out_features(module, keep_idx)


def prune_linear_like_in_features(
    module: nn.Module, keep_idx: Sequence[int]
) -> nn.Module:
    if isinstance(module, nn.Linear):
        return prune_linear_in_features(module, keep_idx)
    return prune_conv1d_in_features(module, keep_idx)


def expand_flatten_channel_indices(
    channel_idx: Sequence[int], old_channels: int, linear_in_features: int
) -> List[int]:
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


def infer_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_batch_to_device(
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
    device: torch.device,
) -> Tuple[Tuple[object, ...], Dict[str, object]]:
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


@torch.no_grad()
def run_calibration_epoch(model: nn.Module, context, device=None) -> int:
    """Run one calibration forward pass (eval + no_grad) to drive hooks/stats; returns batch count."""
    if device is None:
        device = infer_model_device(model)
    was_training = model.training
    model.eval()
    num_batches = 0
    try:
        for args, kwargs in context.iter_model_inputs():
            moved_args, moved_kwargs = move_batch_to_device(args, kwargs, device)
            model(*moved_args, **moved_kwargs)
            num_batches += 1
    finally:
        model.train(was_training)
    return num_batches


def record_prune_width(model: nn.Module, key: str, width: int) -> None:
    """Accumulate a layer's post-prune width into model meta (read by compat.patch_common_config)."""
    meta = getattr(model, "_amct_prune_meta", {})
    meta.setdefault(key, set()).add(int(width))
    setattr(model, "_amct_prune_meta", meta)


def record_prune_size(model: nn.Module, key: str, value: int) -> None:
    """Record the uniform post-prune size scalar into model meta (read by compat.patch_common_config)."""
    meta = getattr(model, "_amct_prune_meta", {})
    meta[key] = int(value)
    setattr(model, "_amct_prune_meta", meta)


def solve_least_squares(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve gram @ x = rhs on the current device, falling back to CPU if it lacks a solver."""
    try:
        return torch.linalg.solve(gram, rhs)
    except RuntimeError:
        return torch.linalg.solve(gram.cpu(), rhs.cpu()).to(gram.device)


def ridge_solve(gram: torch.Tensor, rhs: torch.Tensor, ridge: float) -> torch.Tensor:
    """Add a ridge to gram's diagonal (in place) and solve (with CPU fallback)."""
    gram.diagonal().add_(ridge * gram.diagonal().mean())
    return solve_least_squares(gram, rhs)

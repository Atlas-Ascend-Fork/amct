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
"""输出保留（重构）的 CNN 通道剪枝 —— 依赖图选通道 + 最小二乘补偿，低损失、无需微调。

复用依赖图(DepGraph)决定 *哪些* 通道一起剪（残差/下采样/flatten→linear 等耦合层组），
但在剪掉后，对每个 **consumer**（输入通道被裁掉的 conv/linear）做最小二乘**重构其权重**，
把被移除通道的贡献吸收进保留通道，使该层输出尽量不变（ThiNet/CP 式重构）：

    Y = im2col(X) @ W_flat^T              (X: consumer 输入, Y: 原输出)
    W_kept = argmin || Y[:, keep_out] - im2col(X[:, keep_in]) @ W^T || + ridge

与 ``depgraph`` 同样基于内置(static) Torch-Pruning，无外部依赖；区别仅在于剪枝后多了一步重构补偿。
要求：校准集**有代表性且多样**，im2col 行数（N·H·W）应远多于 keep_in·K·K，否则重构会过拟合。
kwargs 与 ``depgraph`` 相同，外加 ``ridge``（默认 1e-2）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..report import PruneReport
from ..utils import infer_model_device, move_batch_to_device
from .base import BasePruningMethod
from .cnn_depgraph import (
    _auto_ignored_layers,
    _example_inputs,
    _populate_gradients,
    _select_importance,
    _GRAD_BASED_IMPORTANCE,
    _PERSAMPLE_GRAD_IMPORTANCE,
)


def _complement(removed: List[int], total: int) -> List[int]:
    rm = set(int(i) for i in removed)
    return [i for i in range(total) if i not in rm]


@torch.no_grad()
def _reconstruct_consumer(
    module: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    keep_in: List[int],
    keep_out: List[int],
    ridge: float,
) -> bool:
    """用剪枝前激活 (x->module->y) 重构 module 在 keep_in/keep_out 上的权重。成功返回 True。"""
    keep_in_t = torch.as_tensor(keep_in)
    keep_out_t = torch.as_tensor(keep_out)
    if isinstance(module, nn.Conv2d):
        k = module.kernel_size
        # im2col 原输入的保留通道： [rows, keep_in*kh*kw]
        cols = F.unfold(
            x.float(),
            kernel_size=k,
            padding=module.padding,
            stride=module.stride,
            dilation=module.dilation,
        )  # [N, Cin*kh*kw, L]
        n, ckk, length = cols.shape
        cin = x.shape[1]
        cols = cols.reshape(n, cin, k[0] * k[1], length)
        cols = cols[:, keep_in_t, :, :].reshape(n, len(keep_in) * k[0] * k[1], length)
        p = cols.permute(0, 2, 1).reshape(n * length, -1)  # [rows, keep_in*kh*kw]
        target = y.float().permute(0, 2, 3, 1).reshape(-1, y.shape[1])[:, keep_out_t]
    elif isinstance(module, nn.Linear):
        p = x.float().reshape(-1, x.shape[-1])[:, keep_in_t]
        target = y.float().reshape(-1, y.shape[-1])[:, keep_out_t]
    else:
        return False
    if module.bias is not None:
        # 重构目标为 *去偏置* 的线性部分（剪后的 module 仍会加上其(已切片的)偏置）
        bias_full = module.bias.detach().float()
        # 注意：此时 module.bias 已被结构化剪枝切到 keep_out；用原偏置的 keep_out 切片
        target = target - bias_full[: len(keep_out)].to(target.device)
    rows, cols_n = p.shape
    if rows < cols_n:
        return False  # 欠定：行数不足，放弃重构（保留 depgraph 的朴素切片结果）
    a = p.t() @ p
    a.diagonal().add_(ridge * a.diagonal().mean())
    w = torch.linalg.solve(a, p.t() @ target)  # [keep_in*khkw, keep_out]
    if isinstance(module, nn.Conv2d):
        new_w = w.t().reshape(len(keep_out), len(keep_in), k[0], k[1])
    else:
        new_w = w.t()
    if new_w.shape != tuple(module.weight.shape):
        return False
    module.weight.data = new_w.to(module.weight.dtype)
    return True


class ReconstructChannelPruningMethod(BasePruningMethod):
    """依赖图选通道 + 最小二乘重构补偿的输出保留 CNN 通道剪枝（低损失，无需微调）。

    kwargs：同 ``depgraph``（prune_ratio/importance/global_pruning/round_to/iterative_steps），
    外加 ridge（默认 1e-2）。重构在最后一步剪枝后对所有 consumer 执行。
    """

    domain = "cnn"
    name = "reconstruct"
    requires_data = True
    requires_targets = False

    def apply(
        self,
        model: nn.Module,
        domain: BasePruningDomain,
        targets: List[Any],
        context: PruneContext,
        report: PruneReport,
        config: PruneConfig,
        spec: MethodSpec,
    ) -> None:
        from .._vendor import torch_pruning as tp

        prune_ratio = float(spec.kwargs.get("prune_ratio", 0.30))
        if prune_ratio <= 0:
            return
        importance_name = str(spec.kwargs.get("importance", "magnitude")).lower()
        global_pruning = bool(spec.kwargs.get("global_pruning", False))
        round_to = spec.kwargs.get("round_to", None)
        iterative_steps = int(spec.kwargs.get("iterative_steps", 1))
        ridge = float(spec.kwargs.get("ridge", 1e-2))
        if iterative_steps < 1:
            raise ValueError("iterative_steps must be >= 1.")

        device = infer_model_device(model)
        example_inputs = _example_inputs(context, device)
        if example_inputs is None:
            raise ValueError("Reconstruct CNN pruning requires calibration data.")
        ignored = _auto_ignored_layers(model, config.skip_layers)
        importance = _select_importance(importance_name)
        is_grad = importance_name in _GRAD_BASED_IMPORTANCE
        per_sample = importance_name in _PERSAMPLE_GRAD_IMPORTANCE

        was_training = model.training
        model.eval()
        n_recon = 0
        try:
            pruner = tp.pruner.MetaPruner(
                model,
                example_inputs,
                importance=importance,
                pruning_ratio=prune_ratio,
                ignored_layers=ignored,
                global_pruning=global_pruning,
                round_to=round_to,
                iterative_steps=iterative_steps,
            )
            for step_i in range(iterative_steps):
                if is_grad:
                    ok = _populate_gradients(
                        model,
                        context,
                        device,
                        importance=importance,
                        per_sample=per_sample,
                    )
                    if not ok:
                        raise ValueError(
                            f"importance='{importance_name}' needs gradients but none were produced."
                        )
                # 收集本步的剪枝组（interactive：先不应用）
                groups = list(pruner.step(interactive=True))
                # 解析每个 module 被移除的 in / out 通道
                removed_in: Dict[int, List[int]] = {}
                removed_out: Dict[int, List[int]] = {}
                mod_by_id: Dict[int, nn.Module] = {}
                for g in groups:
                    for item in g.items:
                        m = item.dep.target.module
                        hname = getattr(item.dep.handler, "__name__", "")
                        if not isinstance(m, (nn.Conv2d, nn.Linear)):
                            continue
                        mod_by_id[id(m)] = m
                        if "in_channels" in hname or "in_features" in hname:
                            removed_in.setdefault(id(m), []).extend(
                                int(i) for i in item.idxs
                            )
                        elif "out_channels" in hname or "out_features" in hname:
                            removed_out.setdefault(id(m), []).extend(
                                int(i) for i in item.idxs
                            )

                # 只对“输入通道被裁”的 consumer 做重构；记录剪枝前 in/out 维与激活
                consumers = [
                    mid
                    for mid in removed_in
                    if isinstance(mod_by_id[mid], (nn.Conv2d, nn.Linear))
                ]
                orig_dims: Dict[int, Tuple[int, int]] = {}
                acts: Dict[int, Dict[str, torch.Tensor]] = {}
                handles = []
                for mid in consumers:
                    m = mod_by_id[mid]
                    if isinstance(m, nn.Conv2d):
                        orig_dims[mid] = (m.in_channels, m.out_channels)
                    else:
                        orig_dims[mid] = (m.in_features, m.out_features)
                    acts[mid] = {}

                    def hook(mod, inp, out, key=mid):
                        acts[key]["x"] = inp[0].detach()
                        acts[key]["y"] = out.detach()

                    handles.append(m.register_forward_hook(hook))
                # 一次校准前向（模型仍是剪枝前），抓取 consumer 的输入/输出
                if consumers:
                    first = next(iter(context.iter_model_inputs()), None)
                    if first is not None:
                        args, kwargs = move_batch_to_device(first[0], first[1], device)
                        with torch.no_grad():
                            model(*args, **kwargs)
                for h in handles:
                    h.remove()

                # 应用结构化剪枝
                for g in groups:
                    g.prune()

                # 逐 consumer 重构（最后一步后才补偿，避免被后续步覆盖）
                if step_i == iterative_steps - 1:
                    for mid in consumers:
                        m = mod_by_id[mid]
                        if "x" not in acts[mid]:
                            continue
                        in0, out0 = orig_dims[mid]
                        keep_in = _complement(removed_in.get(mid, []), in0)
                        keep_out = _complement(removed_out.get(mid, []), out0)
                        if not keep_in or not keep_out:
                            continue
                        if _reconstruct_consumer(
                            m, acts[mid]["x"], acts[mid]["y"], keep_in, keep_out, ridge
                        ):
                            n_recon += 1
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        opts = []
        if global_pruning:
            opts.append("global")
        if round_to:
            opts.append(f"round_to={round_to}")
        if iterative_steps > 1:
            opts.append(f"iter={iterative_steps}")
        opt_note = (", " + ",".join(opts)) if opts else ""
        report.add(
            self.domain,
            self.name,
            "<depgraph+reconstruct>",
            f"output-preserving channel prune ratio={prune_ratio}, importance={importance_name}, "
            f"reconstructed_consumers={n_recon}{opt_note}",
        )

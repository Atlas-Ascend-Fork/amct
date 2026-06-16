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

import torch
import torch.nn as nn
from loguru import logger

from amct_pytorch.common.optimization import SOLVER_REGISTRY
from amct_pytorch.common.optimization.factory import build_lr_scheduler, build_optimizer, set_require_grad_all
from amct_pytorch.quantization.modules.quant_base import WeightQuantizer, ActivationQuantizer
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

from .base_solver import BaseSolver


@SOLVER_REGISTRY.register(name="block", description="Blockwise calibration solver")
class BlockwiseSolver(BaseSolver):
    granularity = "block"

    def __init__(
            self,
            args,
            layer_idx: int,
            model: nn.Module,
            block_size: int = 128,
            max_iters: int = 100,
    ):
        super().__init__(args, layer_idx, model, None, None, max_iters)
        self.block_size = block_size

    def solve(self, data_loader, forward_kwargs=None):
        if self.optimizer is None:
            self.model = self.model.to(self.args.device)
            param_groups = self._collect_trainable_param_groups(self.model)
            if not param_groups:
                return self.model
            self.optimizer = build_optimizer(self.args, param_groups)
            self.lr_scheduler = build_lr_scheduler(self.args, self.optimizer)
        for epoch in range(self.args.epochs):
            avg_loss = self._optimize_block(data_loader, forward_kwargs=forward_kwargs)
            logger.info(
                "Layer {} epoch {}/{} avg_loss={:.6f}",
                self.layer_idx,
                epoch + 1,
                self.args.epochs,
                avg_loss,
            )
        return None

    def _optimize_block(self, data_loader, forward_kwargs=None):
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been initialized. Call solve() first.")

        kwargs = forward_kwargs or {}
        total_loss = 0.0
        num_batches = 0
        for data_batch in data_loader:
            if not isinstance(data_batch, (tuple, list)) or len(data_batch) != 2:
                raise ValueError("Expected PTQ dataloader to yield (inputs, targets).")
            unit_inp, unit_gt = data_batch
            self.optimizer.zero_grad()

            # [DBG] quantizer enable status on first batch of each layer
            if num_batches == 0:
                wq_enabled = sum(
                    1 for m in self.model.modules()
                    if isinstance(m, WeightQuantizer) and m.enable
                )
                wq_total = sum(
                    1 for m in self.model.modules()
                    if isinstance(m, WeightQuantizer)
                )
                aq_enabled = sum(
                    1 for m in self.model.modules()
                    if isinstance(m, ActivationQuantizer) and m.enable
                )
                aq_total = sum(
                    1 for m in self.model.modules()
                    if isinstance(m, ActivationQuantizer)
                )
                logger.info(
                    "[DBG] L{} batch0 WQ enabled={}/{} AQ enabled={}/{}",
                    self.layer_idx, wq_enabled, wq_total, aq_enabled, aq_total,
                )

            outputs = self.model(unit_inp, **kwargs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = self._reconstruction_loss(outputs, unit_gt)
            total_loss += loss.detach().cpu()

            # [DBG] output/gt stats + first QuantLinear weight quant error
            if num_batches == 0:
                logger.info(
                    "[DBG] L{} batch0 out_mean={:.6e} gt_mean={:.6e} loss={:.6e}",
                    self.layer_idx,
                    outputs.abs().mean().item(),
                    unit_gt.abs().mean().item(),
                    loss.item(),
                )
                with torch.no_grad():
                    for name, m in self.model.named_modules():
                        if not isinstance(m, QuantLinear):
                            continue
                        if m.weight_quantizer.enable and m.linear is not None:
                            float_w = m.linear.weight
                            q_w = m.weight_quantizer(float_w)
                            w_diff = (q_w - float_w).abs().mean().item()
                            w_rel = w_diff / (float_w.abs().mean().item() + 1e-12)
                            logger.info(
                                "[DBG] L{} batch0 {} w_bits={} w_mean={:.6e} rel_err={:.6f}",
                                self.layer_idx, name, m.w_bits,
                                float_w.abs().mean().item(), w_rel,
                            )
                            break

            loss = loss / loss.clone().detach()
            loss.backward()
            self.step()
            num_batches += 1

        if num_batches == 0:
            return 0.0
        return total_loss

    def _collect_trainable_param_groups(self, layer: nn.Module):
        set_require_grad_all(layer, False)
        seen = set()
        trainable_params = []

        for module in layer.modules():
            module_params_fn = getattr(module, "trainable_params", None)
            if not callable(module_params_fn):
                continue
            module_params = module_params_fn()
            for param in module_params:
                if param is None or id(param) in seen:
                    continue
                param.requires_grad = True
                seen.add(id(param))
                trainable_params.append(param)

        if not trainable_params:
            return []
        return [{"params": trainable_params, "lr": self.args.base_lr * 10}]

    def _reconstruction_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(output, target)

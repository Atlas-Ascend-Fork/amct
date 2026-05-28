import torch
import torch.nn as nn
from loguru import logger

from amct_pytorch.common.optimization import SOLVER_REGISTRY
from amct_pytorch.common.optimization.factory import build_lr_scheduler, build_optimizer, set_require_grad_all

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
            outputs = self.model(unit_inp, **kwargs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = self._reconstruction_loss(outputs, unit_gt)
            total_loss += loss.detach().cpu()
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

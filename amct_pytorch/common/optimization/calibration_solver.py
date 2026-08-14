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
from loguru import logger

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler
from amct_pytorch.common.models.llm.common.quant_apply import (
    set_model_act_quant_state, set_model_weight_quant_state)
from amct_pytorch.common.optimization import CalibrateBuildContext, SOLVER_REGISTRY
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

from .base_solver import BaseSolver


@SOLVER_REGISTRY.register(name="calibration", description="Single-pass (no-gradient) calibration solver")
class CalibrationSolver(BaseSolver):
    """Orchestrates no-gradient, single-pass-over-calibration-data PTQ algorithms.

    Unlike BlockwiseSolver (which drives ALGO_REGISTRY "weight"/"activation"/"structure"
    plugins composed inline into QuantLinear/ActivationQuantizer's forward), algorithms in
    this category (solver="calibrate", e.g. gptq -- see ALGO_REGISTRY metadata) declare no
    `targets` at all and are looked up directly from args.algos. LlmPtqWorkflow's solver
    routing already guarantees every algo in args.algos shares one solver kind by the time
    this runs, so args.algos itself is the calibration algo list here.

    Any solver="calibrate" algorithm is driven the same generic way, with no per-algorithm
    knowledge in this class: its registered ALGO_REGISTRY target must be a callable
    build(quant_linear, layer_name, ctx) -> algo (ctx: CalibrateBuildContext), where `algo`
    exposes .forward(inputs), called once per calibration batch with that layer's captured
    float activations. See amct_pytorch/algorithms/quant/gptq.py::build_gptq for the
    reference implementation. Adding a new calibration-only algorithm requires no change
    here -- just register a conforming target with solver="calibrate".
    """

    granularity = "block"

    def __init__(self, args, layer_idx, model, max_iters=100):
        super().__init__(args, layer_idx, model, None, None, max_iters)

    @torch.no_grad()
    def solve(self, data_loader, forward_kwargs=None):
        kwargs = forward_kwargs or {}
        self.model = self.model.to(self.args.device)

        algo_names = list(self.args.algos)
        if len(algo_names) != 1:
            raise ValueError(
                f"CalibrationSolver supports exactly one calibration algo per run, got {algo_names}"
            )
        algo_name = algo_names[0]

        quant_linears = {
            name: module for name, module in self.model.named_modules()
            if isinstance(module, QuantLinear)
        }
        if not quant_linears:
            return None

        num_batches = len(data_loader)
        if num_batches == 0:
            return None

        self._run(algo_name, quant_linears, data_loader, kwargs, num_batches)
        return None

    def _run(self, algo_name, quant_linears, data_loader, kwargs, num_batches):
        build = ALGO_REGISTRY.get(algo_name)  # e.g. build_gptq
        ctx = CalibrateBuildContext(args=self.args, num_batches=num_batches)
        wrapped = {
            name: build(module, name, ctx)
            for name, module in quant_linears.items()
        }

        captured = {}

        def make_hook(name):
            def hook(_module, inputs):
                captured[name] = inputs[0].detach()
            return hook

        handles = [
            module.register_forward_pre_hook(make_hook(name))
            for name, module in quant_linears.items()
        ]

        # Hessians must be accumulated against genuine float activations (matching
        # classic GPTQ), not against whatever this unit's own weight quantizer would
        # currently produce -- mirrors _prepare_unit_batch's materialize_gt disabling
        # the same two states for the same reason.
        set_model_weight_quant_state(self.model, False)
        set_model_act_quant_state(self.model, False)
        try:
            for data_batch in data_loader:
                unit_inp = data_batch[0] if isinstance(data_batch, (tuple, list)) else data_batch
                unit_inp = unit_inp.to(self.args.device)
                captured.clear()
                self.model(unit_inp, **kwargs)
                for name, algo in wrapped.items():
                    if name not in captured:
                        raise RuntimeError(
                            f"CalibrationSolver: layer '{name}' was not reached during the "
                            "calibration forward pass; cannot accumulate its statistics."
                        )
                    # Drives the algorithm's own calibrate/optimize state machine via
                    # forward() -- CalibrationSolver has no knowledge of what happens inside
                    # (e.g. GPTQuant/HiGPTQuant accumulate a Hessian on every call and run
                    # their Cholesky/OBS weight update on the batch_num-th call).
                    algo.forward(captured[name].to(self.args.device))
        finally:
            for handle in handles:
                handle.remove()
            set_model_weight_quant_state(self.model, True)
            set_model_act_quant_state(self.model, True)

        logger.info("Layer {}: {} finished for {} linear(s).", self.layer_idx, algo_name, len(wrapped))

    def finalize(self):
        return PtqParamHandler.export_module(self.model)

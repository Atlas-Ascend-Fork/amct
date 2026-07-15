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
from amct_pytorch.common.optimization import SOLVER_REGISTRY
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

from .base_solver import BaseSolver

# HiGPTQ is the only GPTQ-family algo wired up so far; the quant_config passed to
# HiGPTQuant is fixed accordingly (hifloat4 has exactly one legal symmetric/group_size
# combination -- see amct_pytorch/common/utils/vars.py's WTS_ASYMMETRIC_DTYPE /
# GROUP_SIZE_SUPPORTED_DTYPE). Widening this to plain GPTQuant for int/mxfp targets
# would need this to read weight dtype/strategy from args.bit_policy instead.
_HIFLOAT4_WEIGHTS_CFG = {'quant_type': 'hifloat4', 'symmetric': True, 'strategy': 'channel'}


@SOLVER_REGISTRY.register(name="calibration", description="Single-pass (no-gradient) calibration solver")
class CalibrationSolver(BaseSolver):
    """Orchestrates no-gradient, single-pass-over-calibration-data PTQ algorithms.

    Unlike BlockwiseSolver (which drives ALGO_REGISTRY "weight"/"activation"/"structure"
    plugins composed inline into QuantLinear/ActivationQuantizer's forward), algorithms in
    this category (solver="calibrate", e.g. higptq -- see ALGO_REGISTRY metadata) declare no
    `targets` at all and are looked up directly from args.algos, then driven by name -- each
    one gets its own dispatch branch below, since their construction/call conventions differ
    from each other (and from the weight/activation/structure plugin shape). LlmPtqWorkflow's
    solver routing already guarantees every algo in args.algos shares one solver kind by the
    time this runs, so args.algos itself is the calibration algo list here. Add a new branch
    below when a new calibration-only algorithm is registered.
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

        if algo_name == "higptq":
            self._run_higptq(algo_name, quant_linears, data_loader, kwargs, num_batches)
        else:
            raise ValueError(f"CalibrationSolver: no dispatch branch for algo '{algo_name}'")
        return None

    def _run_higptq(self, algo_name, quant_linears, data_loader, kwargs, num_batches):
        algo_cls = ALGO_REGISTRY.get(algo_name)  # HiGPTQuant
        quant_config = {
            'batch_num': num_batches,
            'weights_cfg': dict(_HIFLOAT4_WEIGHTS_CFG),
            'inputs_cfg': {'enable_quant': False},
        }
        wrapped = {
            name: algo_cls(module.linear, name, quant_config)
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
                for name, higptq in wrapped.items():
                    if name not in captured:
                        raise RuntimeError(
                            f"CalibrationSolver: layer '{name}' was not reached during the "
                            "calibration forward pass; cannot accumulate its Hessian."
                        )
                    # Drives HiGPTQuant's own calibrate/optimize state machine (forward()):
                    # accumulates the Hessian on every call, and on the batch_num-th call runs
                    # the Cholesky/OBS weight update, writing straight into `module.linear.weight`
                    # (HiGPTQuant holds that exact Parameter, not a copy).
                    higptq.forward(captured[name].to(higptq.weight.device))
        finally:
            for handle in handles:
                handle.remove()
            set_model_weight_quant_state(self.model, True)
            set_model_act_quant_state(self.model, True)

        logger.info("Layer {}: {} finished for {} linear(s).", self.layer_idx, algo_name, len(wrapped))

    def finalize(self):
        return PtqParamHandler.export_module(self.model)

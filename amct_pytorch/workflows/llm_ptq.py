from __future__ import annotations

import inspect
import json
import os
from typing import Any

import torch
from loguru import logger

from amct_pytorch.common.datasets.ptq_provider import LlmPtqDataProvider
from amct_pytorch.common.models.llm.common.base import PtqUnit
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.optimization import register_solvers
from amct_pytorch.algorithms.quant import register_algorithms
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.common.models.llm.common.quant_apply import (
    set_model_act_quant_state, set_model_weight_quant_state, set_model_to_observe)
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.optimization import SOLVER_REGISTRY
from amct_pytorch.common.utils.run_logging import ensure_log_dir, setup_run_logging


class LlmPtqWorkflow:

    def __init__(self, args):
        self.args = args
        if len(args.quant_target) != 1:
            raise ValueError("ptq only supports a single quant_target.")
        self.quant_target = args.quant_target[0]
        self.device = args.device
        self.granularity = args.granularity
        self.pipeline = None
        self.data_provider = None
        self.model_name = self.args.model_name
        self.solver_key = getattr(args, "solver", "blockwise")

    @staticmethod
    def _register_components():
        register_algorithms()
        register_llm_models()
        register_dtype()
        register_solvers()

    @staticmethod
    def _unpack_tensor_batch(batch: Any):
        if isinstance(batch, (tuple, list)):
            if len(batch) != 1:
                raise ValueError("Expected TensorDataset batches to contain exactly one tensor.")
            return batch[0]
        return batch

    def setup(self):
        self._register_components()
        self._prepare_experiment_dirs()
        self.pipeline = self._build_pipeline()
        self.data_provider = self._build_data_provider()
        log_name = "ptq"
        if self.granularity == "block":
            log_name = f"ptq_{self.args.start_block_idx}_{self.args.end_block_idx}"
        sink_id, _ = setup_run_logging(self.args, log_name)
        return sink_id

    def run(self):
        sink_id = self.setup()

        solver_cls = SOLVER_REGISTRY.get(self.granularity)
        if self.granularity == "block":
            results = self._run_blockwise(solver_cls)
        elif self.granularity == "model":
            results = self._run_modelwise(solver_cls)
        else:
            raise ValueError(
                f"Unsupported solver granularity '{self.granularity}'."
            )
        logger.remove(sink_id)
        return results

    def _prepare_experiment_dirs(self):
        self.args.log_dir = os.path.join(self.args.output_dir, "logs")
        self.args.quant_param_dir = self._resolve_quant_param_dir()

        for path in (
            self.args.log_dir,
            self.args.quant_param_dir,
        ):
            os.makedirs(path, exist_ok=True)

    def _resolve_quant_param_dir(self):
        target_attr = self._get_quant_param_dir_attr()
        configured_dir = getattr(self.args, target_attr, "")
        if configured_dir:
            return configured_dir

        safe_model_name = self.model_name.strip("/").replace("/", "_")
        auto_dir = os.path.join(
            self.args.output_dir,
            "ptq_params",
            safe_model_name,
            self.quant_target,
        )
        logger.warning(
            "No '{}' specified for quant_target '{}'; auto-create PTQ param dir: {}",
            target_attr,
            self.quant_target,
            auto_dir,
        )
        setattr(self.args, target_attr, auto_dir)
        return auto_dir

    def _get_quant_param_dir_attr(self):
        if self.quant_target == "attn-linear":
            return "attn_linear_param_dir"
        if self.quant_target == "attn-cache":
            return "attn_cache_param_dir"
        if self.quant_target in {"mlp", "moe"}:
            return "moe_mlp_param_dir"
        raise ValueError(f"Unsupported quant_target '{self.quant_target}' for PTQ param dir.")

    def _build_pipeline(self):
        model_cls = MODEL_REGISTRY.get(self.model_name)
        return model_cls(self.args)

    def _build_data_provider(self):
        return LlmPtqDataProvider(self.args, self.pipeline)

    def _move_to_device(self, value):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return value.to(self.device, dtype=torch.float32)
            return value.to(self.device)
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        if isinstance(value, dict):
            return {key: self._move_to_device(item) for key, item in value.items()}
        return value

    def _prepare_unit_batch(self, unit: PtqUnit):
        unit_inputs = self.data_provider.load_unit_inputs(unit)
        if isinstance(unit_inputs, tuple):
            inps, kwargs = unit_inputs
        else:
            inps, kwargs = unit_inputs, {}
        unit.module = unit.module.float().to(self.device)
        inps = self._move_to_device(inps)
        kwargs = self._move_to_device(kwargs)
        set_model_act_quant_state(unit.module, False)
        set_model_weight_quant_state(unit.module, False)
        set_model_to_observe(unit.module, True)
        try:
            gts = self.data_provider.materialize_gt(inps, unit.module, kwargs=kwargs)
        finally:
            set_model_act_quant_state(unit.module, True)
            set_model_weight_quant_state(unit.module, True)
            set_model_to_observe(unit.module, False)
        return self.data_provider.build_unit_batch(unit, inps, kwargs, gts)

    def _run_blockwise(self, solver_cls):
        results = {}
        end_block_idx = min(self.args.end_block_idx, self.pipeline.num_layers)
        if end_block_idx < self.args.end_block_idx:
            logger.warning(
                f"end_block_idx {self.args.end_block_idx} exceeds model num layers "
                f"{self.pipeline.num_layers}; clamped to {end_block_idx}"
            )
        for layer_idx in range(self.args.start_block_idx, end_block_idx):
            logger.info("PTQ block {}", layer_idx)
            block = self.pipeline.build_quant_block(layer_idx)
            units = list(self.pipeline.iter_ptq_units(layer_idx, block))
            if not units:
                logger.warning("No PTQ units found for layer {}", layer_idx)
                continue
            logger.info("Layer {} has {} PTQ unit(s).", layer_idx, len(units))

            layer_results = {}
            for unit in units:
                if self._unit_result_path(unit) and os.path.exists(self._unit_result_path(unit)):
                    logger.info(
                        f"Skip PTQ unit '{unit.name}' in layer {layer_idx}: "
                        f"params already exist at {self._unit_result_path(unit)}"
                    )
                    continue
                unit_batch = self._prepare_unit_batch(unit)
                solver = self._build_block_solver(solver_cls, layer_idx, unit.module)
                logger.info(f"PTQ unit '{unit.name}' ({unit.kind}) in layer {layer_idx}")
                solver.solve(unit_batch.data_loader, forward_kwargs=unit_batch.kwargs)

                unit_result = solver.finalize()
                layer_results[unit.name] = unit_result
                self._save_unit_result(unit, unit_result)

                unit.module.to("cpu")
                del unit_batch, solver, unit_result
                torch.npu.empty_cache()

            results[layer_idx] = layer_results
        return results

    def _run_modelwise(self, solver_cls):
        raise ValueError(
                f"Currently unsupported granularity '{self.granularity}' for ptq."
            )

    def _build_block_solver(self, solver_cls, layer_idx: int, block):
        kwargs = {}
        signature = inspect.signature(solver_cls.__init__)
        parameters = signature.parameters

        if "args" in parameters:
            kwargs["args"] = self.args
        if "layer_idx" in parameters:
            kwargs["layer_idx"] = layer_idx
        if "model" in parameters:
            kwargs["model"] = block
        if "block" in parameters:
            kwargs["block"] = block

        return solver_cls(**kwargs)

    def _build_model_solver(self, solver_cls):
        pass

    def _unit_result_path(self, unit: PtqUnit) -> str | None:
        if not self.args.quant_param_dir:
            return None
        if unit.layer_idx is None:
            file_name = f"{unit.save_name}.pt"
        else:
            file_name = f"layer_{unit.layer_idx}_{unit.save_name}.pt"
        return os.path.join(self.args.quant_param_dir, file_name)

    def _save_unit_result(self, unit: PtqUnit, result: Any):
        save_path = self._unit_result_path(unit)
        torch.save(result, save_path)
        logger.info("Saved PTQ params for layer {} unit '{}' to {}", unit.layer_idx, unit.name, save_path)

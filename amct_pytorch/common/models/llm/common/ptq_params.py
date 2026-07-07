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

import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from amct_pytorch.common.models.llm.common.base import PtqUnit


class PtqParamHandler:
    @staticmethod
    def export_trainable_module(module: nn.Module):
        params = {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            params[name] = param.detach().cpu()
        return params

    @staticmethod
    def export_module(module: nn.Module):
        params = {}
        for name, submodule in module.named_modules():
            if not name:
                continue
            export_fn = getattr(submodule, "export_ptq_params", None)
            if not callable(export_fn):
                continue
            sub_params = export_fn()
            if isinstance(sub_params, dict) and len(sub_params) == 0:
                continue
            if sub_params is None:
                continue
            params[name] = sub_params
        return params

    @staticmethod
    def load_trainable_module(module: nn.Module, params):
        named_params = dict(module.named_parameters())
        for name, value in params.items():
            if name not in named_params:
                raise KeyError(
                    f"Parameter '{name}' is not found in module '{module.__class__.__name__}'."
                )
            target = named_params[name]
            target.data.copy_(value.to(device=target.device, dtype=target.dtype))

    @staticmethod
    def load_module(module: nn.Module, params):
        named_modules = dict(module.named_modules())
        for name, sub_params in params.items():
            if name not in named_modules:
                raise KeyError(
                    f"Submodule '{name}' is not found in module '{module.__class__.__name__}'."
                )
            submodule = named_modules[name]
            load_fn = getattr(submodule, "load_ptq_params", None)
            if not callable(load_fn):
                raise KeyError(
                    f"Submodule '{name}' in module '{module.__class__.__name__}' "
                    "does not implement load_ptq_params()."
                )
            load_fn(sub_params)

    def export_unit(self, unit: "PtqUnit"):
        if hasattr(unit.module, "export_ptq_params"):
            return unit.module.export_ptq_params()
        params = self.export_module(unit.module)
        if params:
            return params
        return self.export_trainable_module(unit.module)

    def load_unit(self, unit: "PtqUnit", params):
        if hasattr(unit.module, "load_ptq_params"):
            unit.module.load_ptq_params(params)
            return
        if isinstance(params, dict) and params and all(isinstance(v, dict) for v in params.values()):
            self.load_module(unit.module, params)
            return
        self.load_trainable_module(unit.module, params)


class PtqParamStore:
    def __init__(self, ptq_param_handler: PtqParamHandler, iter_ptq_units_fn):
        self.ptq_param_handler = ptq_param_handler
        self.iter_ptq_units_fn = iter_ptq_units_fn

    def load_saved_unit(self, param_dir: str, unit: "PtqUnit", strict: bool = False):
        if unit.layer_idx is None:
            file_name = f"{unit.save_name}.pt"
        else:
            file_name = f"layer_{unit.layer_idx}_{unit.save_name}.pt"
        param_path = os.path.join(param_dir, file_name)
        if not os.path.exists(param_path):
            if strict:
                raise FileNotFoundError(f"PTQ params not found for unit '{unit.name}': {param_path}")
            return False
        params = torch.load(param_path, weights_only=True, map_location="cpu")
        self.ptq_param_handler.load_unit(unit, params)
        return True

    def load_layer(self, layer_idx: int, block, param_dir: str, strict: bool = False):
        loaded_units = []
        missing_units = []
        for unit in self.iter_ptq_units_fn(layer_idx, block):
            loaded = self.load_saved_unit(param_dir, unit, strict=strict)
            if loaded:
                loaded_units.append(unit.name)
            else:
                missing_units.append(unit.name)
        if strict and missing_units:
            raise FileNotFoundError(
                f"Missing PTQ params for layer {layer_idx}: {', '.join(missing_units)}"
            )
        return {
            "loaded": loaded_units,
            "missing": missing_units,
        }

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

from dataclasses import dataclass
from typing import Dict, Tuple

from .domains.base import BasePruningDomain
from .domains.cnn import CNNPruningDomain
from .domains.dense import DensePruningDomain
from .domains.moe import MoEPruningDomain
from .prune_op.base import BasePruningMethod
from .prune_op.cnn_reconstruct import ReconstructChannelPruningMethod
from .prune_op.cnn_variance import VarianceChannelPruningMethod
from .prune_op.dense_low_variance import LowVarianceDensePruningMethod
from .prune_op.dense_reconstruct import ReconstructDensePruningMethod
from .prune_op.moe_mass_pruning import MassMoEPruningMethod
from .prune_op.moe_mass_variance_pruning import MassVarianceMoEPruningMethod
from .prune_op.moe_output_merge import OutputMergeMoEPruningMethod


@dataclass
class DomainMethodBinding:
    domain: BasePruningDomain
    method: BasePruningMethod


RegistryMapping = Dict[Tuple[str, str], DomainMethodBinding]


def create_default_registry() -> RegistryMapping:
    """Build the default (domain, method) -> binding mapping for the 7 methods."""
    domains = {
        "cnn": CNNPruningDomain(),
        "dense": DensePruningDomain(),
        "moe": MoEPruningDomain(),
    }
    methods = (
        VarianceChannelPruningMethod(),
        ReconstructChannelPruningMethod(),
        LowVarianceDensePruningMethod(),
        ReconstructDensePruningMethod(),
        MassMoEPruningMethod(),
        MassVarianceMoEPruningMethod(),
        OutputMergeMoEPruningMethod(),
    )
    registry: RegistryMapping = {}
    for method in methods:
        if method.domain not in domains:
            raise KeyError(
                f"Domain '{method.domain}' must be registered before its methods."
            )
        registry[(method.domain, method.name)] = DomainMethodBinding(
            domain=domains[method.domain], method=method
        )
    return registry


def get_binding(
    registry: RegistryMapping, domain_name: str, method_name: str
) -> DomainMethodBinding:
    """Look up the binding for a (domain, method) pair, raising a clear KeyError."""
    binding = registry.get((domain_name, method_name))
    if binding is None:
        known = sorted(m for (d, m) in registry if d == domain_name)
        if not known and not any(d == domain_name for (d, _m) in registry):
            raise KeyError(f"Unknown pruning domain '{domain_name}'.")
        raise KeyError(
            f"Unknown pruning method '{method_name}' for domain '{domain_name}'. Known: {known}"
        )
    return binding

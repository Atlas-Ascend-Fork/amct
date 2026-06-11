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
from typing import Dict

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


@dataclass(slots=True)
class DomainMethodBinding:
    domain: BasePruningDomain
    method: BasePruningMethod


class MethodRegistry:
    def __init__(
        self,
        domains: Dict[str, BasePruningDomain] | None = None,
        methods: Dict[str, Dict[str, BasePruningMethod]] | None = None,
    ) -> None:
        self._domains: Dict[str, BasePruningDomain] = dict(domains) if domains else {}
        self._methods: Dict[str, Dict[str, BasePruningMethod]] = (
            {domain: dict(method_map) for domain, method_map in methods.items()}
            if methods
            else {}
        )

    def clone(self) -> "MethodRegistry":
        return MethodRegistry(self._domains, self._methods)

    def register_domain(self, domain: BasePruningDomain) -> None:
        self._domains[domain.name] = domain

    def register_method(self, method: BasePruningMethod) -> None:
        if method.domain not in self._domains:
            raise KeyError(
                f"Domain '{method.domain}' must be registered before its methods."
            )
        self._methods.setdefault(method.domain, {})[method.name] = method

    def get_binding(self, domain_name: str, method_name: str) -> DomainMethodBinding:
        if domain_name not in self._domains:
            raise KeyError(f"Unknown pruning domain '{domain_name}'.")
        if (
            domain_name not in self._methods
            or method_name not in self._methods[domain_name]
        ):
            known = sorted(self._methods.get(domain_name, {}).keys())
            raise KeyError(
                f"Unknown pruning method '{method_name}' for domain '{domain_name}'. Known: {known}"
            )
        return DomainMethodBinding(
            domain=self._domains[domain_name],
            method=self._methods[domain_name][method_name],
        )


def create_default_registry() -> MethodRegistry:
    registry = MethodRegistry()
    registry.register_domain(CNNPruningDomain())
    registry.register_domain(DensePruningDomain())
    registry.register_domain(MoEPruningDomain())

    registry.register_method(VarianceChannelPruningMethod())
    registry.register_method(ReconstructChannelPruningMethod())
    registry.register_method(LowVarianceDensePruningMethod())
    registry.register_method(ReconstructDensePruningMethod())
    registry.register_method(MassMoEPruningMethod())
    registry.register_method(MassVarianceMoEPruningMethod())
    registry.register_method(OutputMergeMoEPruningMethod())
    return registry

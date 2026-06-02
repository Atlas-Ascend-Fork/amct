from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from .domains.base import BasePruningDomain
from .domains.cnn import CNNPruningDomain
from .domains.dense import DensePruningDomain
from .domains.moe import MoEPruningDomain
from .methods.base import BasePruningMethod
from .methods.cnn_variance import VarianceChannelPruningMethod
from .methods.dense_low_variance import LowVarianceDensePruningMethod
from .methods.moe_mass_pruning import MassMoEPruningMethod
from .methods.moe_mass_variance_pruning import MassVarianceMoEPruningMethod


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
            raise KeyError(f"Domain '{method.domain}' must be registered before its methods.")
        self._methods.setdefault(method.domain, {})[method.name] = method

    def get_binding(self, domain_name: str, method_name: str) -> DomainMethodBinding:
        if domain_name not in self._domains:
            raise KeyError(f"Unknown pruning domain '{domain_name}'.")
        if domain_name not in self._methods or method_name not in self._methods[domain_name]:
            known = sorted(self._methods.get(domain_name, {}).keys())
            raise KeyError(f"Unknown pruning method '{method_name}' for domain '{domain_name}'. Known: {known}")
        return DomainMethodBinding(domain=self._domains[domain_name], method=self._methods[domain_name][method_name])

    def list_methods(self) -> Dict[str, list[str]]:
        return {domain: sorted(methods.keys()) for domain, methods in self._methods.items()}


def create_default_registry() -> MethodRegistry:
    registry = MethodRegistry()
    registry.register_domain(CNNPruningDomain())
    registry.register_domain(DensePruningDomain())
    registry.register_domain(MoEPruningDomain())

    registry.register_method(VarianceChannelPruningMethod())
    registry.register_method(LowVarianceDensePruningMethod())
    registry.register_method(MassMoEPruningMethod())
    registry.register_method(MassVarianceMoEPruningMethod())
    return registry

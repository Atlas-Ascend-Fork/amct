from amct_pytorch.algorithms.registry_factory import (
    ALGO_REGISTRY,
    QUANT_ALGO_REGISTRY,
    register_all_algorithms,
    register_quant_algo,
)
from amct_pytorch.common.utils.registry_factory import Registry


def test_module_level_registries_are_registry_instances():
    assert isinstance(QUANT_ALGO_REGISTRY, Registry)
    assert isinstance(ALGO_REGISTRY, Registry)
    assert QUANT_ALGO_REGISTRY.name == "quant_algo"
    assert ALGO_REGISTRY.name == "algo"


def test_register_quant_algo_returns_none_and_does_not_raise():
    assert register_quant_algo() is None


def test_register_all_algorithms_returns_none_and_does_not_raise():
    assert register_all_algorithms() is None


def test_register_all_algorithms_is_idempotent():
    before_quant = set(QUANT_ALGO_REGISTRY.list_all())
    before_algo = set(ALGO_REGISTRY.list_all())
    register_all_algorithms()
    register_all_algorithms()
    assert set(QUANT_ALGO_REGISTRY.list_all()) == before_quant
    assert set(ALGO_REGISTRY.list_all()) == before_algo

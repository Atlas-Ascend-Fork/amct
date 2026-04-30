from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.common.utils.registry_factory import Registry


def test_algo_registry_is_named_registry_instance():
    assert isinstance(ALGO_REGISTRY, Registry)
    assert ALGO_REGISTRY.name == "algo"

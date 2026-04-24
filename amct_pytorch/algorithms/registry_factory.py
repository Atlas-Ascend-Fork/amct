from amct_pytorch.common.utils.registry_factory import Registry

QUANT_ALGO_REGISTRY = Registry("quant_algo")
ALGO_REGISTRY = Registry("algo")

def register_quant_algo():
    pass

def register_all_algorithms():
    register_quant_algo()
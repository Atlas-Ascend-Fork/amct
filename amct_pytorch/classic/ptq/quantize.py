__all__ = []
from amct_pytorch.algorithms.registry_factory import QUANT_ALGO_REGISTRY
from amct_pytorch.common.optimization.global_solver import GlobalSolver

def quantize(model, config):
    algo = QUANT_ALGO_REGISTRY.get(config.get("algorithm", "minmax"))
    solver = GlobalSolver(config)
    solver.set_adapter(config.get("model_name", "auto"), "classic")
    return solver.solve(model, config.get("calibration_data"))

def convert(model):
    pass

def algorithm_register(name, src_op, quant_op, deploy_op):
    pass
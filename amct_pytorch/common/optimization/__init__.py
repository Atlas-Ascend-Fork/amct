__all__ = ["register_solvers", "SOLVER_REGISTRY"]

from amct_pytorch.common.utils.registry_factory import Registry

SOLVER_REGISTRY = Registry("solver")

_REGISTERED = False


def register_solvers():
    global _REGISTERED
    if _REGISTERED:
        return

    from .blockwise_solver import BlockwiseSolver  # noqa: F401

    _REGISTERED = True

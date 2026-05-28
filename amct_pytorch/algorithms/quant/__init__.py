__all__ = [
    'awq', 'gptq', 'smooth_quant', 'minmax',
    'auto_clip', 'omniquant', 'learnable_hadamard',
    'register_algorithms', 'AlgoBuildContext',
]

from dataclasses import dataclass


_REGISTERED = False


def register_algorithms():
    global _REGISTERED
    if _REGISTERED:
        return

    from .auto_clip import LAC, LWC
    from .auto_round import AutoRound
    from .omniquant import OmniQuant

    _REGISTERED = True


@dataclass
class AlgoBuildContext:
    matrix_size: int | None = None
    dim_size: int | None = None

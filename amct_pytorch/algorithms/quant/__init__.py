from dataclasses import dataclass

__all__ = [
    'awq', 'gptq', 'smooth_quant', 'minmax',
    'auto_clip', 'omniquant', 'learnable_hadamard',
    'register_algorithms', 'AlgoBuildContext',
]


_REGISTERED = False


def register_algorithms():
    global _REGISTERED
    if _REGISTERED:
        return

    from .auto_clip import LAC, LWC
    from .learnable_hadamard import LearnableHadamard
    from .omniquant import OmniQuant
    from .auto_round import AutoRound

    _REGISTERED = True


@dataclass
class AlgoBuildContext:
    matrix_size: int | None = None
    dim_size: int | None = None

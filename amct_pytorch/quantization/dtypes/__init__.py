__all__ = ["register_dtype", "DTYPE_REGISTRY"]

from amct_pytorch.common.utils.registry_factory import Registry

DTYPE_REGISTRY = Registry("dtype")

_REGISTERED = False


def register_dtype():
    global _REGISTERED
    if _REGISTERED:
        return

    from .int import QuantDequantInt
    from .mxfp import QuantDequantMx

    _REGISTERED = True

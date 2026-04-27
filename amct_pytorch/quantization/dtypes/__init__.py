from amct_pytorch.common.utils.registry_factory import Registry

DTYPE_REGISTRY = Registry("dtype")

_REGISTERED = False


def register_dtype():
    global _REGISTERED
    if _REGISTERED:
        return

    from .mxfp import QuantDequantMx  # noqa: F401
    from .int import QuantDequantInt
    try:
        from .hifx import QuantDequantHifp
    except ImportError as exc:
        from loguru import logger
        logger.warning(f"Skip hifp dtype registration: {exc}")

    _REGISTERED = True


__all__ = ["register_dtype", "DTYPE_REGISTRY"]

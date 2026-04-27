import importlib

from loguru import logger


_REGISTERED = False

_LLM_FAMILIES = (
    "amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.deepseekv3_2",
    "amct_pytorch.common.models.llm.longcat.longcat_lite.longcat_lite",
    "amct_pytorch.common.models.llm.longcat.longcat_next.longcat_next",
    "amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b",
    "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5",
    "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe",
    "amct_pytorch.common.models.llm.qwen.qwen3_6.qwen3_6_moe",
    "amct_pytorch.common.models.llm.qwen.qwen3.qwen3",
    "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next",
    "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe",
    "amct_pytorch.common.models.llm.glm.glm5.glm5",
)


def register_llm_models():
    global _REGISTERED
    if _REGISTERED:
        return

    for module_path in _LLM_FAMILIES:
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            logger.warning(f"Skip LLM family registration for {module_path}: {exc}")

    _REGISTERED = True


__all__ = ["register_llm_models"]


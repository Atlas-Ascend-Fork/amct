_REGISTERED = False


def register_llm_models():
    global _REGISTERED
    if _REGISTERED:
        return

    from .deepseek.deepseek_v3_2.deepseekv3_2 import DeepseekV32  # noqa: F401
    from .deepseek.deepseek_v4.deepseekv4 import DeepseekV4  # noqa: F401
    from .longcat.longcat_lite.longcat_lite import LongcatLite  # noqa: F401
    from .longcat.longcat_next.longcat_next import LongcatNext  # noqa: F401
    from .qwen.qwen3_5.qwen3_5 import Qwen3_5  # noqa: F401
    from .qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe  # noqa: F401
    from .qwen.qwen3_6.qwen3_6_moe import Qwen3_6Moe  # noqa: F401
    from .qwen.qwen3.qwen3 import Qwen3  # noqa: F401
    from .qwen.qwen3_next.qwen3_next import Qwen3Next  # noqa: F401
    from .qwen.qwen3.qwen3_moe import Qwen3Moe  # noqa: F401
    from .glm.glm5.glm5 import GLM5

    _REGISTERED = True


__all__ = ["register_llm_models"]


from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
from amct_pytorch.common.models import MODEL_REGISTRY


@MODEL_REGISTRY.register(
    name="qwen3_6_moe",
    task="llm",
    family="qwen",
    description="Qwen3.6 moe model adapter",
)
class Qwen3_6Moe(Qwen3_5Moe):
    pass


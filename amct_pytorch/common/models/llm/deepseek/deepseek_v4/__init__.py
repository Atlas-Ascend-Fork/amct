__all__ = [
    "DeepseekV4Config",
    "DeepseekV4ForCausalLM",
    "DeepseekV4PreTrainedModel",
]

from transformers import AutoConfig, AutoModelForCausalLM

from amct_pytorch.common.models.llm.deepseek.deepseek_v4.modeling.configuration_deepseek_v4 import DeepseekV4Config
from amct_pytorch.common.models.llm.deepseek.deepseek_v4.modeling.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4PreTrainedModel,
)

AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True)

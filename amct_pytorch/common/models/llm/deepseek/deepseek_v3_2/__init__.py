__all__ = [
    "DeepseekV32Config",
    "DeepseekV32ForCausalLM",
    "DeepseekV32PreTrainedModel",
]

from transformers import AutoConfig, AutoModelForCausalLM

from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.modeling.configuration_deepseek import DeepseekV32Config
from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.modeling.modeling_deepseek_v3_2 import (
    DeepseekV32ForCausalLM,
    DeepseekV32PreTrainedModel,
)

AutoConfig.register("deepseek_v32", DeepseekV32Config, exist_ok=True)
AutoModelForCausalLM.register(DeepseekV32Config, DeepseekV32ForCausalLM, exist_ok=True)
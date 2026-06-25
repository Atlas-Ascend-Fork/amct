from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from .modeling_vq import LCVQConfig, LCVQForCausalLM
try:
    from .modeling_qwen3_vq import Qwen3TVQConfig, Qwen3TVQForCausalLM
except ModuleNotFoundError:
    Qwen3TVQConfig = None
    Qwen3TVQForCausalLM = None

AutoConfig.register('lcvq', LCVQConfig)
AutoModel.register(LCVQConfig, LCVQForCausalLM)
AutoModelForCausalLM.register(LCVQConfig, LCVQForCausalLM)

if Qwen3TVQConfig is not None and Qwen3TVQForCausalLM is not None:
    AutoConfig.register('tvqqwen3', Qwen3TVQConfig)
    AutoModel.register(Qwen3TVQConfig, Qwen3TVQForCausalLM)
    AutoModelForCausalLM.register(Qwen3TVQConfig, Qwen3TVQForCausalLM)
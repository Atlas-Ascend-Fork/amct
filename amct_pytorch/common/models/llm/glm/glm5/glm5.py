import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoConfig
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.deepseekv3_2 import DeepseekV32


CUR_DIR = os.path.dirname(__file__)


@MODEL_REGISTRY.register(
    name="glm5",
    task="llm",
    family="glm",
    description="GLM-5.1 model adapter",
)
class GLM5(DeepseekV32):
    def __init__(self, args):
        super().__init__(args)
        self.config = AutoConfig.from_pretrained(os.path.join(CUR_DIR, "config.json"))
        self.model = self.empty_weights_model()

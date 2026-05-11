import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from amct_pytorch.quantization.dtypes.hifloat8 import hifloat8_cast
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY


@DTYPE_REGISTRY.register(name="hifp", description="quant dequant for hifp")
class QuantDequantHifp(torch.nn.Module):
    def __init__(self, bits=8, is_act=False):
        super(QuantDequantHifp, self).__init__()
        self.bits = bits
        self.is_act = is_act
        self._deploy_mod = False
        self.temperature=1.0
        self.hard = True
        self.eps = 1e-12
        self.register_buffer(
            "level_cost",
            torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        )
        self.level_logits = nn.Parameter(torch.tensor([2.0, 0.0, -2.0]))

    def fake_quant(self, x: Tensor) -> Tensor:
        if self.bits == 8:
            qx = hifloat8_cast.float_to_hifloat8(x.float())
            if x.dtype == torch.float32:
                dx = hifloat8_cast.hifloat8_to_float32(qx)
            elif x.dtype == torch.float16:
                dx = hifloat8_cast.hifloat8_to_float16(qx)
            elif x.dtype == torch.bfloat16:
                dx = hifloat8_cast.hifloat8_to_bfloat16(qx)
            else:
                raise TypeError(f"Unknown dtype {x.dtype}")
        else:
            raise ValueError(f" HiFP{self.bits} is comming soon")
        return dx

    def quantize_with_level_prob(self, x, p):
        q1 = self.quantize_level1(x)
        q2 = self.quantize_level2(x)
        q3 = self.quantize_level3(x)

        return p[0] * q1 + p[1] * q2 + p[2] * q3
    
    def forward(self, x: Tensor) -> Tensor:
        if self.bits == 16:
            return x
        
        p = self.get_level_probability(noise_disable=not self.training)
        return self.quantize_with_level_prob(x, p)
        # x = (self.fake_quant(x) - x).detach() + x
        # return x

    @torch.no_grad()
    def export_level(self):
        return int(torch.argmax(self.level_logits).item())

    def expected_cost(self):
        p = F.softmax(self.level_logits, dim=-1)
        return (p * self.level_cost).sum()
    
    def get_level_probability(self, noise_disable=False):
        if noise_disable:
            return F.softmax(self.level_logits, dim=-1)

        return F.gumbel_softmax(
            self.level_logits,
            tau=self.temperature,
            hard=self.hard,
            dim=-1,
        )
    
    def export_deploy(self, x: Tensor):
        qx = hifloat8_cast.float_to_hifloat8(x.float())
        return {
            "qweight": qx.detach().cpu(),
        }

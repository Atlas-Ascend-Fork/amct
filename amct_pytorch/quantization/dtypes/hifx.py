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
    
    def forward(self, x: Tensor) -> Tensor:
        if self.bits == 16:
            return x
        x = (self.fake_quant(x) - x).detach() + x
        return x
    
    def export_deploy(self, x: Tensor):
        qx = hifloat8_cast.float_to_hifloat8(x.float())
        return {
            "qweight": qx.detach().cpu(),
        }

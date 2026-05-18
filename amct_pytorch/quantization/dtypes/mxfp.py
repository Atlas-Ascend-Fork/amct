import torch
from torch import Tensor
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY
from amct_pytorch.quantization.dtypes.mxfp_impl import (
    f32_to_f4_unpacked, pack_uint4, quantize_elewise, shared_exponents,
)

@DTYPE_REGISTRY.register(name="mxfp", description="quant dequant for mxfp")
class QuantDequantMx(torch.nn.Module):
    def __init__(self, bits=8, is_act=False):
        super(QuantDequantMx, self).__init__()
        self.bits = bits
        self.is_act = is_act
        self._get_format_params()
        self.block_size = 32
        
    def _get_format_params(self):
        if self.bits == 8:
            self.ebits, self.mbits, self.emax, self.max_norm, self.shift_val, self.min_exp = 4, 5, 8, 448.0, 8, -6
        else:
            self.ebits, self.mbits, self.emax, self.max_norm, self.shift_val, self.min_exp = 2, 3, 2, 6.0, 2, 0

    def quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        x = x.unflatten(qdim, (-1, self.block_size))
        if isinstance(v, torch.Tensor):
            v = v.to(device=x.device, dtype=x.dtype).unflatten(qdim, (-1, self.block_size))
        E8M0 = shared_exponents(x, self.emax)
        x = x / (2 ** E8M0)
        ExMx = quantize_elewise(x, self.min_exp, self.max_norm, self.shift_val, v=v)
        return E8M0, ExMx

    def fake_quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        E8M0, ExMx = self.quant(x, qdim, v=v)
        dx = (2 ** E8M0) * ExMx
        dx = dx.flatten(qdim - 1, qdim)
        return dx

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.bits == 16:
            return x
        return self.fake_quant(x, v=v)

    def deploy(self, x: Tensor, qdim: int = -1):
        E8M0, ExMx = self.quant(x, qdim)
        E8M0 = (E8M0 + 127).to(torch.uint8).squeeze(-1).cpu()
        ExMx = ExMx.flatten(qdim - 1, qdim).cpu()
        if self.bits == 8:
            ExMx = ExMx.to(torch.float8_e4m3fn)
        else:
            ExMx = f32_to_f4_unpacked(ExMx.float().cpu())
            ExMx = pack_uint4(ExMx)
        return ExMx, E8M0


    def export_deploy(self, x: Tensor):
        qx, scale = self.deploy(x)
        return {
            "qweight": qx.detach().cpu(),
            "weight_scale": scale.detach().cpu(),
        }

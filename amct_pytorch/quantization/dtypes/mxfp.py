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
        
    def deploy(self, x: Tensor, qdim: int = -1):
        e8m0, ex_mx = self.quant(x, qdim)
        e8m0 = (e8m0 + 127).to(torch.uint8).squeeze(-1).cpu()
        ex_mx = ex_mx.flatten(qdim - 1, qdim).cpu()
        if self.bits == 8:
            ex_mx = ex_mx.to(torch.float8_e4m3fn)
        else:
            ex_mx = f32_to_f4_unpacked(ex_mx.float().cpu())
            ex_mx = pack_uint4(ex_mx)
        return ex_mx, e8m0

    def export_deploy(self, x: Tensor):
        qx, scale = self.deploy(x)
        return {
            "qweight": qx.detach().cpu(),
            "weight_scale": scale.detach().cpu(),
        }

    def fake_quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        e8m0, ex_mx = self.quant(x, qdim, v=v)
        dx = (2 ** e8m0) * ex_mx
        dx = dx.flatten(qdim - 1, qdim)
        return dx

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.bits == 16:
            return x
        return self.fake_quant(x, v=v)

    def quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        x = x.unflatten(qdim, (-1, self.block_size))
        if isinstance(v, torch.Tensor):
            v = v.to(device=x.device, dtype=x.dtype).unflatten(qdim, (-1, self.block_size))
        e8m0 = shared_exponents(x, self.emax)
        x = x / (2 ** e8m0)
        ex_mx = quantize_elewise(x, self.min_exp, self.max_norm, self.shift_val, v=v)
        return e8m0, ex_mx

    def _get_format_params(self):
        if self.bits == 8:
            self.ebits, self.mbits, self.emax, self.max_norm, self.shift_val, self.min_exp = 4, 5, 8, 448.0, 8, -6
        else:
            self.ebits, self.mbits, self.emax, self.max_norm, self.shift_val, self.min_exp = 2, 3, 2, 6.0, 2, 0

import math

import torch
import torch.nn as nn

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY


@ALGO_REGISTRY.register(
    name="gptq",
    description="Minimal GPTQ-style Hessian-aware weight quantization",
    targets=("weight",),
)
class GPTQ(nn.Module):
    """
    Minimal GPTQ core adapted to the current framework.

    This implementation keeps the essential GPTQ logic:
    1. collect input Hessian statistics during an observe pass
    2. quantize weights column-by-column with second-order error compensation

    It does not port the original repository's full model wrappers or custom
    quantizer classes. Instead, it reuses the current framework's downstream
    `quant_obj` callable for the actual quantization step.
    """

    def __init__(self, args, w_bits):
        super().__init__()
        self.args = args
        self.blocksize = int(getattr(args, "gptq_blocksize", 128))
        self.percdamp = float(getattr(args, "gptq_percdamp", 0.01))
        self.actorder = bool(getattr(args, "gptq_actorder", False))
        self.is_observe = False

        weight_shape = tuple(args.w_size)
        if len(weight_shape) != 2:
            raise ValueError(f"GPTQ currently expects 2D weight tensors, got shape {weight_shape}")
        self.rows = weight_shape[0]
        self.columns = weight_shape[1]

        self.register_buffer("H", torch.zeros((self.columns, self.columns), dtype=torch.float32))
        self.register_buffer("nsamples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_cached_qweight", torch.empty(0), persistent=False)
        self.quant_obj = DTYPE_REGISTRY.get(args.quant_dtype)(bits=w_bits)
        if hasattr(self.quant_obj, "group_size"):
            self.quant_obj.group_size = int(getattr(args, "group_size", -1))
        self._observed_weight = None

    def observe_input(self, inp: torch.Tensor, weight: torch.Tensor | None = None):
        if not self.is_observe:
            return
        if weight is not None:
            self._observed_weight = weight.detach()
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)

        tmp = inp.shape[0]
        if inp.ndim == 3:
            inp = inp.reshape(-1, inp.shape[-1])

        inp = inp.t().float()
        current_nsamples = int(self.nsamples.item())
        total_nsamples = current_nsamples + tmp
        if total_nsamples <= 0:
            return

        if current_nsamples > 0:
            self.H.mul_(current_nsamples / total_nsamples)
        self.nsamples.fill_(total_nsamples)
        inp = math.sqrt(2.0 / total_nsamples) * inp
        self.H.add_(inp.matmul(inp.t()))
        self._cached_qweight = torch.empty(0, device=self.H.device)

    def trainable_params(self):
        return []

    def export_ptq_params(self):
        if int(self.nsamples.item()) == 0:
            return {}
        params = {
            "H": self.H.detach().cpu(),
            "nsamples": int(self.nsamples.item()),
        }
        # Run the corrected per-channel quantization once at calibration time
        # and persist the result, so eval/deploy reuse it instead of redoing
        # the Cholesky path on every forward.
        if self._observed_weight is not None:
            qweight = self.quantize(self._observed_weight, self.quant_obj)
            params["qweight"] = qweight.detach().cpu()
        return params

    def load_ptq_params(self, params):
        if "nsamples" in params:
            self.nsamples.fill_(int(params.get("nsamples", 0)))
        if "H" in params:
            self.H.copy_(params["H"].to(device=self.H.device, dtype=self.H.dtype))
        # Prefer the persisted quantized weight: quantize() then hits its
        # cache and skips the Cholesky recompute entirely.
        if "qweight" in params:
            self._cached_qweight = params["qweight"].to(device=self.H.device)
        else:
            self._cached_qweight = torch.empty(0, device=self.H.device)

    @torch.no_grad()
    def quantize(self, weight: torch.Tensor, quant_obj):
        if self._cached_qweight.numel() != 0 and tuple(self._cached_qweight.shape) == tuple(weight.shape):
            return self._cached_qweight.to(device=weight.device, dtype=weight.dtype)

        if int(self.nsamples.item()) == 0:
            return quant_obj(weight)

        # Fixed per-output-channel scale from the full weight. Feeding a single
        # column to quant_obj degenerates the scale to per-element (identity);
        # GPTQ needs one scale per row, computed once and applied per column.
        bits = getattr(quant_obj, "bits", None)
        gsize = getattr(quant_obj, "group_size", -1)
        if bits is None or bits >= 16:
            qmax = None
            group_scale = None
            eff_group = self.columns
        else:
            qmax = 2 ** (bits - 1) - 1
            if gsize is not None and gsize > 0 and self.columns % gsize == 0:
                eff_group = gsize
            else:
                eff_group = self.columns
            n_groups = self.columns // eff_group
            group_scale = (
                weight.detach().float().abs()
                .reshape(self.rows, n_groups, eff_group)
                .amax(dim=2) / qmax
            ).clamp(min=1e-9)

        W = weight.detach().float().clone()
        H = self.H.to(device=weight.device, dtype=torch.float32).clone()

        dead = torch.diag(H) == 0
        if torch.any(dead):
            H[dead, dead] = 1
            W[:, dead] = 0

        perm = None
        invperm = None
        if self.actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=weight.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Q = torch.zeros_like(W)

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if group_scale is None:
                    q = quant_obj(w.unsqueeze(1)).flatten().to(W1)
                else:
                    s = group_scale[:, (i1 + i) // eff_group]
                    q = (torch.clamp(torch.round(w / s), -qmax, qmax) * s).to(W1)
                Q[:, i1 + i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            if i2 < self.columns:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if self.actorder and invperm is not None:
            Q = Q[:, invperm]

        Q = Q.reshape(weight.shape).to(dtype=weight.dtype)
        self._cached_qweight = Q.detach().to(device=self.H.device, dtype=weight.dtype)
        return Q

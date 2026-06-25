import os
import math

import torch
from torch import nn

from ..utils.matmul_had import matmul_hadU_cuda

import glog

_LI_CODESZ = 4
_LI_CODEBIT = 2

# M8.3.5 Lever 10: separable bucketize+LUT nearest-neighbour selection.
# The codebook grid is an axis-aligned regular lattice (linear_proj.weight =
# eye*proj_scale, ortho_mat = identity), so the global argmin ‖w-g‖² over the
# 4^4 product grid equals the per-dimension nearest level.  Replacing the
# addmm([256,cs]@[cs,m]) + argmax(256) + gather hot path with bucketize +
# 4-entry decode-LUT removes the [256,m] score blow-up and the msprof #1 op
# (aclnnArgMax, ~34%), cutting the return_idx=False hot path from 3 NPU
# dispatches to 2.  Verified MORE accurate than addmm-argmax at fp16/bf16
# (closer to the fp32 ground-truth NN; less low-precision accumulation).
# Default on; set HARNESS_QUANTIZE_ARGMAX=1 to fall back (e.g. if a target
# lacks aclnnBucketize support).
_USE_BUCKETIZE = os.environ.get('HARNESS_QUANTIZE_ARGMAX', '0') != '1'

class Index_codebook(nn.Module):
    
    def __init__(self, inference=False, **kwargs):
        super().__init__()
        self.id = "index"
        self.codesz = _LI_CODESZ
        self.opt_scale = 1.0
        self.idx_dtype = torch.uint8
        self.packsz = 1
        self.pack_out = False
        self.version = 0
        self.codebook_scale = 1.0

        self.min_var = 0
        self.max_var = 2**_LI_CODEBIT - 1
        
        linear_proj = nn.Linear(_LI_CODESZ, _LI_CODESZ, bias=True, dtype=torch.bfloat16)
        linear_proj.weight.requires_grad = False
        linear_proj.bias.requires_grad = False
        input_index_sample = torch.cartesian_prod(*([torch.arange(2**_LI_CODEBIT)]*_LI_CODESZ)).to(device=linear_proj.weight.device, dtype=linear_proj.weight.dtype)
        # print(input_index_sample)
        # if len(input_index_sample.shape) == 1:
        #     input_index_sample = input_index_sample.unsqueeze(1)
        ortho_mat = torch.eye(_LI_CODESZ)
        ortho_mat = ortho_mat.to(device=linear_proj.weight.device, dtype=linear_proj.weight.dtype)
        proj_mean = (2**_LI_CODEBIT-1)/2
        proj_scale = (torch.pow(torch.arange(2**_LI_CODEBIT, dtype=torch.bfloat16), 2).mean().item() - proj_mean**2)
        proj_scale = 1/math.sqrt(proj_scale)
        linear_proj.weight.data = ortho_mat * proj_scale
        linear_proj.bias.data = -nn.functional.linear(torch.ones(1, _LI_CODESZ, device=linear_proj.weight.device, dtype=linear_proj.weight.dtype), ortho_mat).squeeze() * proj_scale * proj_mean
        fake_codebook = linear_proj(input_index_sample) 
        self.register_buffer('linear_proj_weight', linear_proj.weight.data.clone())
        self.register_buffer('linear_proj_bias', linear_proj.bias.data.clone())
        self.register_buffer('grid', fake_codebook)
        if not inference:
            self.register_buffer('grid_norm', fake_codebook.norm(dim=-1)**2)
            # M8.3.5 Lever 5: fold the constant factor 2 into the (transposed) grid so the
            # per-call scalar `2 * w` elementwise op disappears.  Quantization picks
            # argmax_g (2*w·g - |g|^2); since 2*(w @ grid.T) == w @ (2*grid).T, precompute
            # (2*grid).T contiguous once.  This removes one NPU kernel dispatch from EVERY
            # cb.quantize() call (quip + tune + block_LDL paths) and is bit-identical.
            self.register_buffer('grid_2T', (2 * fake_codebook).T.contiguous())
            # M8.3.5 Lever 6: column-major scoring/output buffers for quantize_T.
            # grid_2  = 2*grid as [256, codesz]; gridT = grid.T as [codesz, 256].
            # These let quantize_T consume a [codesz, m] tensor directly and emit
            # [codesz, m] without the two `.T` round-trips that msprof flagged as the
            # #1 NPU device op (aclnnInplaceCopy_Transpose, 10.2%) in the GS tune loop.
            self.register_buffer('grid_2', (2 * fake_codebook).contiguous())
            self.register_buffer('gridT', fake_codebook.T.contiguous())
            # M8.3.5 Lever 7: fold the `- grid_norm` scoring bias into the scoring
            # matmul via torch.addmm.  Both `quantize` and `quantize_T` compute
            # `grid @ w - |grid|^2` (argmax over grid points → nearest neighbour);
            # the subtract was a SEPARATE broadcast NPU dispatch on every call.
            # quantize_T/quantize are the single hottest kernels in the pipeline
            # (~60k calls/layer: down_proj tune alone = 2240 blocks × 10 iters =
            # 22.4k).  Since wall-clock is host/dispatch-bound (Levers 1–6 proved
            # device-compute cuts barely move wall-clock), removing one dispatch
            # from EVERY call is the highest-leverage quality-neutral lever left.
            # addmm(bias, A, B) = bias + A@B, so bias = -grid_norm makes the score
            # bit-identical (same fp matmul reduction; norm is an additive constant
            # per grid point, identical argmax).  `_neg_grid_norm`  → row-major
            # [256] broadcast over [N,256]; `_neg_grid_norm_col` → col-major [256,1]
            # broadcast over [256,m].
            self.register_buffer('_neg_grid_norm', (-self.grid_norm).contiguous())
            self.register_buffer('_neg_grid_norm_col', (-self.grid_norm).unsqueeze(1).contiguous())

            # M8.3.5 Lever 10: separable bucketize+LUT buffers (see module header).
            # `_decode_lut` = the 4 per-dim grid level values = grid[0:4, codesz-1]
            # (every dimension shares the same level→value map since proj is
            # axis-aligned with a single scale).  Gathering with the per-dim level
            # is bit-identical to grid[flat_idx] reconstruction.
            self.register_buffer('_decode_lut',
                fake_codebook[:2**_LI_CODEBIT, _LI_CODESZ - 1].clone().contiguous())
            # `_bucket_bnd` = the (2^codebit - 1) midpoints between consecutive
            # *stored* level values.  Deriving them from `_decode_lut` (in fp32,
            # then cast back) keeps the decision boundaries exactly consistent with
            # the values the LUT decodes to, so bucketize finds the true nearest
            # stored grid point.  torch.bucketize (default right=False) then maps
            # each scalar to its nearest level in 0..3.
            _lut_f = self._decode_lut.float()
            self.register_buffer('_bucket_bnd',
                (0.5 * (_lut_f[:-1] + _lut_f[1:])).to(fake_codebook.dtype).contiguous())
            # cartesian_prod orders the first dim slowest, so the flat grid index
            # is sum_k level[k] * (2^codebit)^(codesz-1-k).
            _idx_powers = torch.tensor(
                [(2**_LI_CODEBIT) ** (_LI_CODESZ - 1 - k) for k in range(_LI_CODESZ)],
                dtype=torch.long)
            self.register_buffer('_idx_powers', _idx_powers.contiguous())
            self.register_buffer('_idx_powers_col', _idx_powers.unsqueeze(1).contiguous())


    def quantize(self, w, return_idx=True, **kwargs):
        assert w.shape[-1] == self.codesz
        # glog.debug(f'before quantize: {w.shape}, {w.mean().item()}, {w.std().item()}, {torch.norm(w, dim=1).tolist()}')
        # M8.3.5 Lever 5: (2*grid).T precomputed (grid_2T) → one fewer NPU dispatch per
        # call vs `2 * w @ self.grid.T`.  Math is bit-identical (w @ (2*grid).T == 2*w@grid.T).
        if _USE_BUCKETIZE:
            # M8.3.5 Lever 10: separable bucketize + decode-LUT (see module header).
            # 2 NPU dispatches on the hot return_idx=False path (bucketize + gather)
            # vs 3 (addmm + argmax + gather), and no [256,N] score blow-up.
            lvl = torch.bucketize(w, self._bucket_bnd)   # [N, codesz] levels in 0..3
            wq = self._decode_lut[lvl]                   # [N, codesz], == grid recon
            if return_idx:
                wqidx = (lvl * self._idx_powers).sum(-1)
                return wq, wqidx.to(self.idx_dtype)
            return wq
        # M8.3.5 Lever 7: addmm folds `- grid_norm` into the matmul (one fewer NPU
        # dispatch/call). Bit-identical to `w @ grid_2T - grid_norm`.
        wqidx = torch.addmm(self._neg_grid_norm, w, self.grid_2T).argmax(1)
        wq = self.grid[wqidx, :]
        # glog.debug(f'after quantize: {wq.shape}, {wq.mean().item()}, {wq.std().item()}')
        if return_idx:
            return wq, wqidx.to(self.idx_dtype)
        return wq

    def quantize_T(self, wT, return_idx=True, **kwargs):
        '''
        M8.3.5 Lever 6 — column-major variant of `quantize`.

        `wT` is [codesz, m]: each COLUMN is a codesz-vector to quantize (this is
        exactly the `WXWX` orientation produced by the LDLQ/GS inner loop, which
        previously had to `.T` it into [m, codesz] for `quantize` and `.T` the
        result back).  Output is [codesz, m] directly.

        Math is bit-identical to `quantize(wT.T).T`:
          scores_cm[g, j] = (grid_2 @ wT)[g, j] = sum_k 2*grid[g,k]*wT[k,j]
                          = (wT.T @ grid_2T)[j, g] = scores_rm[j, g],
        so `argmax(0)` over g yields the same indices (same scan order over the
        256 grid points, identical fp reduction over the codesz dim).  The only
        difference vs `quantize` is operand layout — both `.T` round-trips in the
        caller disappear, which directly removes the msprof #1 device op.
        '''
        assert wT.shape[0] == self.codesz
        if _USE_BUCKETIZE:
            # M8.3.5 Lever 10: column-major separable bucketize + decode-LUT.
            # `_decode_lut[lvl]` for lvl=[codesz, m] yields [codesz, m] directly,
            # so the [codesz, m] -> [codesz, m] orientation of Lever 6 is preserved
            # with no [256, m] score tensor and no argmax.
            lvl = torch.bucketize(wT, self._bucket_bnd)  # [codesz, m] levels in 0..3
            wq = self._decode_lut[lvl]                   # [codesz, m]
            if return_idx:
                wqidx = (lvl * self._idx_powers_col).sum(0)  # [m]
                return wq, wqidx.to(self.idx_dtype)
            return wq
        # M8.3.5 Lever 7: addmm folds `- grid_norm[:,None]` into the matmul (one
        # fewer NPU dispatch/call). Bit-identical to `grid_2 @ wT - grid_norm[:,None]`.
        wqidx = torch.addmm(self._neg_grid_norm_col, self.grid_2, wT).argmax(0)  # [m]
        wq = self.gridT[:, wqidx]                                        # [codesz, m]
        if return_idx:
            return wq, wqidx.to(self.idx_dtype)
        return wq
    
    def maybe_pack_idxs(self, idxs):
        return idxs

    def by_idxs(self, Qidxs, **kwargs):
        m = Qidxs.shape[0]
        Qidxs = Qidxs.to(torch.long).flatten()
        w = self.grid[Qidxs, :].reshape(m, -1)
        return w

class IndexLinear(nn.Module):
    
    def __init__(self, device, **kwarg):
        super().__init__()
        self.codebook = Index_codebook().to(torch.bfloat16).to(device)
        self.scale = 32

    def maybe_unpack_idxs(self, idxs):
        return (idxs, )

    def forward(self, input, Qidxs_list, SU, SV, had_left_T, had_right, K_left, K_right, rescale_WH=False, scaleWH=None, **kwargs):
        n, m = len(SU), len(SV)
        x = input.view(-1, n).to(torch.float32)
        if rescale_WH:
            x /= scaleWH
        x = x * SU

        # had_left_T buffer now holds the un-transposed hadK_L (QuantizedLinear no
        # longer transposes at register time); transpose=True applies hadK_L.T.
        x = matmul_hadU_cuda(x, had_left_T, K_left, transpose=True) / self.scale
    
        W_decompressed = self.codebook.by_idxs(Qidxs_list[0])
        x = (x.to(W_decompressed.dtype) @ W_decompressed.T).to(torch.float32)

        x = matmul_hadU_cuda(x, had_right, K_right)
        x = x * SV * self.scale

        output = x.view(*input.shape[:-1], m)
        
        return output
    

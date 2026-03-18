# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import torch

SINQ_KURTOSIS_THRESHOLD = 5.0


def kurtosis(x, eps=1e-6):
    x = x.to(torch.float32).reshape(-1)
    mu = x.mean()
    xc = x - mu
    var = (xc * xc).mean().clamp_min(eps)
    m4 = xc.pow(4).mean()
    return m4 / (var * var)


def sinkhorn_log(matrix, order=16, clip_min=1e-3, clip_max=1e3, eps=1e-6,
                 stop_on_increasing_imbalance=True):
    """
    Port of the original SINQ sinkhorn helper. It returns the scaling vectors
    that produced the minimum imbalance encountered during iteration.
    """
    dtype = torch.float32
    m = matrix.to(dtype)
    dev = m.device

    def imbalance(mat):
        s1 = torch.std(mat, dim=1)
        s2 = torch.std(mat, dim=0)
        s_min = torch.minimum(s1.min(), s2.min()).clamp_min(1e-12)
        s_max = torch.maximum(s1.max(), s2.max())
        return s_max / s_min

    imb_min = torch.tensor(float('inf'), dtype=dtype, device=dev)
    gate = torch.tensor(0.0, dtype=dtype, device=dev)
    tgt_small = torch.minimum(
        m.std(dim=1).clamp(clip_min, clip_max).min(),
        m.std(dim=0).clamp(clip_min, clip_max).min()
    ) + eps

    log_mu1 = torch.zeros(m.shape[1], dtype=dtype, device=dev)
    log_mu2 = torch.zeros(m.shape[0], 1, dtype=dtype, device=dev)
    mu1_star = log_mu1.exp().clone()
    mu2_star = log_mu2.exp().clone()

    cur0 = m
    ib0 = imbalance(cur0)
    imb_min = torch.minimum(imb_min, ib0)

    for _ in range(order):
        cur = (m / log_mu1.exp()) / log_mu2.exp()
        ib = imbalance(cur)

        better = (ib <= imb_min).to(dtype)
        imb_min = torch.minimum(imb_min, ib)
        mu1_star = torch.where(better.bool(), log_mu1.exp(), mu1_star)
        mu2_star = torch.where(better.bool(), log_mu2.exp(), mu2_star)

        if stop_on_increasing_imbalance:
            rising = (ib > imb_min).to(dtype)
            gate = torch.clamp(gate + rising, max=1.0)

        g = 1.0 - gate
        std_r = torch.std(cur, dim=1).clamp(clip_min, clip_max)
        std_c = torch.std(cur, dim=0).clamp(clip_min, clip_max)
        sal_col = (std_c / tgt_small).clamp(0.7, 2.0).log()
        sal_row = (std_r[:, None] / tgt_small).clamp(0.7, 2.0).log()
        log_mu1 = (log_mu1 + sal_col * g).clamp(-0.3, 10.0)
        log_mu2 = (log_mu2 + sal_row * g).clamp(-0.3, 10.0)

    scaled = m / mu1_star / mu2_star
    return scaled, mu1_star, mu2_star


def quantize_rtn(matrix, min_max):
    w = matrix.to(torch.float32)
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = min_max[1]
    min_int = min_max[0]
    scales = (max_val - min_val).clamp(min=1e-4) / max_int
    zeros = -torch.round(min_val / scales)
    q = torch.clamp(torch.round(w / scales + zeros), min_int, max_int).to(torch.int32)
    return q.contiguous(), scales, zeros


def quantize_symmetric_rtn(matrix, min_max):
    w = matrix.to(torch.float32)
    max_val = w.abs().amax(dim=1, keepdim=True)
    max_int = (min_max[1] - min_max[0]) // 2
    min_int = -max_int - 1
    scales = max_val.clamp(min=1e-5) / max_int
    q = torch.round(w / scales).clamp(min_int, max_int).to(torch.int32)
    return q, scales


def quantize_dual_scale_shift(matrix, min_max, order=16, method='sinq'):
    """
    Exact SINQ 1D tile quantization semantics:
      q, s1, s2, z such that dequant = (q - z) * s1 * s2
    """
    matrix = matrix.to(torch.float32)
    normalized, mu1, mu2 = sinkhorn_log(matrix, order=order)

    awq_scale = None
    if isinstance(method, tuple):
        method, awq_scale = method

    if 'sinq' not in method:
        normalized = normalized * mu1 * mu2
        mu1 = torch.ones_like(mu1)
        mu2 = torch.ones_like(mu2)

    if 'awq' in method and awq_scale is not None:
        awq_scale = awq_scale.to(normalized.device, normalized.dtype)
        normalized = normalized * awq_scale
        mu1 = mu1 / awq_scale

    if 'noz' in method:
        q, scales = quantize_symmetric_rtn(normalized, min_max)
        z = torch.tensor(0, dtype=torch.float32, device=matrix.device)
    else:
        q, scales, z = quantize_rtn(normalized, min_max)

    scales2 = torch.ones(1, normalized.shape[1], device=matrix.device, dtype=mu1.dtype) * mu1
    scales = scales * mu2
    return q, scales, scales2, z


def tiled_quant_rectangle(matrix, min_max, block, order=16, method='sinq', awq_scale=None):
    """
    Port of SINQ's 1D tiling path. Output ordering matches the original:
    rows are flattened as [cout, groups].
    """
    h, w = matrix.shape
    block = int(block)
    if w % block != 0 or block < 16:
        raise ValueError(f'block must divide the input dimension and be >= 16, got shape {matrix.shape}, block={block}')

    n_w = w // block
    matrix = matrix.view(h, n_w, block)
    batched = matrix.permute(1, 0, 2).contiguous().view(n_w, h, block)

    if awq_scale is not None:
        awq_scale = awq_scale.view(1, n_w, block).permute(1, 0, 2).contiguous().view(n_w, 1, block)

        def process_block(mat, block_awq_scale):
            return quantize_dual_scale_shift(mat, min_max, order=order, method=(method, block_awq_scale))

        q, s1, s2, z = torch.vmap(process_block)(batched, awq_scale)
    else:
        def process_block(mat):
            return quantize_dual_scale_shift(mat, min_max, order=order, method=method)

        q, s1, s2, z = torch.vmap(process_block)(batched)

    if 'noz' in method:
        z = z.view(-1)[0]
    else:
        z = z.permute(1, 0, 2).reshape(-1, 1)

    q = q.permute(1, 0, 2).reshape(-1, block)
    s1 = s1.permute(1, 0, 2).reshape(-1, 1)
    s2 = s2.permute(1, 0, 2).reshape(1, -1)
    return q, s1, s2, z


def _reshape_activations(activations):
    if activations is None:
        return None
    if activations.ndim < 2:
        raise ValueError('SINQ calibration activations must have at least 2 dimensions')
    return activations.reshape(-1, activations.shape[-1]).to(torch.float32)


def dequantize_tiled_params(q, scales, scales2, shifts, weight_shape, block):
    cout, cin = weight_shape
    groups = cin // block
    q = q.view(cout, groups, block).to(torch.float32)
    scales = scales.view(cout, groups, 1).to(torch.float32)
    scales2 = scales2.view(1, groups, block).to(torch.float32)
    if isinstance(shifts, torch.Tensor) and shifts.numel() > 1:
        shifts = shifts.view(cout, groups, 1).to(torch.float32)
    else:
        shift_value = float(shifts.view(-1)[0].item()) if isinstance(shifts, torch.Tensor) else float(shifts)
        shifts = torch.full((cout, groups, 1), shift_value, dtype=torch.float32, device=q.device)
    return ((q - shifts) * scales * scales2).reshape(cout, cin)


def compute_awq_scale(weight, activations, min_max, tile=64, num_alphas=20, use_weightscale=True, method='asinq'):
    activations = _reshape_activations(activations)
    activations = activations.to(weight.device, dtype=torch.float32)
    weight = weight.to(torch.float32)

    dims = tuple(range(activations.ndim - 1))
    mu_tokens = activations.abs().mean(dim=dims)
    std_tokens = activations.std(dim=dims).clamp_min(1e-4)
    mu_weights = weight.abs().mean(dim=0).clamp_min(1e-4)
    gt = activations @ weight.transpose(0, 1)

    best_scale = torch.ones_like(mu_tokens)
    best_error = None

    for alpha_idx in range(num_alphas):
        alpha = float(alpha_idx) / float(num_alphas)
        scales = mu_tokens.pow(alpha) * std_tokens.pow(0.0)
        if use_weightscale:
            scales = (scales / mu_weights.pow(1.0 - alpha)).clamp_min(1e-4)
        scales = scales / (scales.max() * scales.min()).sqrt().clamp_min(1e-4)

        q, s1, s2, z = tiled_quant_rectangle(
            weight,
            min_max=min_max,
            block=tile,
            order=16,
            method=method,
            awq_scale=scales,
        )
        dq_weight = dequantize_tiled_params(q, s1, s2, z, weight.shape, tile)
        pred = activations @ dq_weight.transpose(0, 1)
        if 'l1' in method:
            reconstruction_error = (pred - gt).abs().mean()
        else:
            reconstruction_error = (pred - gt).pow(2).mean()

        if best_error is None or reconstruction_error < best_error:
            best_error = reconstruction_error
            best_scale = scales

    return best_scale.to(weight.dtype)


def generate_sinq_params(weight, quant_config, activations=None):
    """
    Produce AMCT-native SINQ parameters by porting the original SINQ 1D tile
    quantization semantics exactly:
      dequant(weight_q) = (q - z) * scale_w * scale
    where:
      scale   -> input-dimension factor, shape [1, cin]
      scale_w -> per-group weight scale, shape [cout, groups, 1]
      offset_w -> per-group zero point, shape [cout, groups, 1]
    """
    weights_cfg = quant_config.get('weights_cfg', {})
    if weights_cfg.get('strategy') != 'group':
        raise ValueError('SINQ only supports group weight quantization in AMCT')
    quant_type = weights_cfg.get('quant_type')
    if quant_type not in ['int4', 'int8']:
        raise ValueError('SINQ only supports int4 and int8 in AMCT')

    group_size = weights_cfg.get('group_size')
    symmetric = weights_cfg.get('symmetric', False)
    sinq_cfg = quant_config.get('algorithm', {}).get('sinq', {})
    order = sinq_cfg.get('order', 16)
    method = sinq_cfg.get('method', 'sinq')
    if method == 'asinq':
        method = 'sinq'
    if activations is not None and 'awq' not in method and 'sinq' in method:
        method = method.replace('sinq', 'sinq_awq_l1', 1)
    if symmetric and 'noz' not in method:
        method = f'{method}_noz'

    if 'sinq' in str(method).lower():
        k0 = float(kurtosis(weight))
        if k0 > SINQ_KURTOSIS_THRESHOLD:
            method = 'rtn_noz' if 'noz' in method else 'rtn'

    min_max = [0, 15] if quant_type == 'int4' else [0, 255]
    offset_shift = 8 if quant_type == 'int4' else 128
    awq_scale = None
    if 'awq' in method:
        if activations is None:
            raise ValueError('Activation-assisted SINQ requires calibration activations')
        awq_scale = compute_awq_scale(
            weight.view(weight.shape[0], weight.shape[1]),
            activations,
            min_max=min_max,
            tile=group_size,
            num_alphas=sinq_cfg.get('grids_num', 20),
            method=method,
        )

    q, scale_w_flat, scale, offset_flat = tiled_quant_rectangle(
        weight.to(torch.float32),
        min_max=min_max,
        block=group_size,
        order=order,
        method=method,
        awq_scale=awq_scale,
    )

    cout = weight.shape[0]
    groups = weight.shape[1] // group_size
    scale_w = scale_w_flat.view(cout, groups, 1).contiguous()

    if symmetric:
        offset_w = torch.zeros_like(scale_w, dtype=torch.int32)
    elif isinstance(offset_flat, torch.Tensor):
        if offset_flat.numel() == 1:
            offset_w = torch.full_like(scale_w, int(offset_flat.item()) - offset_shift, dtype=torch.int32)
        else:
            offset_w = (offset_flat.view(cout, groups, 1).to(torch.int32) - offset_shift).contiguous()
    else:
        offset_w = torch.full_like(scale_w, int(offset_flat) - offset_shift, dtype=torch.int32)

    return scale.contiguous(), scale_w, offset_w, q

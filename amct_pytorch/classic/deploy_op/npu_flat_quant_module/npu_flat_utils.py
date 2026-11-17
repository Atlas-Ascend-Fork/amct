# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""W4A4 operator-chain helpers for NpuFlatQuant deploy modules.

Pieces:

- pack_int4_weights: int4 integers -> npu_quant_matmul's packed layout
- batched_kronecker_quant: npu_kronecker_quant wrapper that auto-splits
  inputs above the operator's dim0 limit (32768)
- w4a4_flatquant_linear: the per-projection forward — fused
  "transform + INT4 quant" on the activation side (left/right kron
  factors + clip_ratio) then INT4xINT4 quant matmul.
- w4a4_dynamic_linear: the o_proj special case — single matrix transform
  on the fly, npu_dynamic_quant for the activation, then quant matmul.

All are stateless functions/classes; deploy modules hold the tensors."""

import torch


def _torch_npu():
    """Resolve torch_npu at call time; deploys run only where it exists.

    No process-level caching on purpose: tests swap a fake torch_npu into
    sys.modules per case, and a cached module would ignore that swap."""
    import torch_npu

    return torch_npu


KRONECKER_QUANT_MAX_BATCH_SIZE = 32768


def reparameterize_ln(ln, trans):
    """Deprecated alias: the executor moved to the framework side
    (amct_pytorch.classic.optimizer.external_reparam) so the convert Pass
    does not import from an algorithm's deploy package. Kept here for the
    package __init__ re-export and existing callers."""
    from amct_pytorch.classic.optimizer.external_reparam import (
        reparameterize_ln as _impl,
    )

    return _impl(ln, trans)


def pack_int4_weights(weight_tensor):
    """int4 integer weights -> the packed layout npu_quant_matmul expects."""
    torch_npu = _torch_npu()

    if not torch_npu.npu.is_available():
        # CPU-side convert (e.g. unit tests / fake-quant evaluation runs):
        # keep the integer tensor as-is; packing happens on the NPU path.
        return weight_tensor.contiguous().to(torch.int32)
    original_device = weight_tensor.device
    weight_tensor_npu = weight_tensor.npu()

    weight_int4_packed = torch_npu.npu_convert_weight_to_int4pack(
        weight_tensor_npu.to(torch.int32), inner_k_tiles=1
    )
    return weight_int4_packed.to(original_device)


def batched_kronecker_quant(x, left_trans, right_trans, clip_ratio):
    """npu_kronecker_quant with automatic dim0 chunking above 32768.

    Temporary workaround for the operator's batch-size limit; remove when
    the operator supports larger inputs natively."""
    torch_npu = _torch_npu()

    # dst_dtype stays default (None -> int32 inside the operator): the
    # op schema types it as int?, and torch_npu's meta kernel only accepts
    # None or the mxfp4 enum — an explicit torch.int32 fails the fake-
    # tensor shape inference dynamo needs for AIR/OM graph export. Eager
    # semantics are identical (verified bitwise: dtype/shape/values).
    batch_tokens = x.shape[0]
    if batch_tokens <= KRONECKER_QUANT_MAX_BATCH_SIZE:
        return torch_npu.npu_kronecker_quant(
            x, left_trans, right_trans, clip_ratio=clip_ratio
        )
    x_chunks = torch.split(x, KRONECKER_QUANT_MAX_BATCH_SIZE, dim=0)
    processed_chunks = [
        torch_npu.npu_kronecker_quant(
            chunk, left_trans, right_trans, clip_ratio=clip_ratio
        )
        for chunk in x_chunks
    ]
    quantized_list, scale_list = zip(*processed_chunks)
    x_quantized_int4 = torch.cat(quantized_list, dim=0)
    activation_scale = torch.cat(scale_list, dim=0)
    return x_quantized_int4, activation_scale


def w4a4_flatquant_linear(
    left_trans,
    right_trans,
    weight_packed,
    weight_scale,
    x,
    clip_ratio=1.0,
    bias=None,
):
    """Regular projection forward: fused transform+INT4 quant on
    the activation, then INT4xINT4 quant matmul.

    left_dim*right_dim == in_features must hold (kron factor dims)."""
    original_dtype = x.dtype
    # npu_kronecker_quant / npu_quant_matmul accept only fp16/bf16 activations
    # (aclnnFlatQuant rejects DT_FLOAT with error 161002). CLIP loads fp32,
    # so cast the activation down: the whole projection (quant matmul, bias
    # add, and the returned tensor) is fp16 from here on — an fp32 CLIP model
    # ends up with an fp16 deployment output.
    if original_dtype == torch.float32:
        x = x.to(torch.float16)
        original_dtype = torch.float16
        # bias/scale handled below in the matched dtype
    input_shape = x.shape
    in_features = input_shape[-1]
    left_dim = left_trans.shape[0]
    right_dim = right_trans.shape[0]
    if left_dim * right_dim != in_features:
        raise ValueError(
            f'FlatQuant transform matrices dimension mismatch: '
            f'left_dim({left_dim}) * right_dim({right_dim}) '
            f'!= in_features({in_features})'
        )
    left_trans_matched = left_trans.to(original_dtype)
    right_trans_matched = right_trans.to(original_dtype)
    x_reshaped = x.view(-1, left_dim, right_dim)
    x_quantized_int4, activation_scale = batched_kronecker_quant(
        x_reshaped, left_trans_matched, right_trans_matched, clip_ratio
    )

    x_quantized_reshaped = x_quantized_int4.view(-1, left_dim * right_dim // 8)
    pertoken_scale = activation_scale.view(-1).to(torch.float32)
    torch_npu = _torch_npu()

    output = torch_npu.npu_quant_matmul(
        x_quantized_reshaped,
        weight_packed.t(),
        weight_scale.view(-1).to(torch.float32),
        pertoken_scale=pertoken_scale,
        bias=None,
        output_dtype=original_dtype,
    )
    output = output.view(*input_shape[:-1], -1)
    if bias is not None:
        output = output + bias.to(original_dtype)
    return output


def w4a4_dynamic_linear(x, weight_packed, weight_scale, bias=None):
    """o_proj special-case forward: the composite transform is
    already baked into the weights at convert time, so the runtime side
    only carries a single matrix (applied by the caller) — dynamic
    per-token INT4 quant, then quant matmul."""
    output_dtype = x.dtype
    # npu_dynamic_quant / npu_quant_matmul accept only fp16/bf16 activations;
    # CLIP loads fp32, cast down (error 161002 on DT_FLOAT otherwise).
    if output_dtype == torch.float32:
        x = x.to(torch.float16)
        output_dtype = torch.float16
    torch_npu = _torch_npu()

    # npu_quant_matmul rejects 3D inputs with batch >= 2 (error 161002);
    # flatten (bsz, seq, k) -> (bsz*seq, k) and restore on the way out.
    orig_shape = x.shape
    needs_reshape = x.dim() > 2
    if needs_reshape:
        x = x.reshape(-1, orig_shape[-1])
    x_quantized, pertoken_scale = torch_npu.npu_dynamic_quant(
        x, dst_type=torch.quint4x2
    )
    output = torch_npu.npu_quant_matmul(
        x_quantized,
        weight_packed.t(),
        weight_scale.view(-1).to(torch.float32),
        pertoken_scale=pertoken_scale.view(-1).to(torch.float32),
        bias=bias,
        output_dtype=output_dtype,
    )
    if needs_reshape:
        output = output.view(*orig_shape[:-1], -1)
    return output


def pad_to_kronecker_multiple(hidden_states):
    """npu_kronecker_quant requires bsz*q_len divisible by 16; pad q_len.

    Returns (padded_tensor, pad_size); the caller slices the tail off the
    outputs. No-op when already aligned."""
    bsz, q_len, _ = hidden_states.size()
    pad_size = 0
    if bsz % 16 and q_len % 16:
        pad_size = (q_len // 16 + 1) * 16 - q_len
        padding = (0, 0) * (hidden_states.dim() - 2) + (0, pad_size) + (0, 0)
        hidden_states = torch.nn.functional.pad(
            hidden_states, padding, mode='constant', value=0
        )
    return hidden_states, pad_size


def pack_projection(proj, trans):
    """From a reparameterized FlatQuantizedLinear (+ its input-side
    SVDDecomposeTransMatrix) produce the deploy-side tensors:
    (packed_weight, weight_scale, left_trans, right_trans, clip_ratio).

    quantonly=True walks the quantizer's integer path; the weights are
    already on the grid so this is a lossless extraction."""
    proj.weight_quantizer.find_params(proj.linear.weight)
    weight_int, weight_scale = proj.weight_quantizer.quantize(
        proj.linear.weight, quantonly=True
    )
    return (
        pack_int4_weights(weight_int.data),
        weight_scale,
        trans.matrix_left.T.contiguous(),
        trans.matrix_right.contiguous(),
        proj.lac_ratio.item(),
    )

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
from onnx import TensorProto
import torch

try:
    from torch.onnx import _globals
except ImportError:
    # PyTorch 2.10 moved the legacy TorchScript exporter globals.
    from torch.onnx._internal.torchscript_exporter import _globals

from ...amct_pytorch.common.utils.vars_util import RNN_TENSOR_NUM
from ...amct_pytorch.utils.vars import DEQUANTIZE_LINEAR, QUANTIZE_LINEAR


def is_dynamo_export():
    """Return whether the current call is being traced by the Dynamo exporter."""
    compiler = getattr(torch, 'compiler', None)
    is_compiling = getattr(compiler, 'is_compiling', None)
    onnx_export = getattr(torch.onnx, 'is_in_onnx_export', None)
    return bool(
        is_compiling
        and is_compiling()
        and onnx_export
        and onnx_export()
        and hasattr(torch.onnx, 'ops')
    )


def check_int4_export(wts_param):
    if not hasattr(TensorProto, 'INT4'):
        raise RuntimeError('INT4 export requires ONNX native TensorProto.INT4.')
    if _globals.GLOBALS.export_onnx_opset_version != 21:
        raise RuntimeError('INT4 Q/DQ export requires ONNX opset 21.')

    if not wts_param.get('channel_wise', False):
        return
    module = wts_param.get('module')
    scale_count = int(module.wts_scales.numel())
    out_channels = int(module.out_channels)
    if scale_count != out_channels:
        raise ValueError(
            'INT4 per-channel scale count {} must equal out_channels {}.'.format(
                scale_count, out_channels
            )
        )


def check_int4_dynamo_export(wts_param):
    """Validate INT4 prerequisites available while tracing with Dynamo."""
    if not hasattr(TensorProto, 'INT4'):
        raise RuntimeError('INT4 export requires ONNX native TensorProto.INT4.')
    if not wts_param.get('channel_wise', False):
        return
    module = wts_param.get('module')
    scale_count = int(module.wts_scales.numel())
    out_channels = int(module.out_channels)
    if scale_count != out_channels:
        raise ValueError(
            'INT4 per-channel scale count {} must equal out_channels {}.'.format(
                scale_count, out_channels
            )
        )


def add_qdq(g, tensor, scale, zero_point, num_bits, axis=None):
    """Add standard ONNX Q/DQ nodes for an AMCT QAT weight tensor."""
    attributes = {}
    if axis is not None:
        attributes['axis_i'] = axis

    if num_bits == 4:
        quant = g.op(
            QUANTIZE_LINEAR,
            tensor,
            scale,
            output_dtype_i=TensorProto.INT4,
            **attributes,
        )
        return g.op(DEQUANTIZE_LINEAR, quant, scale, **attributes)

    quant = g.op(QUANTIZE_LINEAR, tensor, scale, zero_point, **attributes)
    return g.op(DEQUANTIZE_LINEAR, quant, scale, zero_point, **attributes)


def add_qdq_dynamo(tensor, scale, zero_point, num_bits, axis=None, shape=None):
    """Build Q/DQ nodes through the Dynamo ONNX symbolic-op API."""
    onnx_ops = getattr(torch.onnx, 'ops', None)
    if onnx_ops is None or not hasattr(onnx_ops, 'symbolic'):
        raise RuntimeError('Dynamo Q/DQ export requires PyTorch 2.10 or newer.')
    if num_bits == 4 and not hasattr(TensorProto, 'INT4'):
        raise RuntimeError('INT4 export requires ONNX native TensorProto.INT4.')

    if shape is None:
        shape = tensor.shape
    quant_attrs = {'output_dtype': TensorProto.INT4} if num_bits == 4 else {}
    if axis is not None:
        quant_attrs['axis'] = axis
    quant_inputs = (tensor, scale) if num_bits == 4 else (tensor, scale, zero_point)
    quant_dtype = {
        4: TensorProto.INT4,
        8: torch.int8,
        16: torch.int16,
    }.get(num_bits)
    if quant_dtype is None:
        raise ValueError('Unsupported quantization bit width: {}'.format(num_bits))

    quant = onnx_ops.symbolic(
        QUANTIZE_LINEAR,
        quant_inputs,
        attrs=quant_attrs,
        dtype=quant_dtype,
        shape=shape,
        version=21,
    )

    dequant_attrs = {'axis': axis} if axis is not None else {}
    dequant_inputs = (quant, scale) if num_bits == 4 else (quant, scale, zero_point)
    return onnx_ops.symbolic(
        DEQUANTIZE_LINEAR,
        dequant_inputs,
        attrs=dequant_attrs,
        dtype=tensor.dtype,
        shape=shape,
        version=21,
    )


def add_weight_qdq_dynamo(
    tensor, scale, zero_point, num_bits, module_type, channel_wise, module
):
    """Build a weight Q/DQ graph using the layout expected by ONNX operators."""
    axis = 1 if channel_wise else None
    if module_type in ('ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'):
        return add_qdq_dynamo(tensor, scale, zero_point, num_bits, axis)

    if module_type in ('Conv1d', 'Conv2d', 'Conv3d'):
        rank = tensor.dim()
        perm = [1, 0] + list(range(2, rank))
        transposed = tensor.permute(perm)
        dequant = add_qdq_dynamo(transposed, scale, zero_point, num_bits, axis)
        return dequant.permute(perm)

    if module_type == 'Linear' and channel_wise:
        rank = module.weight.dim()
        perm = [1, 0] + list(range(2, rank))
        transposed = tensor.permute(perm)
        dequant = add_qdq_dynamo(transposed, scale, zero_point, num_bits, axis)
        return dequant.permute(perm)

    if module_type in ('Linear',) or module_type in RNN_TENSOR_NUM:
        return add_qdq_dynamo(tensor, scale, zero_point, num_bits, axis)

    raise RuntimeError(
        'Unsupported QAT module type for Dynamo export: {}'.format(module_type)
    )

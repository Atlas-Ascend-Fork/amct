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
from torch.onnx import _globals

from ...amct_pytorch.utils.vars import DEQUANTIZE_LINEAR, QUANTIZE_LINEAR


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

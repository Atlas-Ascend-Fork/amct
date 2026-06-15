# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

__all__ = [
    'ALGO_REGISTRY',
    'AlgorithmRegistry',
    'BUILT_IN_ALGORITHM',
]


from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant
from amct_pytorch.classic.quantize_op.linear_awq_module import LinearAWQuant
from amct_pytorch.classic.quantize_op.smooth_quant_module import SmoothQuant
from amct_pytorch.classic.quantize_op.minmax_module import MinMaxQuant
from amct_pytorch.classic.quantize_op.ofmr_quant_module import OfmrQuant
from amct_pytorch.classic.quantize_op.quantile_module import QuantileQuant
from amct_pytorch.classic.quantize_op.cast_module import HIF8CastQuant
from amct_pytorch.classic.quantize_op.deepseekv3_attention_quant_module import DeepseekV3AttentionQuant
from amct_pytorch.classic.quantize_op.longcat_flashmla_quant_module import LongcatFlashMLAQuant
from amct_pytorch.classic.deploy_op.npu_mx_quantization_linear import NpuMXQuantizationLinear
from amct_pytorch.classic.deploy_op.npu_quantization_linear import NpuQuantizationLinear
from amct_pytorch.classic.deploy_op.weight_npu_quant_module import NpuWeightQuantizedLinear
from amct_pytorch.classic.deploy_op.npu_quantization_conv2d import NpuQuantizationConv2d
from amct_pytorch.classic.deploy_op.npu_hif8_cast_quantization_linear import NpuHIF8CastLinear
from amct_pytorch.classic.deploy_op.npu_hif8_quantization_linear import NpuHIF8Linear
from amct_pytorch.classic.deploy_op.npu_quantization_deepseekv3_attention import NpuDeepseekV3AttentionQuant
from amct_pytorch.classic.deploy_op.npu_quantization_longcat_flashmla import NpuLongcatFlashMLA
from .register_algo import Algorithm
from .registry_factory import ALGO_REGISTRY

AlgorithmRegistry = Algorithm()
BUILT_IN_ALGORITHM = ['minmax', 'awq', 'gptq', 'smoothquant', 'mxquant', 'ofmr', 'cast', 'quantile']

AlgorithmRegistry.register('gptq', 'Linear', GPTQuant, NpuWeightQuantizedLinear)
AlgorithmRegistry.register('awq', 'Linear', LinearAWQuant, NpuWeightQuantizedLinear)
AlgorithmRegistry.register('smoothquant', 'Linear', SmoothQuant, NpuQuantizationLinear)
AlgorithmRegistry.register('minmax', 'Linear', MinMaxQuant, [NpuWeightQuantizedLinear, NpuQuantizationLinear])
AlgorithmRegistry.register('mxquant', 'Linear', NpuMXQuantizationLinear, NpuMXQuantizationLinear)
AlgorithmRegistry.register('ofmr', 'Linear', OfmrQuant, [NpuWeightQuantizedLinear, NpuQuantizationLinear])
AlgorithmRegistry.register('ofmr', 'Conv2d', OfmrQuant, NpuQuantizationConv2d)
AlgorithmRegistry.register('cast', 'Linear', HIF8CastQuant, NpuHIF8CastLinear)
AlgorithmRegistry.register('quantile', 'Linear', QuantileQuant, [NpuWeightQuantizedLinear, NpuQuantizationLinear])
AlgorithmRegistry.register('cast', None, 'FP8Linear', NpuHIF8Linear)
AlgorithmRegistry.register('quantile', 'DeepseekV3Attention', DeepseekV3AttentionQuant, NpuDeepseekV3AttentionQuant)
AlgorithmRegistry.register('quantile', 'LongcatFlashMLA', LongcatFlashMLAQuant, NpuLongcatFlashMLA)

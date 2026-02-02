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
    'AlgorithmRegistry',
    'BUILT_IN_ALGORITHM'
]

from amct_pytorch.algorithm.register_algo import Algorithm
from amct_pytorch.quantize_op.gptq_module import GPTQuant
from amct_pytorch.quantize_op.linear_awq_module import LinearAWQuant
from amct_pytorch.quantize_op.smooth_quant_module import SmoothQuant
from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
from amct_pytorch.quantize_op.ofmr_quant_module import OfmrQuant
from amct_pytorch.deploy_op.npu_mx_quantization_linear import NpuMXQuantizationLinear
from amct_pytorch.deploy_op.npu_quantization_linear import NpuQuantizationLinear
from amct_pytorch.deploy_op.weight_npu_quant_module import NpuWeightQuantizedLinear
from amct_pytorch.deploy_op.npu_quantization_conv2d import NpuQuantizationConv2d

AlgorithmRegistry = Algorithm()
AlgorithmRegistry.register('gptq', 'Linear', GPTQuant, NpuWeightQuantizedLinear)
AlgorithmRegistry.register('awq', 'Linear', LinearAWQuant, NpuWeightQuantizedLinear)
AlgorithmRegistry.register('smoothquant', 'Linear', SmoothQuant, NpuQuantizationLinear)
AlgorithmRegistry.register('minmax', 'Linear', MinMaxQuant, [NpuWeightQuantizedLinear, NpuQuantizationLinear])
AlgorithmRegistry.register('mxquant', 'Linear', NpuMXQuantizationLinear, NpuMXQuantizationLinear)
AlgorithmRegistry.register('ofmr', 'Linear', OfmrQuant, [NpuWeightQuantizedLinear, NpuQuantizationLinear])
AlgorithmRegistry.register('ofmr', 'Conv2d', OfmrQuant, NpuQuantizationConv2d)

BUILT_IN_ALGORITHM = ['minmax', 'awq', 'gptq', 'smoothquant', 'mxquant', 'ofmr']
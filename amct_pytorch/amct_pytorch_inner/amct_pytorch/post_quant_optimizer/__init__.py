#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

init file for post quant optimizer

"""
from amct_pytorch.optimizer.model_optimizer import ModelOptimizer
from amct_pytorch.post_quant_optimizer.insert_decoder_awq_pass import InsertDecoderAWQuantPass
from amct_pytorch.post_quant_optimizer.replace_fakequant_module_pass import ReplaceFakeQuantModulePass
from amct_pytorch.post_quant_optimizer.replace_npu_quant_pass import ReplaceNpuQuantPass

__all__ = [
    'ModelOptimizer',
    'InsertDecoderAWQuantPass',
    'ReplaceFakeQuantModulePass',
    'ReplaceNpuQuantPass'
]

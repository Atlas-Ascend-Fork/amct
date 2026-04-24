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

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import torch
import torch.nn.functional as F

from ....amct_pytorch.custom_op.comp_module.comp_module_base \
    import CompModuleBase
from ....amct_pytorch.common.utils.util import version_higher_than


class CompModuleConvTranspose2d(CompModuleBase):
    """
    Function: Quantization module class after Quantization module after ConvTranspose2d encapsulation.
    APIs: __init__, forward
    """
    def __init__(self, *args, **kwargs):
        super(CompModuleConvTranspose2d, self).__init__(*args, **kwargs)
        if not self.wts_config.get('channel_wise'):
            self.num_scales = 1
        else:
            self.num_scales = self.replaced_module.weight.size(1) * self.replaced_module.groups
        self._init_output()

    def forward(self, inputs, output_size=None):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        compressed_inputs, compressed_weights = \
            super(CompModuleConvTranspose2d, self).forward(inputs)

        # num_spatial_dims is module's kernel_size
        num_spatial_dims = 2

        # Forward
        if version_higher_than(torch.__version__, '1.12.0'):
            output_padding = self.replaced_module._output_padding(inputs, output_size, \
                self.replaced_module.stride, self.replaced_module.padding, self.replaced_module.kernel_size, \
                num_spatial_dims, self.replaced_module.dilation)
        elif version_higher_than(torch.__version__, '1.7.0'):
            output_padding = self.replaced_module._output_padding(inputs, output_size, \
                self.replaced_module.stride, self.replaced_module.padding, \
                self.replaced_module.kernel_size, self.replaced_module.dilation)
        else:
            output_padding = self.replaced_module._output_padding(inputs, output_size, \
                self.replaced_module.stride, self.replaced_module.padding, self.replaced_module.kernel_size)

        output = F.conv_transpose2d(compressed_inputs, compressed_weights, self.replaced_module.bias, \
            self.replaced_module.stride, self.replaced_module.padding, output_padding, \
            self.replaced_module.groups, self.replaced_module.dilation)

        return output

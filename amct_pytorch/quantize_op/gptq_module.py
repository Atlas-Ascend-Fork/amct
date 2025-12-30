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
import math
import torch
from torch import nn
import torch.nn.functional as F

from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.utils.data_utils import check_linear_input_dim
from amct_pytorch.quantize_op.utils import calculate_scale_offset, get_weight_min_max_by_granularity
from amct_pytorch.utils.quant_util import quant_dequant_weight
from amct_pytorch.utils.log import LOGGER


class GPTQuant(BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the GPTQuant class.
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 layer_name,
                 quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        layer_name: ori_module's name.
        quant_config: quantization parameters.
        """
        super().__init__(ori_module, layer_name, quant_config)
        self.weight = ori_module.weight
        if not torch.isfinite(self.weight.data).all():
            raise ValueError(
                "{}'s weight has invalid value, inf or nan.".format(layer_name))
        self.bias = ori_module.bias
        
        self.layer_name = layer_name
        self.quant_config = quant_config
        
        self.cur_batch = 0
        self.nsamples = 0
        self.perc_damp = 0.01
        self.block_size = 128
        self.hessian = None


    @torch.no_grad()
    def forward(self, inputs):
        """
        Function: GPTQuant forward function.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        check_linear_input_dim(inputs)
        inputs = inputs.to(self.weight.device)

        # When batch_num is reached, h stops updating
        if self.cur_batch < self.quant_config.get('batch_num'):
            self.update_hessian(inputs)
            self.cur_batch += 1

        # do gptq
        if self.cur_batch == self.quant_config.get('batch_num'):
            self.cur_batch += 1
            optimized_weight, self.scale_w, self.offset_w = self.get_opt_weight_and_quant_factor()
            self.weight.data = optimized_weight
            LOGGER.logd("Calculate gptq quant params of layer '{}' success!".format(self.layer_name), 'GPTQuant')
        return F.linear(inputs, self.weight, self.bias)


    def get_opt_weight_and_quant_factor(self):
        """
        Get optimized weights and quantization factors.

        Returns:
        tuple: Optimized weights, scale and offset for quantization.
        """
        columns = self.weight.shape[1]
        self.group_size = self.quant_config.get('weights_cfg').get('group_size')
        self.wts_type = self.quant_config.get('weights_cfg').get('quant_type')
        weight = self.weight.clone().float() # Do not alter the original weights

        self.hessian = self.hessian.to(weight.device)
        dead = torch.diag(self.hessian) == 0
        self.hessian[dead, dead] = 1
        weight[:, dead] = 0

        perm = torch.argsort(torch.diag(self.hessian), descending=True)
        invperm = torch.argsort(perm)

        # 0 calculate quantization factors
        weight_min, weight_max = get_weight_min_max_by_granularity(weight, self.quant_config)

        scale_w, offset_w = calculate_scale_offset(weight_max, weight_min, \
            self.quant_config.get('weights_cfg').get('symmetric'), self.wts_type)

        # 1 Sort based on the size of the diagonal elements of the Hessian matrix.
        weight_column_sorted = weight[:, perm]
        hessian_sorted = self.hessian[perm][:, perm]

        # 2 Calculate the inverse of the Hessian matrix
        damp = self.perc_damp * torch.mean(torch.diag(hessian_sorted))
        diag = torch.arange(columns, device=self.weight.device)
        hessian_sorted[diag, diag] += damp
        hessian_sorted = torch.linalg.cholesky(hessian_sorted)
        hessian_sorted = torch.cholesky_inverse(hessian_sorted)
        hessian_inverse = torch.linalg.cholesky(hessian_sorted, upper=True)
        del hessian_sorted, self.hessian
        import torch_npu
        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()
        # 3 Optimize weights according to the granularity of blocks.
        for i1 in range(0, columns, self.block_size):
            i2 = min(i1 + self.block_size, columns)
            count = i2 - i1

            weight_block = weight_column_sorted[:, i1:i2]
            err_block = torch.zeros_like(weight_block)
            hessian_inverse_block = hessian_inverse[i1:i2, i1:i2]

            for i in range(count):
                # Each column shares the value of a Hessian matrix.
                w = weight_block[:, i]
                d = hessian_inverse_block[i, i]

                if self.group_size:
                    column_index = perm[i1 + i]
                    scale = scale_w[:, column_index // self.group_size].unsqueeze(-1) if scale_w is not None else None
                    offset = offset_w[:, column_index // self.group_size].unsqueeze(-1) \
                        if offset_w is not None else None
                    w_q = quant_dequant_weight(w.unsqueeze(-1), self.quant_config, scale, offset).reshape(-1)
                else:
                    w_q = quant_dequant_weight(w.reshape(-1, 1), self.quant_config, scale_w, offset_w).reshape(-1)
                err = (w - w_q) / d
                weight_block[:, i:] -= err.unsqueeze(1).matmul(hessian_inverse_block[i, i:].unsqueeze(0))
                err_block[:, i] = err

            weight_column_sorted[:, i2:] -= err_block.matmul(hessian_inverse[i1:i2, i2:])

        # 4 Restore the sorting of the weight column
        weight = weight_column_sorted[:, invperm].to(self.weight.dtype)

        return weight, scale_w, offset_w


    def update_hessian(self, input_data):
        """
        Update the Hessian matrix.

        Args:
            input_data (Tensor): Input data, data used for calibration.
        """
        if len(input_data.shape) == 2:
            input_data = input_data.unsqueeze(0)
        batch_size = input_data.shape[0]

        # input_data shape is (-1, cin)
        input_data = input_data.reshape((-1, input_data.shape[-1]))
        input_data = input_data.t()
        if self.hessian is None:
            self.hessian = torch.zeros((self.weight.shape[1], self.weight.shape[1]))
        self.hessian = self.hessian.to(input_data.device)
        self.hessian *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        input_data = math.sqrt(2 / self.nsamples) * input_data.float()
        self.hessian += input_data.matmul(input_data.t())

        if not torch.isfinite(self.hessian).all():
            raise RuntimeError("{}'s hessian matrix has invalid value, inf or nan.".format(self.layer_name))
        self.hessian = self.hessian.cpu()
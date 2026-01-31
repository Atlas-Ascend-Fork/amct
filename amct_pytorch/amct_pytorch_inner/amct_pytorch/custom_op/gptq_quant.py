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
from torch import nn
import torch.nn.functional as F
import math
from amct_pytorch.utils.data_utils import check_linear_input_dim, cal_shared_exponent
from amct_pytorch.custom_op.utils import calculate_scale_offset, get_weight_min_max_by_granularity
from amct_pytorch.utils.weight_quant_utils import quant_dequant_weight


class GPTQuant(nn.Module):
    """
    Function: Customized torch.nn.Module of the GPTQuant class.
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 record_module,
                 layer_name,
                 quant_config,
                 perc_damp=0.01,
                 block_size=128):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        record_module: customized record module. To read and write.
        layer_name: ori_module's name.
        quant_config: calibration algorithm parameters.
        perc_damp: Control the degree of suppression of "significant singular values" during the quantification process
                    of weights.
        block_size: The size of the sub-blocks into which the weight matrix is divided
        """
        super().__init__()
        self.weight = ori_module.weight
        if not torch.isfinite(self.weight.data).all():
            raise ValueError(
                "{}'s weight has invalid value, inf or nan.".format(layer_name))
        self.bias = ori_module.bias
        self.post_quant_record_module = record_module
        self.layer_name = layer_name
        self.quant_config = quant_config
        self.cur_batch = 0
        self.hessian = torch.zeros((self.weight.shape[1], self.weight.shape[1]),
                              device=self.weight.device)
        self.nsamples = 0
        self.perc_damp = perc_damp
        self.block_size = block_size


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
            self.update_H(inputs)
            self.cur_batch += 1

        # do gptq
        if self.cur_batch == self.quant_config.get('batch_num'):
            self.cur_batch += 1
            optimized_weight, scale_w, offset_w = self.get_opt_weight_and_quant_factor()
            self.save_record(optimized_weight, scale_w, offset_w)

        return F.linear(inputs, self.weight, self.bias)


    def get_opt_weight_and_quant_factor(self):
        """
        Get optimized weights and quantization factors.

        Returns:
        tuple: Optimized weights, scale and offset for quantization.
        """
        columns = self.weight.shape[1]
        group_size = self.quant_config.get('group_size')
        wts_type = self.quant_config.get('wts_type')
        weight = self.weight.clone().float() # Do not alter the original weights

        dead = torch.diag(self.hessian) == 0
        self.hessian[dead, dead] = 1
        weight[:, dead] = 0

        perm = torch.argsort(torch.diag(self.hessian), descending=True)
        invperm = torch.argsort(perm)

        # 0 calculate quantization factors
        scale_w = None
        offset_w = None

        # MXFP4 is directly converted without quantization factors
        if wts_type not in ('MXFP4_E2M1',):
            weight_min, weight_max = get_weight_min_max_by_granularity(weight, self.quant_config)

            scale_w, offset_w = calculate_scale_offset(weight_max, weight_min, \
                self.quant_config.get('wts_asymmetric'), wts_type)
        else:
            # scale_w shape is (cout, cin//group_size, 1)
            scale_w = cal_shared_exponent(weight, "MXFP4_E2M1")

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
        # 3 Optimize weights according to the granularity of blocks.
        for i1 in range(0, columns, self.block_size):
            i2 = min(i1 + self.block_size, columns)
            count = i2 - i1

            weight_block = weight_column_sorted[:, i1:i2]
            err_block = torch.zeros_like(weight_block)
            H_inverse_block = hessian_inverse[i1:i2, i1:i2]

            for i in range(count):
                # Each column shares the value of a Hessian matrix.
                w = weight_block[:, i]
                d = H_inverse_block[i, i]

                if group_size:
                    column_index = perm[i1+i]
                    scale = scale_w[:, column_index//group_size].unsqueeze(-1) if scale_w is not None else None
                    offset = offset_w[:, column_index//group_size].unsqueeze(-1) if offset_w is not None else None
                    w_q = quant_dequant_weight(w.unsqueeze(-1), self.quant_config, scale, offset).reshape(-1)
                else:
                    w_q = quant_dequant_weight(w.reshape(-1, 1), self.quant_config, scale_w, offset_w).reshape(-1)
                err = (w - w_q) / d
                weight_block[:, i:] -= err.unsqueeze(1).matmul(H_inverse_block[i, i:].unsqueeze(0))
                err_block[:, i] = err

            weight_column_sorted[:, i2:] -= err_block.matmul(hessian_inverse[i1:i2, i2:])

        # 4 Restore the sorting of the weight column
        weight = weight_column_sorted[:, invperm].to(self.weight.dtype)

        return weight, scale_w, offset_w

    def save_record(self, optimized_weight, scale_w, offset_w):
        """
        Save Quantification Factor
        Args:
        optimized_weight (Tensor): Optimized weight, used for calibration.
        scale_w (Tensor): Scale factor for the weight.
        offset_w (Tensor): Offset for the weight.
        """
        quant_params = {
            self.layer_name[0]: {
                'optimized_weight': optimized_weight.detach().cpu()
            }
        }

        # Except for MXFP4, all have quantization factors.
        if self.quant_config.get('wts_type') not in ('MXFP4_E2M1',):
            quant_factor = {
                "quant_factors": {
                    'scale_w': scale_w.detach().cpu(),
                    'offset_w': offset_w.detach().cpu()
                }
            }
            quant_params[self.layer_name[0]].update(quant_factor)

        quant_info = {'wts_type': self.quant_config.get('wts_type'),
                      'weight_compress_only': True,
                      'round_mode': self.quant_config.get("round_mode")}

        if self.quant_config.get("group_size"):
            quant_info["group_size"] = self.quant_config.get("group_size")

        self.post_quant_record_module(self.layer_name, quant_info, quant_params)


    def update_H(self, input_data):
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

        self.hessian *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        input_data = math.sqrt(2 / self.nsamples) * input_data.float()
        self.hessian += input_data.matmul(input_data.t())

        if not torch.isfinite(self.hessian).all():
            raise RuntimeError("{}'s hessian matrix has invalid value, inf or nan.".format(self.layer_name))

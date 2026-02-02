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
import numpy as np

import amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer as opt
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.custom_op.ada_round_quant import AdaRoundQuant
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.configuration.configuration import Configuration
from ...amct_pytorch.custom_op.ada_round_quant import GAMMA, ZETA, ANNEAL_COEFFICIENT
from ...amct_pytorch.ada_round.ada_round_data_manager import AdaRoundDataManager


def replace_adaround_module(object_module_name, model, data_tensor, tensor_balance_factor):
    '''
    Function: Replace the adaround module.
    Parameters:
        object_module_name: module to be replaced.
        model: torch.nn model.
        data_tensor: weight after dmq and ConvTranspose2d pre-processing.
        tensor_balance_factor: dmq factor
    Return:
        Complete the replacement module.
    '''
    conf = Configuration()

    # Step1: get module by name
    model_helper = ModuleHelper(model)
    module = model_helper.get_module(object_module_name)

    # Step2: generate new model
    wts_param = conf.get_quant_config()[object_module_name].get('weight_quant_params')
    new_ada_module = AdaRoundQuant(module, wts_param, data_tensor, tensor_balance_factor)

    # Step3: replace new model
    model_helper.replace_module_by_name(model, object_module_name, new_ada_module)
    LOGGER.logd(
        "replace AdaRoundQuant module to '{}' success!".format(
            object_module_name))
 
    return new_ada_module


def set_ada_module_model(module_helper, groups, optimize_switch):
    '''
    Function: Sets the control forward propagation flag bit.
    Parameters:
        module_helper: Helper for torch.nn.module.
        groups: layer to be quantized.
        optimize_switch: bool.
    Return:
        None.
    '''
    for group in groups:
        ada_module = module_helper.get_module(group[0])
        ada_module.ada_round_optimize = optimize_switch


def optimize_alpha(model, model_all_input, groups, graph):
    '''
    Function: Optimized the alpha in ada round module
    Parameters:
        model: torch.nn model.
        model_all_input: data used for training.
        groups: layer to be quantized.
        graph: inner graph.
    Return:
        None.
    '''
    if not groups:
        return
    device = next(model.parameters()).device

    model.train()
    sampler = AdaRoundDataManager(model)
    module_helper = ModuleHelper(model)
    for group in groups:
        ada_module = module_helper.get_module(group[0])
        # input data from quantize model and output data from original model
        set_ada_module_model(module_helper, groups, False)
        output_data_o = sampler.get_output_data(model_all_input, group[0])
        act_module = group[1] # activation function
        if act_module:
            act_module.to(output_data_o.device)
            output_data_o = act_module(output_data_o)

        set_ada_module_model(module_helper, groups, True)
        input_data_q = sampler.get_input_data(model_all_input, group[0])
        
        optimizer = torch.optim.Adam([ada_module.alpha])

        wts_param = ada_module.wts_param
        num_iteration = wts_param.get('num_iteration')
        for iteration in range(num_iteration):
            indices = torch.randperm(input_data_q.size(0))[:32]
            input_data_rand = input_data_q[indices].to(device)
            output_data_rand = output_data_o[indices].to(device)
 
            optimizer.zero_grad()
            x = input_data_rand
            x = ada_module.forward(x)
            if act_module:
                x = act_module(x)
            recon_loss = _compute_recon_loss(x, output_data_rand)
            round_loss = _compute_round_loss(ada_module.alpha, wts_param, iteration)
            total_loss = recon_loss + round_loss
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    "{}'s activation quant loss has invalid value, inf or nan. "
                    "Please check activation value.".format(group[0]))
            total_loss.backward()
            optimizer.step()

        LOGGER.logd('Do layer \'{}\' weights AdaRound fine-tuning success!'
                    .format(group[0]))

    model.eval()
    set_ada_module_model(module_helper, groups, False)

    # Set Weight and Set the adaround module back
    for group in groups:
        ada_module = module_helper.get_module(group[0])
        calied_weight = ada_module.get_quantize_weight()

        # set calied_weight to module
        ada_module.module.weight.data = calied_weight

        # set calied_weight to graph
        object_node = graph.get_node_by_name(group[0])
        opt.WeightsCalibrationPass._graph_weight_set_process(object_node=object_node,
                                                            object_module=ada_module.module,
                                                            weight=ada_module.module.weight.data)
        # Set the adaround module back
        ModuleHelper.replace_module_by_name(model, group[0], ada_module.module)


def _compute_recon_loss(ada_quantized_output, original_output):
    '''
    Function: Compute Reconstruction Loss using Squared Frobenius Norm - first part of Combined Loss.
    Parameters:
        ada_quantized_output: Activation output from quantized wrapper module.
        original_output: Activation output from original module.
    Return:
        recon_loss: reconstruction loss.
    '''
    frobenius_norm = torch.norm(ada_quantized_output - original_output, p="fro", dim=1)
    recon_loss = frobenius_norm.pow(2).mean()

    return recon_loss


def _compute_beta(max_iter, cur_iter, beta_range, warm_start):
    '''
    Function: Compute beta parameter used in regularization function using cosine decay.
    Parameters:
        max_iter: total maximum number of iterations.
        cur_iter: current iteration.
        beta_range: range for beta decay (initial_beta, end_beta).
        warm_start: warm up period, during which rounding loss has zero effect.
    Return:
        beta: parameter beta.
    '''
    # iteration marking the conclusion of the warm-up phase
    warm_initial_end_iter = warm_start * max_iter

    # calculate the relative position of the current iteration
    # Ensure that iter_coefficient is greater than 0
    rel_iter = (cur_iter - warm_initial_end_iter) / (max_iter - warm_initial_end_iter)
    iter_coefficient = 1 + np.cos(rel_iter * np.pi)

    initial_beta, end_beta = beta_range
    beta = end_beta + ANNEAL_COEFFICIENT * (initial_beta - end_beta) * iter_coefficient

    return beta


def _compute_round_loss(alpha, opt_params, cur_iter):
    '''
    Function: Compute Rounding Loss - second part of Combined Loss.
    Parameters:
        alpha: parameter 'alpha' to be optimized, float32 tensor same shape as weight tensor.
        opt_params: Optimization parameters for Adaround.
        cur_iter: current iteration.
    Return:
        round_loss: rounding loss.
    '''
    round_loss = 0
    if cur_iter >= opt_params.get('num_iteration') * opt_params.get('warm_start'):
        # calculate the rectified sigmoid function of the parameter 'alpha' to map it to the range between 0 and 1
        h_alpha = torch.clamp(torch.sigmoid(alpha) * (ZETA - GAMMA) + GAMMA, 0, 1)
 
        # calculate beta parameter
        beta = _compute_beta(opt_params.get('num_iteration'), cur_iter, 
                             tuple(opt_params.get('beta_range')), opt_params.get('warm_start'))
 
        # calculate the regularization term, Convergence of parameters to 0 or 1
        term = -(torch.add(2 * h_alpha, -1).abs()).pow(beta)
        reg_term = torch.add(1, term).sum()
 
        round_loss = opt_params.get('reg_param') * reg_term
 
    return round_loss


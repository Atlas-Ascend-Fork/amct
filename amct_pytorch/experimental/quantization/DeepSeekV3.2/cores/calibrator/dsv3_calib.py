# coding=utf-8
# Adapted from
# https://github.com/ruikangliu/FlatQuant/blob/main/flatquant/train_utils.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2024 ruikangliu. Licensed under MIT.
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

import os
import time
import gc
import functools
from loguru import logger
import torch

from cores.calibrator.function_utils import set_require_grad_all, get_n_set_parameters_byname, check_params_grad
from cores.quantization.node import WeightQuantizer, ActivationQuantizer
from cores.models.deepseek_v3_2.modeling_deepseek_v3_2 import _prepare_4d_causal_attention_mask
from cores.models.deepseek_v3_2.quant_utils import QuantLinear


def set_model_to_observe(model, flag):
    for name, mod in model.named_modules():
        if isinstance(mod, ActivationQuantizer):
            logger.info(f"set {name} to observe {flag}")
            mod.is_observe = flag


def set_model_weight_quant_state(model, flag):
    for name, mod in model.named_modules():
        if isinstance(mod, WeightQuantizer):
            mod.enable = flag


def forward_once(layer, fp_inps, seq_length, index, args):
    if args.train_mode == "mla":
        attention_mask = None
        attention_mask_batch = _prepare_4d_causal_attention_mask(
            attention_mask,
            (args.cali_bsz, seq_length),
            torch.randn_like(fp_inps),
            0,
        )
        out = layer(fp_inps[index:index + args.cali_bsz, ], attention_mask=attention_mask_batch)[0]
    else:
        out = layer(fp_inps[index:index + args.cali_bsz, ])
    return out


def cali_quant(args, layer, layer_idx, inps, dev, param_prefix=None):
    layer.eval()

    # check trainable parameters
    for name, param in layer.named_parameters():
        param.requires_grad = False

    dtype = torch.bfloat16
    traincast = functools.partial(torch.amp.autocast, device_type="npu", dtype=dtype)

    # same input of first layer for fp model and quant model
    fp_inps = inps
    seq_length = fp_inps.shape[1]
    args.nsamples = fp_inps.shape[0]

    loss_func = torch.nn.MSELoss()

    # start training
    dtype_dict = {}

    for name, param in layer.named_parameters():
        dtype_dict[name] = param.dtype

    layer.to(dev)
    fp_inps = fp_inps.to(dev)

    with torch.no_grad():
        layer.float()
        fp_inps = fp_inps.float()

    set_model_to_observe(layer, True)
    set_model_weight_quant_state(layer, False)

    fp_outs = torch.zeros_like(fp_inps)
    with torch.no_grad():
        for j in range(args.nsamples // args.cali_bsz):
            index = j * args.cali_bsz
            fp_outs[index:index + args.cali_bsz, ] = forward_once(layer, fp_inps, seq_length, index, args)

    set_model_to_observe(layer, False)
    set_model_weight_quant_state(layer, True)
    torch.npu.empty_cache()

    set_require_grad_all(layer, False)
    trained_params, paras_name = [], []

    if args.lwc:
        trained_params.append(
            {"params": get_n_set_parameters_byname(layer, ["clip_factor_w", ]), "lr": args.base_lr * 10})
        paras_name.append("clip_factor_w")
    if args.lac:
        trained_params.append(
            {"params": get_n_set_parameters_byname(layer, ["clip_factor_a", ]), "lr": args.base_lr * 10})
        paras_name.append("clip_factor_a")

    for name, param in layer.named_parameters():
        if param in trained_params:
            param.requires_grad = True

    optimizer = torch.optim.AdamW(trained_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs * (args.nsamples // args.cali_bsz),
                                                           eta_min=args.base_lr * 1e-3)

    check_params_grad(layer)

    for epoch in range(args.epochs):
        mse = 0
        start_tick = time.time()
        with traincast():

            for j in range(args.nsamples // args.cali_bsz):
                index = j * args.cali_bsz
                quant_out = forward_once(layer, fp_inps, seq_length, index, args)

                loss = loss_func(fp_outs[index:index + args.cali_bsz, ], quant_out)
                mse += loss.detach().cpu()
                loss = loss / loss.clone().detach()

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()

        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f"lwc lac iter {epoch}, lr {cur_lr:.8f} time {time.time() - start_tick:.6f}s, mse: {mse:.8f}")

    layer.to("cpu")

    # save scale after deploy
    for name, mod in layer.named_modules():
        if isinstance(mod, ActivationQuantizer):
            mod.deploy()

    quant_parameters = {}
    for name, mod in layer.named_modules():
        mod_name = f"{param_prefix}.{name}" if param_prefix is not None else name
        if isinstance(mod, ActivationQuantizer):
            if mod.is_per_tensor:
                quant_parameters[f"{mod_name}.scale"] = mod.scale
                quant_parameters[f"{mod_name}.zero"] = mod.zero
                quant_parameters[f"{mod_name}.maxval"] = mod.maxval
                quant_parameters[f"{mod_name}.minval"] = mod.minval
            if hasattr(mod, 'clip_factor_a_min'):
                quant_parameters[f"{mod_name}.clip_factor_a_min"] = mod.clip_factor_a_min
                quant_parameters[f"{mod_name}.clip_factor_a_max"] = mod.clip_factor_a_max
        if isinstance(mod, QuantLinear):
            if hasattr(mod, 'clip_factor_w_max'):
                quant_parameters[f"{mod_name}.clip_factor_w_max"] = mod.clip_factor_w_max
                quant_parameters[f"{mod_name}.clip_factor_w_min"] = mod.clip_factor_w_min

    logger.info(quant_parameters.keys())
    if args.train_mode == "mla":
        torch.save(quant_parameters, os.path.join(args.exp_dir, f"quant_parameters_{layer_idx}.pth"))
        logger.info("saved parameters at {}".format(os.path.join(args.exp_dir, f"quant_parameters_{layer_idx}.pth")))

    for name, param in layer.named_parameters():
        param.requires_grad = False
        if name in dtype_dict.keys():
            param.data = param.to(dtype_dict[name])

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.npu.empty_cache()

    return quant_parameters

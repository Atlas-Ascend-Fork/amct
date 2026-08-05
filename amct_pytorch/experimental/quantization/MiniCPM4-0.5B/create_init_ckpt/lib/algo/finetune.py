# Adapted from QuIP# (Cornell-RelaxML/quip-sharp), licensed under GPLv3.
# Source: https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/algo/finetune.py
# Modified for MiniCPM quantization on Ascend NPU.
"""
Utilities for fine tuning
"""
import copy
import os
import time
import traceback
from operator import attrgetter

import glog
import torch
from torch import nn

from lib import codebook, utils
from lib.linear import *
from lib.utils import matmul_had

from . import quip

# Env-gated Stage2 finetune profiler (HARNESS_HAD_PROFILE=1): break down a
# layer's finetune wall time into forward+loss / backward / hadamard (the last
# read from matmul_had's chokepoint counter). Used to decide whether the Stage2
# fallback bottleneck is hadamard (=> native kernel is the fix) or elsewhere.
_FT_PROFILE = os.environ.get("HARNESS_HAD_PROFILE", "0") == "1"


def _ft_sync():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, position_embeddings, position_ids, args):
    layer = layer.to(device)

    susv_params, params = utils.extract_susv_params(layer)
    optim = utils.get_susv_adam(susv_params, params, args)
    best_loss = utils.calculate_mse_loss(layer, valid_dl, device=device, position_embeddings=position_embeddings, position_ids=position_ids)
    best_sd = copy.deepcopy(layer.state_dict())
    glog.info(f'layer {name} initial loss {best_loss}')
    device_type = device.type if isinstance(device, torch.device) else device.split(':')[0]
    scaler = torch.amp.GradScaler(device_type)
    worse_ct = 0
    batch_position_embeddings = None
    batch_position_ids = None
    if _FT_PROFILE:
        matmul_had.had_profile_reset()
        _prof = {"fwd": 0.0, "bwd": 0.0, "nbatch": 0}
        _t_layer0 = time.perf_counter()
    for epoch in range(args.ft_epochs):
        for bidx, (source, targets) in enumerate(train_dl):
            if _FT_PROFILE:
                _ft_sync(); _tf = time.perf_counter()
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=True):
                layer_kwargs = {
                    "hidden_states": source.to(device)
                }
                if position_embeddings is not None:
                    if batch_position_embeddings is None:
                        batch_position_embeddings = (position_embeddings[0][:args.ft_bs], position_embeddings[1][:args.ft_bs])
                    layer_kwargs["position_embeddings"] = batch_position_embeddings
                else:
                    if batch_position_ids is None:
                        batch_position_ids = position_ids[:args.ft_bs]
                    layer_kwargs["position_ids"] = batch_position_ids
                output = layer(**layer_kwargs)[0]
                loss = nn.MSELoss()(output, targets.to(device=device, dtype=output.dtype))
            if _FT_PROFILE:
                _ft_sync(); _prof["fwd"] += time.perf_counter() - _tf; _tb = time.perf_counter()
            scaler.scale(loss).backward()
            if _FT_PROFILE:
                _ft_sync(); _prof["bwd"] += time.perf_counter() - _tb; _prof["nbatch"] += 1
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_mse_loss(layer, valid_dl, device=device, position_embeddings=position_embeddings, position_ids=position_ids)
            if test_loss < best_loss:
                glog.info(f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER')
                best_loss = test_loss
                best_sd = copy.deepcopy(layer.state_dict())
                worse_ct = 0
            else:
                glog.info(f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE')
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    if _FT_PROFILE:
        _ft_sync()
        total = time.perf_counter() - _t_layer0
        had = matmul_had.had_profile_stats()
        nb = max(_prof["nbatch"], 1)
        glog.info(
            f"[ft-profile] layer {name}: total={total:.1f}s nbatch={_prof['nbatch']} "
            f"fwd={_prof['fwd']:.1f}s ({100*_prof['fwd']/total:.0f}%) "
            f"bwd={_prof['bwd']:.1f}s ({100*_prof['bwd']/total:.0f}%) "
            f"hadamard={had['time']:.1f}s ({100*had['time']/total:.0f}%, "
            f"calls={had['count']}, {1000*had['time']/max(had['count'],1):.2f}ms/call) "
            f"other={total-_prof['fwd']-_prof['bwd']:.1f}s "
            f"| per-batch fwd={1000*_prof['fwd']/nb:.0f}ms bwd={1000*_prof['bwd']/nb:.0f}ms"
        )

    del optim, train_dl, valid_dl

    layer.load_state_dict(best_sd)
    utils.clean()
    layer = layer.cpu()


def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args, device, pre_orig_emb, orig_emb, position_embeddings, position_ids, hessian_data=None):
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)

    codebook_id = codebook.get_id(args.codebook)
    mixed_layer = mixed_layer.float()

    train_dl, valid_dl = utils.split_data(X=pre_orig_emb, Y=orig_emb, args=args)

    shared_args = (cb.codesz, cb.packsz, cb.pack_out, str(cb.idx_dtype), cb.version)
    shared_kwargs = {
        'rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'resid_scale_override': args.resid_scale_override,
        'grad_ckpt': args.ft_grad_ckpt,
        'codebook_id': codebook_id
    }

    for _, (linear_attr, group_name) in enumerate(quant_order):
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        save_path = f'{args.save_path}/{idx}_{linear_attr}.pt'
        with torch.no_grad():
            weights = [orig_linear.weight]
            if hessian_data is None:
                hessian_path = f'{args.hessian_path}/{idx}_{group_name}.pt'
                quip.quantize_linear(weights, save_path, hessian_path, cb, args, device, bias=orig_linear.bias)
            else:
                quip.quantize_linear_with_hessian(weights, save_path, hessian_data[group_name], cb, args, device, bias=orig_linear.bias)
            saved_linear = torch.load(save_path, map_location=torch.device('cpu'), weights_only=True)
            shared_kwargs['bias'] = (orig_linear.bias is not None)
            quant_linear = QuantizedLinear(
                saved_linear['shapes'][0][1],
                saved_linear['shapes'][0][0],
                *shared_args, **shared_kwargs
            )
            utils.unpack_quip(quant_linear, saved_linear, codebook_id, cb.codesz)
        quant_linear.SU = nn.Parameter(quant_linear.SU.float(), requires_grad=True)
        quant_linear.SV = nn.Parameter(quant_linear.SV.float(), requires_grad=True)
        if "tuneable" in args.codebook:
            quant_linear.codebook_class.codebook.linear_proj.to(dtype=torch.float)
        split_attr = linear_attr.split('.')
        setattr(attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1], quant_linear)

    with torch.enable_grad():
        finetune_decoder_layer(
            layer=mixed_layer, 
            name=f'{idx}_{group_name}', 
            device=device, 
            train_dl=train_dl, 
            valid_dl=valid_dl, 
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            args=args
        )

    with torch.no_grad():
        utils.clean()
        for linear_attr, group_name in quant_order:
            utils.save_susv(attrgetter(linear_attr)(mixed_layer), f'{args.save_path}/{idx}_{linear_attr}.pt', save_tuneable = ("tuneable" in args.codebook))

    # mixed_layer = mixed_layer.to(torch.float16).cpu()
    utils.clean()
    torch.set_grad_enabled(False)

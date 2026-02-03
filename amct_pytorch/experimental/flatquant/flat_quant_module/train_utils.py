import time
import gc
import functools
import torch
import torch_npu
import torch.nn as nn
import transformers

from .function_utils import set_require_grad_all, get_n_set_parameters_byname
from .config_utils import InnerConfig


def cali_flat_quant(model, dataloader, dev, logger):
    """
    The parameters related to calibration (e.g. epochs, batch size, learning rate) can be adjusted in the "Calibration" section in the InnerConfig enum
    """
    # TODO: maybe standardize the configuration procedure for calibration in the future (also for other quant methods)
    model.eval()
    model.cpu()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # set trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # activate AMP
    dtype = torch.bfloat16
    traincast = functools.partial(torch_npu.npu.amp.autocast, dtype=dtype)

    # move embedding layer and first layer to target device
    layers = model.model.layers
    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    # catch the first layer input
    inps = []
    layer_kwargs = {}
    def layer_input_data_hook(module, args, kwargs):
        inps.append(args[0])
        layer_kwargs.update(kwargs)
        raise ValueError('early exit to break later interface')

    handle = layers[0].register_forward_pre_hook(layer_input_data_hook, with_kwargs=True)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= InnerConfig.nsamples.value:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass
    handle.remove()

    batch_kwargs = layer_kwargs.copy()
    if batch_kwargs['attention_mask'] is not None:
        batch_kwargs['attention_mask'] = batch_kwargs['attention_mask'].repeat(InnerConfig.cali_bsz.value, 1, 1, 1).float()
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.npu.empty_cache()

    # same input of first layer for fp model and quant model
    fp_inps = torch.stack(inps)
    fp_outs = torch.zeros_like(fp_inps)

    loss_func = torch.nn.MSELoss()
    # start training
    num_train_layer = len(layers)
    for i in range(num_train_layer):
        logger.info(f"========= Layer {i} =========")
        dtype_dict = {}
        layer = layers[i].to(dev)
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        # original forward to get output data & ln data
        layer.self_attn._ori_mode = True
        layer.mlp._ori_mode = True
        with torch.no_grad():
            for j in range(InnerConfig.nsamples.value):
                fp_outs[j] = layer(fp_inps[j].unsqueeze(0), **layer_kwargs)[0]
        layer.self_attn._ori_mode = False
        layer.mlp._ori_mode = False

        # init diag scale with ln data
        layer.self_attn.init_diag_scale(alpha=InnerConfig.diag_alpha.value)
        layer.mlp.init_diag_scale(alpha=InnerConfig.diag_alpha.value)

        set_require_grad_all(layer, False)
        trained_params, paras_name = [], []
        if InnerConfig.cali_trans.value:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.linear", ]), "lr": InnerConfig.flat_lr.value})
            paras_name.append("trans.linear")
        if InnerConfig.add_diag.value:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.diag_scale", ]), "lr": InnerConfig.flat_lr.value})
            paras_name.append("trans.diag_scale")
        if InnerConfig.lwc.value:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_w", ]), "lr": InnerConfig.flat_lr.value * 10})
            paras_name.append("clip_factor_w")
        if InnerConfig.lac.value:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_a", ]), "lr": InnerConfig.flat_lr.value * 10})
            paras_name.append("clip_factor_a")

        optimizer = torch.optim.AdamW(trained_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
            T_max=InnerConfig.epochs.value * (InnerConfig.nsamples.value // InnerConfig.cali_bsz.value),
            eta_min=InnerConfig.flat_lr.value * 1e-3)

        for epoch in range(InnerConfig.epochs.value):
            mse = 0
            start_tick = time.time()
            with traincast():
                for j in range(InnerConfig.nsamples.value // InnerConfig.cali_bsz.value):
                    index = j * InnerConfig.cali_bsz.value
                    quant_out = layer(fp_inps[index:index+InnerConfig.cali_bsz.value,], **batch_kwargs)[0]
                    loss = loss_func(fp_outs[index:index+InnerConfig.cali_bsz.value,], quant_out)
                    mse += loss.detach().cpu()
                    loss = loss / loss.clone().detach()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(f"layer {i} lwc lac iter {epoch}, lr {cur_lr:.8f}  time {time.time() - start_tick:.6f}s, mse: {mse:.8f}" )

        fp_inps, fp_outs = fp_outs, fp_inps
        layers[i] = layer.to("cpu")
        for name, param in layer.named_parameters():
            param.requires_grad = False
            if name in dtype_dict.keys():
                param.data = param.to(dtype_dict[name])
        del layer
        torch.npu.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.npu.empty_cache()
    model.config.use_cache = use_cache
    return model

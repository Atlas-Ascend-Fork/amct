import time
import gc
import functools
import torch
import torch_npu

from .function_utils import set_require_grad_all, get_n_set_parameters_byname
from .default_model_utils import FlatQuantAttention, FlatQuantMLP


def cali_flat_quant(model, dataloader, dev):
    """
    The parameters related to calibration (e.g. epochs, batch size, learning rate) 
    """
    # TODO: maybe standardize the configuration procedure for calibration in the future (also for other quant methods)
    model.eval()
    model.cpu()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # set trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

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
        inps.append(args[0].squeeze(0))
        layer_kwargs.update(kwargs)
        raise ValueError('early exit to break later interface')

    handle = layers[0].register_forward_pre_hook(layer_input_data_hook, with_kwargs=True)

    n_samples = len(dataloader)
    with torch.no_grad():
        for batch in dataloader:
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass
    handle.remove()

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.npu.empty_cache()

    # same input of first layer for fp model and quant model
    fp_inps = torch.stack(inps)
    fp_outs = torch.zeros_like(fp_inps)

    # start training
    for i in range(len(layers)):
        print(f"========= Layer {i} =========")
        dtype_dict = {}
        layer = layers[i].to(dev)
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        training_flag = False
        if isinstance(layer.self_attn, FlatQuantAttention) or isinstance(layer.mlp, FlatQuantMLP):
            training_flag = True

        # origin forward
        if training_flag:
            layer.self_attn._ori_mode = True
            layer.mlp._ori_mode = True

        with torch.no_grad():
            for j in range(n_samples):
                fp_outs[j] = layer(fp_inps[j].unsqueeze(0), **layer_kwargs)[0]
        
        # calibration layer by layer
        if training_flag:
            layer.self_attn._ori_mode = False
            layer.mlp._ori_mode = False
            cali_layer(layer, fp_inps, fp_outs, layer_kwargs)

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


def cali_layer(layer, fp_inps, fp_outs, layer_kwargs):
    n_samples = len(fp_inps)

    # activate AMP
    dtype = torch.bfloat16
    traincast = functools.partial(torch_npu.npu.amp.autocast, dtype=dtype)
    loss_func = torch.nn.MSELoss()

    # init diag scale with ln data
    if hasattr(layer.self_attn, 'init_diag_scale'):
        layer.self_attn.init_diag_scale(alpha=layer.self_attn.flat_config.diag_alpha)
    if hasattr(layer.mlp, 'init_diag_scale'):
        layer.mlp.init_diag_scale(alpha=layer.self_attn.flat_config.diag_alpha)

    set_require_grad_all(layer, False)
    trained_params, paras_name = [], []
    if layer.self_attn.flat_config.cali_trans:
        trained_params.append(
            {"params": get_n_set_parameters_byname(layer, ["trans.linear", ]), 
            "lr": layer.self_attn.flat_config.flat_lr})
        paras_name.append("trans.linear")
    if layer.self_attn.flat_config.add_diag:
        trained_params.append(
            {"params": get_n_set_parameters_byname(layer, ["trans.diag_scale", ]), 
            "lr": layer.self_attn.flat_config.flat_lr})
        paras_name.append("trans.diag_scale")
    if layer.self_attn.flat_config.lwc:
        trained_params.append(
            {"params": get_n_set_parameters_byname(layer, ["clip_factor_w", ]), 
            "lr": layer.self_attn.flat_config.flat_lr * 10})
        paras_name.append("clip_factor_w")
    if layer.self_attn.flat_config.lac:
        trained_params.append(
            {"params": get_n_set_parameters_byname(layer, ["clip_factor_a", ]), 
            "lr": layer.self_attn.flat_config.flat_lr * 10})
        paras_name.append("clip_factor_a")

    bsz = layer.self_attn.flat_config.cali_bsz
    batch_kwargs = layer_kwargs.copy()
    if batch_kwargs['attention_mask'] is not None:
        batch_kwargs['attention_mask'] = batch_kwargs['attention_mask'].repeat(bsz, 1, 1, 1).float()

    optimizer = torch.optim.AdamW(trained_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=layer.self_attn.flat_config.epochs * (n_samples // bsz), 
        eta_min=layer.self_attn.flat_config.flat_lr * 1e-3)

    for epoch in range(layer.self_attn.flat_config.epochs):
        mse = 0
        start_tick = time.time()
        with traincast():
            for j in range(n_samples // bsz):
                index = j * bsz
                quant_out = layer(fp_inps[index: index + bsz, ], **batch_kwargs)[0]
                loss = loss_func(fp_outs[index: index + bsz, ], quant_out)
                mse += loss.detach().cpu()
                loss = loss / loss.clone().detach()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f"layer {layer.self_attn.layer_idx} lwc lac iter {epoch}, "
            f"lr {cur_lr:.8f}  time {time.time() - start_tick:.6f}s, mse: {mse:.8f}")

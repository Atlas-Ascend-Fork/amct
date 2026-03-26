import os
import torch
import logging

from .function_utils import get_paras_dict_by_name


def kronecker_matmul(x, hadL, hadR):
    """equivalent to
    
        had = torch.kron(hadL, hadR)
        x = x.reshape(-1, had.shape[0])
        x = x.matmul(had).reshape(init_shape)
    """
    init_shape = x.shape
    had = torch.kron(hadL, hadR)
    x = x.reshape(-1, had.shape[0])
    x = x.matmul(had)
    return x.reshape(init_shape)


def reparameterize_ln(ln, trans):
    ln_weight = ln.weight.data
    ori_dtype = ln_weight.dtype
    ln_weight = ln_weight.to(torch.float64)
    ln_weight = ln_weight * trans.diag_scale.to(torch.float64)
    ln.weight.data = ln_weight.to(ori_dtype)
    trans.use_diag = False


def save_flat_matrices(model, matrices_path):
    from .default_model_utils import FlatQuantAttention, FlatQuantMLP
    flat_matrices = {}
    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
        if isinstance(layer.self_attn, FlatQuantAttention) and isinstance(layer.mlp, FlatQuantMLP):
            layer.self_attn.rep_matrix_only()
            layer.mlp.rep_matrix_only()
            paras_name = ["trans.matrix", "trans.diag_scale", "clip_factor_w", "clip_factor_a"]
            flat_matrices[i] = get_paras_dict_by_name(layer, required_names=paras_name)
    torch.save(flat_matrices, matrices_path)
    logging.info("saved paramaters at {}".format(matrices_path))


def load_flat_matrices(model, matrix_path):
    from .default_model_utils import FlatQuantAttention, FlatQuantMLP
    flat_parameters = torch.load(matrix_path)
    layers = model.model.layers
    
    for i in range(len(flat_parameters.keys())):
        if isinstance(layers[i].self_attn, FlatQuantAttention) and isinstance(layers[i].mlp, FlatQuantMLP):
            flat_param = flat_parameters[i]
            layers[i].self_attn.rep_matrix_only()
            layers[i].mlp.rep_matrix_only()
            layers[i].load_state_dict(flat_param, strict=False)
        else:
            print(f'not flatquant layer {i}')
    return model
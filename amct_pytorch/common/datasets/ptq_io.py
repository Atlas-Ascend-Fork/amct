import os
import torch
from loguru import logger


def save_ptq_kwargs(position_ids, position_embeddings, attention_mask, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    if position_ids is not None:
        torch.save(position_ids, os.path.join(data_dir, f"position_ids.pkl"))
    if position_embeddings is not None:
        torch.save(position_embeddings, os.path.join(data_dir, f"position_embeddings.pkl"))
    if attention_mask is not None:
        torch.save(attention_mask, os.path.join(data_dir, f"attention_mask.pkl"))


def save_ptq_inps(act_stat, hook_name, quant_target, layer_idx, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    outs = act_stat[f"{hook_name}_out"]
    outs = torch.cat(outs)
    torch.save(outs, os.path.join(data_dir, f"block_{layer_idx}_{quant_target}_in.pkl"))


def load_ptq_inps(data_dir, quant_target, layer_idx):
    kwargs = {}
    if quant_target == "attn":
        if os.path.exists(os.path.join(data_dir, f"position_ids.pkl")):
            kwargs["position_ids"] = torch.load(os.path.join(data_dir, f"position_ids.pkl"), weights_only=True)
        if os.path.exists(os.path.join(data_dir, f"position_embeddings.pkl")):
            kwargs["position_embeddings"] = torch.load(os.path.join(data_dir, f"position_embeddings.pkl"), 
                                                       weights_only=True)
        if os.path.exists(os.path.join(data_dir, f"attention_mask.pkl")):
            kwargs["attention_mask"] = torch.load(os.path.join(data_dir, f"attention_mask.pkl"), weights_only=True)
    file_path = os.path.join(data_dir, f"block_{layer_idx}_{quant_target}_in.pkl")
    try:
        cached_inps = torch.load(file_path, weights_only=True)
    except FileNotFoundError:
        logger.warning("PTQ input file not found: {}", file_path)
        return None, kwargs
    return cached_inps, kwargs

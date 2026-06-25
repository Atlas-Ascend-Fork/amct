import os
import sys
sys.path.append("/home/wanghaoyu2/tvq")
import tqdm
import json
import shutil
import argparse

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model.tvq import TVQHadLinear, DGEFunction, matmul_hadU_cuda


def unpack_tvq_linear(linear: TVQHadLinear):
    in_features = linear.in_features
    out_features = linear.out_features
    device = linear.weight.device
    dtype = linear.weight.dtype
    new_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=(linear.bias is not None))
    new_linear.to(device=device, dtype=dtype)

    weight_zero_point = linear.code_generator.weight_zero_point
    weight_scale = linear.code_generator.weight_scale
    ori_weight = weight_scale*linear.weight.data + weight_zero_point
    q_weight = DGEFunction.apply(ori_weight, 3, 5.0)
    code_vec_size = linear.vec_size

    W_decompressed = linear.code_generator(q_weight.reshape(-1, code_vec_size)).reshape(out_features, -1)

    W_decompressed = matmul_hadU_cuda(W_decompressed, linear.hadamard_left_T.T.contiguous(), linear.K_left)
    W_decompressed = matmul_hadU_cuda(W_decompressed.T, linear.hadamard_right, linear.K_right).T
    W_decompressed = linear.SV.unsqueeze(1) * W_decompressed * linear.SU.unsqueeze(0)

    new_linear.weight.data = W_decompressed.to(device=device, dtype=new_linear.weight.dtype)
    if linear.bias is not None:
        new_linear.bias.data = linear.bias.to(device=device, dtype=new_linear.bias.dtype)
    return new_linear

def find_layers(module: nn.Module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--overwrite_config_path", type=str, default=None)

args = parser.parse_args()
if args.overwrite_config_path is not None:
    assert os.path.exists(args.overwrite_config_path) and os.path.isfile(args.overwrite_config_path)
model_path = args.model_path #"/data/groups/QY_LLM_Other/wanghaoyu/exp/tvq/Qwen3-4B-instruct-2507-normal"
output_path = args.output_path #"/data/groups/QY_LLM_Other/wanghaoyu/exp/tvq/Qwen3-4B-instruct-2507-debug"

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device="cuda")
pbar = tqdm.tqdm(total=len(model.model.layers))

for i, layer in enumerate(model.model.layers):
    linears = find_layers(layer, layers=[TVQHadLinear])
    pbar.set_description_str(f"unpacking layer {i}")
    for name, q_linear in linears.items():
        new_linear = unpack_tvq_linear(q_linear)
        set_op_by_name(layer, name, new_linear)
    pbar.update(1)

model.to(dtype=torch.bfloat16)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

if args.overwrite_config_path is not None:
    custom_files = [args.overwrite_config_path]
    config_dir = os.path.dirname(args.overwrite_config_path)
else:
    custom_files = []
    config_dir = args.model_path
for config_name in ["config.json", "tokenizer_config.json"]:

    config_path = os.path.join(config_dir, config_name)
    print(config_path)
    if os.path.exists(config_path):
        with open(config_path) as f:
            automap = json.load(f).get('auto_map', {})
        for module_name in automap.values():
            if type(module_name) == list:
                module_name = module_name[0]
            module_name = module_name.split(".")[0]
            if not module_name in custom_files:
                custom_files.append(module_name + ".py")
                print(module_name)
for f in custom_files:
    shutil.copy(os.path.join(config_dir, f), output_path)
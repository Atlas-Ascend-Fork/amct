# Adapted from QuIP# (Cornell-RelaxML/quip-sharp), licensed under GPLv3.
# Source: https://github.com/Cornell-RelaxML/quip-sharp/blob/main/quantize_llama/hfize_llama.py
# Modified for MiniCPM quantization on Ascend NPU.
import os
import json
import shutil
import argparse

import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers.cache_utils as _cache_utils
if not hasattr(_cache_utils, "CacheLayerMixin"):
    class CacheLayerMixin: pass
    _cache_utils.CacheLayerMixin = CacheLayerMixin
if not hasattr(_cache_utils, "DynamicLayer"):
    class DynamicLayer: pass
    _cache_utils.DynamicLayer = DynamicLayer

from lib import codebook, utils
from lib.utils.model_version import MODEL_VERSION
from model.general_model import format_quantize_model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--hf_output_path', type=str)


def main(args):
    assert os.path.exists(args.quantized_path)
    base_config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'))
    model_config = saved_config['model_config']
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype='auto', low_cpu_mem_usage=True, config=model_config, trust_remote_code=True).half()
    codebook_id = codebook.get_id(model_config.quip_params['codebook'])
    codesz = model_config.quip_params['codesz']
    # print("id", codebook_id)
    # print(saved_config)
    # exit(0)
    model_config.quip_params['model_version'] = MODEL_VERSION
   
    model = format_quantize_model(model, model_config)

    cpu = torch.device('cpu')
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt', map_location=cpu)
        model.lm_head.weight.copy_(lmhead_data['lm_head'])
        model.model.norm.weight.copy_(lmhead_data['norm'])

    pbar = tqdm.tqdm(range(len(model.model.layers)))
    for layer_index in pbar:
        layer = model.model.layers[layer_index]

        if os.path.exists(f'{args.quantized_path}/{layer_index}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{layer_index}_layernorm.pt',map_location=cpu)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'])
            layer.post_attention_layernorm.weight.copy_(ln_data['post_attention_layernorm'])

        
        attn_linears = ['q', 'k', 'v', 'o'] if hasattr(layer.self_attn, "q_proj") else ['qkv', 'o']
        mlp_linears = ['up', 'gate', 'down'] if hasattr(layer.mlp, "up_proj") else ['gate_up', 'down']

        for attn_linear in attn_linears:
            try: 
                saved_layer = torch.load(f'{args.quantized_path}/{layer_index}_self_attn.{attn_linear}_proj.pt',map_location=cpu)
                utils.unpack_quip(getattr(layer.self_attn, f"{attn_linear}_proj"), saved_layer, codebook_id, codesz)
            except:
                print(f"exception occured when processing '{args.quantized_path}/{layer_index}_self_attn.{attn_linear}_proj.pt'")
                print("saved checkpoints has: ", saved_layer.keys())
                exit(0)

        for mlp_linear in mlp_linears:
            saved_layer = torch.load(f'{args.quantized_path}/{layer_index}_mlp.{mlp_linear}_proj.pt',map_location=cpu)
            utils.unpack_quip(getattr(layer.mlp, f"{mlp_linear}_proj"), saved_layer, codebook_id, codesz, load_tuneable=("tuneable" in model_config.quip_params["codebook"]))

        pbar.write(f'loaded layer {layer_index} done')

    pbar.write(f'saving model...')
    try: 
        model.save_pretrained(args.hf_output_path, safe_serialization=True)
    except:
        model.save_pretrained(args.hf_output_path, safe_serialization=False)

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.hf_output_path)

    custom_files = []
    config_dir = model_config._name_or_path
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
        shutil.copy(os.path.join(config_dir, f), args.hf_output_path)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)

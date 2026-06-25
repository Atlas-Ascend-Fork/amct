import os
import math
import json
import shutil
import argparse
import warnings
from tqdm import tqdm

warnings.simplefilter('once', UserWarning)

import torch
import torch.nn.functional as F
from safetensors import safe_open, SafetensorError
from transformers import AutoTokenizer
from model.modeling_vq import LCVQConfig, LCVQForCausalLM
from model.matmul_had import matmul_hadU_cuda, get_hadK

parser = argparse.ArgumentParser()
parser.add_argument("--fp16_model_path", type=str, required=True, help="path to the fp16 model")
parser.add_argument("--init_ckpt_path", type=str, required=True, help="path to the init checkpoint")
parser.add_argument("--output_path", type=str, required=True, help="path to save the converted checkpoint")

eps = 1e-6


def idx_to_quant_weight(idx: torch.Tensor, code_vec_size: int, code_nbit: int) -> torch.Tensor:
    quant_weight = idx.new_zeros(idx.shape[0], code_vec_size*idx.shape[1])
    for i in range(3, -1, -1):
        quant_weight[:, i::code_vec_size] = idx % (2**code_nbit)
        idx = idx // (2**code_nbit)
    return quant_weight


def create_init_weight(
    fp16_weight: torch.Tensor,
    scale_su: torch.Tensor,
    scale_sv: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    vec_size: int=4
):
    out_features, in_features = fp16_weight.shape
    weight = 1./scale_sv.unsqueeze(1) * fp16_weight * 1./scale_su.unsqueeze(0)
    hadamard_k_left, k_left = get_hadK(in_features)
    hadamard_k_right, k_right = get_hadK(out_features)
    weight = matmul_hadU_cuda(weight, hadamard_k_left.T.contiguous(), k_left)
    weight = matmul_hadU_cuda(weight.T, hadamard_k_right.T, k_right).T
    weight = weight.reshape(-1, vec_size)
    inv_proj = torch.linalg.inv(proj_weight.float()).to(weight.dtype)
    weight = F.linear((weight-proj_bias), inv_proj)
    return weight.reshape(out_features, in_features)


if __name__ == "__main__":

    args = parser.parse_args()

    ori_model_path = args.fp16_model_path
    init_ckpt_path = args.init_ckpt_path
    output_path = args.output_path

    with open(os.path.join(init_ckpt_path, "config.json")) as f:
        quip_params = json.load(f)["quip_params"]
    
    code_nbit = 2
    code_vec_size = quip_params["codesz"]
    if quip_params["codebook"] == "identical":
        proj_type = "hadamard_identical"
    else:
        proj_type = "hadamard_linear"

    config = LCVQConfig.from_pretrained(
        init_ckpt_path,
        vec_size=code_vec_size,
        vec_bit=code_nbit,
        torch_dtype=torch.bfloat16,
        use_hadamard=True,
        proj_type=proj_type
    )
    delattr(config, "quip_params")
    if hasattr(config, "auto_map"):
        delattr(config, "auto_map")

    model = LCVQForCausalLM.from_pretrained(ori_model_path, config=config, ignore_mismatched_sizes=True)
    import torch_npu  # noqa: F401
    device = torch.device("npu:0")
    with safe_open(os.path.join(init_ckpt_path, "model.safetensors"), framework="pt") as quip_ckpt:
        # 生成规则码本
        pbar = tqdm(total=len(model.model.layers))
        last_layer = 0
        for key, module in model.named_modules():
            module.to(device=device, dtype=config.torch_dtype)
            if not key.endswith("_proj"):
                continue
            pbar.set_description_str(f"processing module {key}")
            # 生成初始值
            codes = quip_ckpt.get_tensor(key + ".Qidxs").to(device=device, dtype=torch.uint8).to(torch.long)
            num_out_groups, num_in_groups = codes.shape
            out_features = num_out_groups
            in_features = num_in_groups * code_vec_size
            # 获取Scale
            su = quip_ckpt.get_tensor(key + ".SU").to(device=device, dtype=config.torch_dtype)
            sv = quip_ckpt.get_tensor(key + ".SV").to(device=device, dtype=config.torch_dtype)
            try:
                scale_wh = quip_ckpt.get_tensor(key + ".scaleWH")
            except SafetensorError as err:
                scale_wh = None
            if scale_wh is not None:
                sv = sv/scale_wh
            sv = quip_ckpt.get_tensor(key + ".Wscale") * sv
            su_norm = torch.norm(su, p=2)
            sv_norm = torch.norm(sv, p=2)
            scaling_norm = math.sqrt(su_norm*sv_norm+eps)
            su = su * scaling_norm/su_norm
            sv = sv * scaling_norm/sv_norm
            module.SU.data = su.to(config.torch_dtype)
            module.SV.data = sv.to(config.torch_dtype)
            # bias
            if module.bias is not None:
                try:
                    bias = quip_ckpt.get_tensor(key + ".bias")
                except SafetensorError as err:
                    bias = torch.zeros(out_features, device=device, dtype=config.torch_dtype)
                module.bias.data = bias.to(device=device, dtype=config.torch_dtype)
      
            # 生成权重、添加参考值、进行缩放
            weight = idx_to_quant_weight(codes, code_vec_size=code_vec_size, code_nbit=code_nbit).to(dtype=config.torch_dtype)
            xavier_zero_point = (2**code_nbit-1)/2
            xavier_scale = math.sqrt(out_features+in_features)/2
            weight = (weight - xavier_zero_point)/xavier_scale
            if proj_type == "hadamard_linear":
                module.code_generator.g1.weight.data = quip_ckpt.get_tensor(key + ".codebook_class.codebook.linear_proj_weight").to(device=device, dtype=config.torch_dtype)
                module.code_generator.g1.bias.data = quip_ckpt.get_tensor(key + ".codebook_class.codebook.linear_proj_bias").to(device=device, dtype=config.torch_dtype)
            else:
                codebook_scale = quip_ckpt.get_tensor(key + ".codebook_class.codebook.linear_proj_weight")[0, 0].item()
                codebook_zero_point = quip_ckpt.get_tensor(key + ".codebook_class.codebook.linear_proj_bias")[0].item()
                module.code_generator.codebook_scale.data = torch.tensor(codebook_scale)
                module.code_generator.codebook_zero_point.data = torch.tensor(codebook_zero_point)
            module.weight.data = weight.to(config.torch_dtype)
            hadamard_left, K_left = get_hadK(in_features)
            module.hadamard_left_T.data = hadamard_left.to(device=device, dtype=config.torch_dtype).T.contiguous()
            hadamard_right, K_right = get_hadK(out_features)
            module.hadamard_right.data = hadamard_right.to(device=device, dtype=config.torch_dtype)

            current_layer = int(key.split('.')[2])
            if current_layer > last_layer:
                pbar.update(1)
                last_layer = current_layer

        pbar.update(1)

    # 保存
    pbar.set_description_str(f"saving to {output_path}")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(init_ckpt_path)
    model.to(torch.bfloat16)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    custom_files = []
    for config_name in ["tokenizer_config.json"]:
        config_path = os.path.join(ori_model_path, config_name)
        if os.path.exists(config_path):
            with open(config_path) as f:
                automap = json.load(f).get('auto_map', {})
            for module_name in automap.values():
                if type(module_name) == list:
                    module_name = module_name[0]
                module_name = module_name.split(".")[0]
                if not module_name in custom_files:
                    custom_files.append(module_name + ".py")
    for f in custom_files:
        shutil.copy(
            os.path.join(ori_model_path, f),
            output_path)

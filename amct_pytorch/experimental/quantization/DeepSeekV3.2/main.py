# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

import json
import torch
from loguru import logger

from cores.utils import args_utils as args_utils, utils as utils
from cores.models.deepseek_v3_2.quant_utils import apply_quant_to_moe, get_float_block, apply_quant_to_mla
from cores.models.deepseek_v3_2.quant_utils import QuantDeepseekV3MLP
from cores.models.deepseek_v3_2.quant_dsa import QuantDSA
from cores.models.deepseek_v3_2.indexer import ModelArgs
from cores.calibrator.dsv3_calib import cali_quant
from cores.calibrator.utils import *


def train_mla_layer(args, data_dir, layer_idx, dev=0, cls=QuantDSA):
    logger.info(f"train_mla_layer - {layer_idx}")
    layer = get_float_block(args.model, layer_idx, dev, model_args=args.model_args)

    layer_inps, layer_outs = load_block_inps_outs(data_dir, layer_idx)

    inps = get_self_attn_inps_outs(layer, layer_inps)

    layer = apply_quant_to_mla(args, layer, cls=cls)

    logger.info(layer)
    if args.quantize:
        cali_quant(args, layer.self_attn, layer_idx, inps, dev, param_prefix="self_attn")
    torch.npu.empty_cache()


def train_experts_layer(args, data_dir, layer_idx, dev=0, cls=QuantDSA):
    logger.info(f"train_experts_layer - {layer_idx}")
    quant_params = {}

    def refactor_quant_params(quant_params, prefix):
        new_params = {}
        for k, v in quant_params.items():
            new_params[prefix + k] = v
        return new_params

    layer = get_float_block(args.model, layer_idx, dev, model_args=args.model_args)

    # replace float indexer
    layer = apply_quant_to_mla(args, layer, cls)
    layer_inps, layer_outs = load_block_inps_outs(data_dir, layer_idx)
    inps = get_mla_moe_inputs(layer, layer_inps, dev)

    torch.npu.empty_cache()
    layer = apply_quant_to_moe(args, layer, shared_expert_bits=8, routed_expert_bits=args.w_bits)
    logger.info(layer)
    if args.quantize:
        if (args.lwc or args.lac):

            if hasattr(layer.mlp, 'experts'):
                for idx in range(0, len(layer.mlp.experts)):
                    expert = layer.mlp.experts[idx]
                    prefix = f"mlp.experts.{idx}."

                    part_params = cali_quant(args, expert, layer_idx, inps, dev)
                    quant_params.update(refactor_quant_params(part_params, prefix))

                if isinstance(layer.mlp.shared_experts, QuantDeepseekV3MLP):
                    logger.info(" --- train shared expert --- ")
                    part_params = cali_quant(args, layer.mlp.shared_experts, layer_idx, inps, dev)
                    prefix = f"mlp.shared_experts."
                    quant_params.update(refactor_quant_params(part_params, prefix))

                logger.info(quant_params.keys())
                torch.save(quant_params, os.path.join(args.exp_dir, f"quant_parameters_{layer_idx}.pth"))
                logger.info(
                    "saved parameters at {}".format(os.path.join(args.exp_dir, f"quant_parameters_{layer_idx}.pth")))

            elif isinstance(layer.mlp, QuantDeepseekV3MLP):
                part_params = cali_quant(args, layer.mlp, layer_idx, inps, dev)
                prefix = f"mlp."
                quant_params.update(refactor_quant_params(part_params, prefix))
                logger.info(quant_params.keys())
                torch.save(quant_params, os.path.join(args.exp_dir, f"quant_parameters_{layer_idx}.pth"))
                logger.info(
                    "saved parameters at {}".format(os.path.join(args.exp_dir, f"quant_parameters_{layer_idx}.pth")))


def main():
    args = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)

    dev = args.dev
    data_dir = args.data_dir
    config = './cores/models/deepseek_v3_2/config_671B_v3.2.json'
    with open(config) as f:
        model_args = ModelArgs(**json.load(f))
    args.model_args = model_args

    for layer_idx in range(args.start_block_idx, args.end_block_idx):
        if args.train_mode == "mla":
            train_mla_layer(args, data_dir, layer_idx, dev)
        elif args.train_mode == "moe":
            train_experts_layer(args, data_dir, layer_idx, dev)
        else:
            raise ValueError("train_mode must be 'mla' or 'moe'")


if __name__ == '__main__':
    main()

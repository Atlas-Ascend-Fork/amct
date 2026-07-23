# coding=utf-8
# Adapted from
# https://github.com/ruikangliu/FlatQuant/blob/main/flatquant/args_utils.py
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
import argparse

from amct_pytorch.quantization.bit_policy import BitPolicy


def _validate_eval_mode(args):
    if args.eval_mode != "bf16":
        return
    policy = args.bit_policy
    if policy.has_quant_linear() or policy.has_quant_cache():
        raise ValueError(
            "eval_mode=bf16 requires a bit_config with no <16-bit entries.\n"
            f"{policy.summary()}"
        )


def parser_gen(command=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-V4-Flash',
                        help='Model to load.')
    parser.add_argument('--model_name', type=str, default='deepseek-ai/DeepSeek-V4-Flash',
                        help='Model to load.')
    parser.add_argument('--device', type=str, default='npu:0',
                        help='Device to use.')
    parser.add_argument('--granularity', type=str, default='model',
                        help='eval for block-wise or global.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for HuggingFace and PyTorch.')
    parser.add_argument(
        '--quant_target',
        nargs="+",
        default=[],
        choices=["mlp", "moe", "attn-linear", "attn-cache"],
        help='Only support [mlp, moe, attn-linear, attn-cache]',
    )
    parser.add_argument("--seq_len", type=int, default=4096)

    parser.add_argument('--data_dir', default="")

    parser.add_argument('--output_dir', type=str, default="./outputs", help='Output directory path.')

    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples.')
    parser.add_argument('--eval_mode', type=str, default='bf16', choices=['bf16', 'quant'],
                        help='bf16 uses the original model path; '
                             'quant rebuilds quant modules and toggles quantizers by bit-widths.')

    parser.add_argument("--quant_dtype", type=str, default="", choices=['int', 'mxfp', 'hifp'],
                        help='Quantization data type.')

    parser.add_argument('--cali_bsz', type=int, default=4,
                        help='Batch size default is 4.')
    parser.add_argument('--base_lr', type=float, default=1e-5,
                        help='Learning rate for learnable transformation.')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--lr_step_size', type=int, default=1)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')

    parser.add_argument('--algos', nargs="*", default=[],
                        help='Enabled quantization algorithms. The framework routes them by registry targets.')
    parser.add_argument('--is_per_tensor', action="store_true", default=False,
                        help='Use per-tensor statistics in activation clipping.')
    parser.add_argument('--k_size', type=int, default=128, help='Learnable hadamard-like matrix size.')

    parser.add_argument("--start_block_idx", type=int, default=0)
    parser.add_argument("--end_block_idx", type=int, default=61)

    parser.add_argument('--bit_config', type=str, default=None,
                        help='Path to a yaml file describing per-role bit-widths. '
                             'See configs/ for examples.')

    parser.add_argument('--attn_linear_param_dir', default="")
    parser.add_argument('--attn_cache_param_dir', default="")
    parser.add_argument('--moe_mlp_param_dir', default="")
    parser.add_argument('--wikitext_final_out', default="")

    args = parser.parse_args()

    if args.bit_config:
        args.bit_policy = BitPolicy.from_yaml(args.bit_config)
    else:
        args.bit_policy = BitPolicy()

    if command == "eval":
        _validate_eval_mode(args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return args

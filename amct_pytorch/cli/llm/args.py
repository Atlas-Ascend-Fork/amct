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


LINEAR_QUANT_TARGETS = {"mlp", "moe", "attn-linear"}
CACHE_QUANT_TARGET = "attn-cache"
LINEAR_BIT_FIELDS = ("a_bits", "w_bits")
CACHE_BIT_FIELDS = ("q_bits", "k_bits", "p_bits", "v_bits")
ALL_BIT_FIELDS = LINEAR_BIT_FIELDS + CACHE_BIT_FIELDS


def _format_bit_fields(args, field_names):
    return ", ".join(f"{field}={getattr(args, field)}" for field in field_names)


def _validate_quant_target_bits(args):
    quant_targets = set(args.quant_target)
    uses_linear_bits = any(target in LINEAR_QUANT_TARGETS for target in quant_targets)
    uses_cache_bits = CACHE_QUANT_TARGET in quant_targets

    invalid_linear_bits = [field for field in LINEAR_BIT_FIELDS if getattr(args, field) != 16]
    if not uses_linear_bits and invalid_linear_bits:
        raise ValueError(
            "a_bits/w_bits only apply to quant_target in {mlp, moe, attn-linear}, "
            f"but got quant_target={args.quant_target} with {_format_bit_fields(args, invalid_linear_bits)}."
        )

    invalid_cache_bits = [field for field in CACHE_BIT_FIELDS if getattr(args, field) != 16]
    if not uses_cache_bits and invalid_cache_bits:
        raise ValueError(
            "q_bits/k_bits/p_bits/v_bits only apply to quant_target containing 'attn-cache', "
            f"but got quant_target={args.quant_target} with {_format_bit_fields(args, invalid_cache_bits)}."
        )


def _validate_eval_mode(args):
    if args.eval_mode != "bf16":
        return

    invalid_bits = [field for field in ALL_BIT_FIELDS if getattr(args, field) != 16]
    if invalid_bits:
        raise ValueError(
            "eval_mode=bf16 requires all quantization bits to be 16, "
            f"but got {_format_bit_fields(args, invalid_bits)}."
        )


def parser_gen(command=None):
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path or HuggingFace repo id of the model to load.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Registry key used to dispatch the model adapter '
                             '(e.g. qwen3, qwen3_moe, deepseek-v3.2, glm5).')
    parser.add_argument('--device', type=str, default='npu:0',
                        help='Device to use.') # granularity
    parser.add_argument('--granularity', type=str, default='model',
                        help='eval for block-wise or global.') # granularity
    parser.add_argument('--seed', type=int, default=0, help='Random seed for HuggingFace and PyTorch.')
    parser.add_argument(
        '--quant_target',
        nargs="+",
        default=[],
        choices=["mlp", "moe", "attn-linear", "attn-cache"],
        help='Only support [mlp, moe, attn-linear, attn-cache]',
    )
    parser.add_argument("--seq_len", type=int, default=4096)

    # ptq extract
    parser.add_argument('--data_dir', default="")

    # ptq eval extract
    parser.add_argument('--output_dir', type=str, default="./outputs", help='Output directory path.')

    # extract
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples.')
    # eval
    parser.add_argument('--eval_mode', type=str, default='bf16', choices=['bf16', 'quant'],
                        help='bf16 uses the original model path; quant rebuilds quant modules and toggles quantizers by bit-widths.')

    # ptq eval
    parser.add_argument("--quant_dtype", type=str, default="mxfp", help='Quantization data type.')

    # ptq
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

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the linear layers.
                            This applies to all linear layers in the model, including down-projection and out-projection.''')

    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16,
                        help='Number of bits for weights of the linear layers.')

    # KV-Cache Quantization Arguments
    parser.add_argument('--q_bits', type=int, default=16,
                        help='Number of bits for queries quantization')

    parser.add_argument('--k_bits', type=int, default=16,
                        help='Number of bits for K-cache quantization')

    parser.add_argument('--p_bits', type=int, default=16,
                        help='Number of bits for P quantization.')

    parser.add_argument('--v_bits', type=int, default=16,
                        help='Number of bits for V-cache quantization.')

    # Experiments Arguments
    parser.add_argument('--attn_linear_param_dir', default="")
    parser.add_argument('--attn_cache_param_dir', default="")
    parser.add_argument('--moe_mlp_param_dir', default="")
    parser.add_argument('--wikitext_final_out', default="")

    args = parser.parse_args()
    _validate_quant_target_bits(args)
    if command == "eval":
        _validate_eval_mode(args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return args

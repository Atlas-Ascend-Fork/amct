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


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-V3.2',
                        help='Model to load.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for HuggingFace and PyTorch.')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the linear layers.
                            This applies to all linear layers in the model, including down-projection and out-projection.''')
    parser.add_argument('--a_asym', action="store_true", default=False,
                        help='Use asymmetric activation quantization.')

    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16,
                        help='Number of bits for weights of the linear layers.')
    parser.add_argument('--w_asym', action="store_true", default=False,
                        help='Use asymmetric weight quantization.')

    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples.')
    parser.add_argument('--cali_bsz', type=int, default=4,
                        help='Batch size default is 4.')
    parser.add_argument('--base_lr', type=float, default=1e-5,
                        help='Learning rate for learnable transformation.')
    parser.add_argument("--lwc", default=False, action="store_true",
                        help="Use learnable weight clipping.")
    parser.add_argument("--lac", default=False, action="store_true",
                        help="Use learnable activation clipping.")

    # KV-Cache Quantization Arguments
    parser.add_argument('--q_bits', type=int, default=16,
                        help='''Number of bits for queries quantization.
                        Note that quantizing the queries needs another rotation for the keys/queries.''')
    parser.add_argument('--q_asym', action="store_true", default=False,
                        help='Use asymmetric quantization for queries.')

    parser.add_argument('--k_bits', type=int, default=16,
                        help='''Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries.''')
    parser.add_argument('--k_asym', action="store_true", default=False,
                        help='Use asymmetric quantization for K-cache.')

    parser.add_argument('--v_bits', type=int, default=16,
                        help='Number of bits for V-cache quantization.')
    parser.add_argument('--v_asym', action="store_true", default=False,
                        help='Use asymmetric quantization for V-cache.')

    # Experiments Arguments
    parser.add_argument('--output_dir', type=str, default="./outputs", help='Output directory path.')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name.')

    parser.add_argument('--param_dir', default="")
    parser.add_argument('--mla_param_dir', default="")
    parser.add_argument('--moe_param_dir', default="")
    parser.add_argument('--wikitext_final_out', default="")
    parser.add_argument('--data_dir', default="")

    # Train and Val
    parser.add_argument("--train_mode", default="block", choices=["mla", "moe", "block"], help="Train mode.")
    parser.add_argument("--dev", type=int, default=1)
    parser.add_argument("--start_block_idx", type=int, default=0)
    parser.add_argument("--end_block_idx", type=int, default=61)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument('--load_c8_param', action="store_true", default=False, help='enable mla a8w8 and c8.')

    parser.add_argument("--cls", type=str, default='bf16')

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.quantize = (args.w_bits < 16) or (args.a_bits < 16) or (args.q_bits < 16) or (args.k_bits < 16) or (
                args.v_bits < 16)

    args.model_name = args.model.split("/")[-1]
    args.exp_dir = os.path.join(args.output_dir, f"w{args.w_bits}a{args.a_bits}_{args.train_mode}")
    os.makedirs(args.exp_dir, exist_ok=True)

    return args

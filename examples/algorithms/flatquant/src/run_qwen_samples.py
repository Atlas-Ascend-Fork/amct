# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details at
# http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------------

import argparse
import os

from utils import get_qwen, build_enc, seed_everything, get_wikitext2
import amct_pytorch as amct

from amct_pytorch.experimental.flatquant.flat_quant_module.flat_utils import save_flat_matrices, load_flat_matrices
from amct_pytorch.experimental.flatquant.flat_quant_module.train_utils import cali_flat_quant

INT4_FLAT_QUANT_CFG = {
    'quant_cfg': {
        'inputs': {
            'type': 'int4',
            'symmetric': True,
            'strategy': 'token'
        },
        'weights': {
            'type': 'int4',
            'symmetric': True,
            'strategy': 'channel',
        },
    },
    'algorithm': {
        'flatquant': {
            # KV-cache quantization
            'use_kcache_quant': False,
            'k_bits': 16,
            'k_sym': False,
            'use_vcache_quant': False,
            'v_bits': 16,
            'v_sym': False,

            # special control for o_proj & down_proj
            'use_o_quant': False,
            'use_down_quant': False,    # we choose to skip dowm_proj quantization, 
                                        # for qwen3 model '/no_think' prompt is sensitive to quantization

            # Other quantization parameters
            'lac': True,
            'lwc': True,
            'diag_alpha': 0.8,

            # Calibration
            'epochs': 15,
            'cali_bsz': 4,
            'flat_lr': 3e-3,
            'cali_trans': True,
        },
    },
    'skip_layers': {'lm_head'}
}


def content_generate(model, tokenizer):
    prompt = "Give me a short introduction to the Ascend Model Compression Toolkit(AMCT). /no_think"
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors='pt').to(model.device)

    # conduct text completion
    generated_ids = model.generate(**model_inputs, max_new_tokens=16384)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='model location')
    parser.add_argument('--device', type=str, default="npu:0", help='NPU device')
    parser.add_argument('--load_matrix', action='store_true', help="whether to load matrix")
    parser.add_argument('--flat_matrix_path', type=str, 
        default="./outputs/qwen/flat_matrices.pth", help='flat matrix location'
    )
    args = parser.parse_args()

    seed_everything()
    os.makedirs(os.path.dirname(args.flat_matrix_path), exist_ok=True)

    # prepare model
    model = get_qwen(args.model_path)
    model.to(args.device)
    tokenizer = build_enc(args.model_path)

    # original content generate
    ori_content = content_generate(model, tokenizer)

    # quantize & calibration
    amct.quantize(model, INT4_FLAT_QUANT_CFG)
    if args.load_matrix:
        model = load_flat_matrices(model, args.flat_matrix_path)
    else:
        calib_dataset = get_wikitext2(nsamples=128, seed=0, seqlen=2048, tokenizer=tokenizer)
        cali_flat_quant(model, calib_dataset, args.device)
        save_flat_matrices(model, args.flat_matrix_path)

    model.to(args.device)
    amct.convert(model)
    print(f'quantize model to W4A4 with FlatQuant success.')

    # test quantize model
    print(f'original model content: \n{ori_content}\n')
    # quatized model content generate
    quant_content = content_generate(model, tokenizer)
    print(f'quantized model content: \n{quant_content}\n')

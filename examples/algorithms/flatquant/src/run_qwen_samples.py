# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the Apache License Version 2.0.You may not use this file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS OF A PARTICULAR PURPOSE. See the
# Apache License for more details.
# http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------------

"""Qwen3 FlatQuant量化示例（quantize -> sequential_calibrate -> convert）。

quantize生成量化校准模型后，由sequential_calibrate逐层做梯度校准
（框架负责block切分与输入捕获，FlatQuant注册的gradient block calibrator
负责每层的AdamW训练），最后convert固化为NPU部署模块。
"""

import argparse
import logging
import os

import torch

from utils import build_enc, get_qwen, get_wikitext2, seed_everything
import amct_pytorch as amct
from amct_pytorch.classic.quantize_op.flatquant.flat_utils import (
    load_flat_matrices,
    save_flat_matrices,
)
from amct_pytorch.common.config.config import INT4_INT4_FLAT_QUANT_CFG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("Log").propagate = False


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
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='model location')
    parser.add_argument('--device', type=str, default="npu:0", help='NPU device')
    parser.add_argument(
        '--load_matrix', action='store_true', help="whether to load matrix"
    )
    parser.add_argument(
        '--flat_matrix_path',
        type=str,
        default="./outputs/qwen/flat_matrices.pth",
        help='flat matrix location',
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.flat_matrix_path), exist_ok=True)

    # prepare model
    model = get_qwen(args.model_path)
    model.to(args.device)
    tokenizer = build_enc(args.model_path)

    # original content generate
    ori_content = content_generate(model, tokenizer)

    # quantize & calibration
    amct.quantize(model, INT4_INT4_FLAT_QUANT_CFG)
    if args.load_matrix:
        model = load_flat_matrices(model, args.flat_matrix_path)
        for _, module in model.named_modules():
            if hasattr(module, 'flat_config') and hasattr(module, 'set_fake_quant'):
                module._calibrated = True
    else:
        calib_dataset = get_wikitext2(
            nsamples=128, seed=0, seqlen=2048, tokenizer=tokenizer
        )
        # 每项(inp, tar): sequential_calibrate只消费模型级输入
        calib_set = [inp.to(args.device) for inp, _ in calib_dataset]
        amct.sequential_calibrate(model, calib_set)
        save_flat_matrices(model, args.flat_matrix_path)

    model.to(args.device)
    amct.convert(model)
    logger.info('quantize model to W4A4 with FlatQuant success.')

    # test quantize model
    logger.info('original model content: %s', ori_content)
    # quatized model content generate
    quant_content = content_generate(model, tokenizer)
    logger.info('quantized model content: %s', quant_content)

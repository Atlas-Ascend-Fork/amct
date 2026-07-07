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

from datetime import datetime

import torch
import torch_npu
from utils import get_llama, build_enc, seed_everything, get_wikitext2, create_logger, eval_total
import amct_pytorch as amct

from amct_pytorch.experimental.flatquant.config import INT4_FLAT_QUANT_CFG
from amct_pytorch.experimental.flatquant.flat_quant_module.flat_utils import save_flat_matrices, load_flat_matrices
from amct_pytorch.experimental.flatquant.flat_quant_module.train_utils import cali_flat_quant


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='model location')
    parser.add_argument('--device', type=str, default="npu:0", help='NPU device')
    parser.add_argument('--load_matrix', action='store_true', help="whether to load matrix")
    parser.add_argument('--flat_matrix_path', type=str, 
        default="./outputs/llama2_7b/flat_matrices.pth", help='flat matrix location'
    )
    parser.add_argument('--eval_fake_quant', action='store_true', help="whether to evaluate fake quant")
    args = parser.parse_args()

    logger = create_logger()
    seed_everything()
    os.makedirs(os.path.dirname(args.flat_matrix_path), exist_ok=True)

    # Phase0: choose model && data
    model = get_llama(args.model_path)
    model.to(args.device)
    tokenizer = build_enc(args.model_path)
    calib_dataset = get_wikitext2(nsamples=128, seed=0, seqlen=2048, tokenizer=tokenizer)
    calib_dataset_eval = get_wikitext2(nsamples=128, seed=0, seqlen=2048, tokenizer=tokenizer, eval_mode=True)

    # Prompt to test speed
    text = "Hello world! Please say something"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Inference speed of original model
    t1_orig = datetime.now()
    with torch.no_grad():
        outputs = model(**inputs)
    t2_orig = datetime.now()
    t_diff_orig = (t2_orig - t1_orig).total_seconds() * 1000
    logger.info(f"Time diff orig: {t_diff_orig}")

    # Phase1: quantize model
    amct.quantize(model, INT4_FLAT_QUANT_CFG)
    logger.info(f"Model after quantization: \n{model}")

    # Phase2: inference calibration model to cal quantized factors
    if args.load_matrix:
        model = load_flat_matrices(model, args.flat_matrix_path)
        logger.info(f"Model after loading matrices: \n{model}")
    else:
        cali_flat_quant(model, calib_dataset, args.device)
        save_flat_matrices(model, args.flat_matrix_path)
    model.to(args.device)
    torch_npu.npu.empty_cache()

    # Optionally evaluate fake quant; much slower than real quant
    if args.eval_fake_quant:
        eval_total(model, tokenizer, calib_dataset_eval, logger)

    # Phase3: convert deploy model
    amct.convert(model)
    torch_npu.npu.empty_cache()
    logger.info(f"Model converted: \n{model}")

    # Inference speed of (real) quantized model
    t1_real_quant = datetime.now()
    with torch.no_grad():
        outputs = model(**inputs)
    t2_real_quant = datetime.now()
    t_diff_real_quant = (t2_real_quant - t1_real_quant).total_seconds() * 1000
    logger.info(f"Time diff after real quant: {t_diff_real_quant}")

    # Evaluate real quant
    eval_total(model, tokenizer, calib_dataset_eval, logger)

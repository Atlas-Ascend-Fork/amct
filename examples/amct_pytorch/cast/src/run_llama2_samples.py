# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import torch
import torch_npu

from utils import get_test_dataset, get_llama2, get_calib_dataset, infer_model, test_ppl
import amct_pytorch as amct

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example')
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    args = parser.parse_args()

    # Phase0: choose model
    model, enc = get_llama2(args.model_path)
    quant_model = model.eval().npu()

    # Phase1: quantize model
    cfg = amct.HIFP8_CAST_CFG
    amct.quantize(quant_model, cfg)

    # The quantized model is already a deployment model, no need to convert

    # Phase2: Test ppl result
    testenc = get_test_dataset(enc=enc, seqlen=model.seqlen)
    testenc = testenc.input_ids.npu()
    test_ppl(quant_model, testenc)

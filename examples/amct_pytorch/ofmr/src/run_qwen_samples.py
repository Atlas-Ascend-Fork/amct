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
import torch
import torch_npu

from utils import get_loaders, get_qwen, get_calib_dataset, infer_model, test_ppl
import amct_pytorch as amct

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example')
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    args = parser.parse_args()

    # Phase0: choose model && data
    model, enc = get_qwen(args.model_path)
    quant_model = model.eval().npu()

    samples = get_calib_dataset(tokenizer=enc, n_samples=512, block_size=256)
    samples = torch.cat(samples, dim=0)[:1, :]

    # Phase1: quantize model
    cfg = {
        'batch_num': 1,
        'quant_cfg': {
            'weights': {
                'type': 'float8_e4m3fn',
                'symmetric': True,
                'strategy': 'channel',
            },
            'inputs': {
                'type': 'float8_e4m3fn',
                'symmetric': True,
                'strategy': 'tensor',
            },
        },
        'algorithm': {'ofmr'}
    }
    amct.quantize(quant_model, cfg)
    
    # Phase2: inference calibration model to cal quantized factors
    infer_model(quant_model, samples)
    torch_npu.npu.empty_cache()

    # Phase3: convert deploy model
    amct.convert(quant_model)
    torch_npu.npu.empty_cache()

    # Phase4: Test ppl result
    testenc = get_loaders(enc=enc, seqlen=model.seqlen)
    testenc = testenc.input_ids.npu()
    test_ppl(quant_model, testenc)

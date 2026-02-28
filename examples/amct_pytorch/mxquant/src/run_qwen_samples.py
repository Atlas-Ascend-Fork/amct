# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import argparse
import torch
import torch_npu

from utils import get_test_dataset, get_qwen, get_calib_dataset, infer_model, test_ppl
import amct_pytorch as amct

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    args = parser.parse_args()

    # Phase0: choose model && data
    model, enc = get_qwen(args.model_path)
    quant_model = model.eval().npu()

    samples = get_calib_dataset(tokenizer=enc, n_samples=512, block_size=256)
    samples = torch.cat(samples, dim=0)[:64, :]

    # Phase1: quantize model
    cfg = amct.MXFP8_QUANT_CFG
    amct.quantize(quant_model, cfg)
    
    # Phase2: inference calibration model to cal quantized factors
    infer_model(quant_model, samples)
    torch_npu.npu.empty_cache()

    # Phase3: convert deploy model
    amct.convert(quant_model)
    torch_npu.npu.empty_cache()

    # Phase4: Test ppl result
    testenc = get_test_dataset(enc=enc, seqlen=model.seqlen)
    testenc = testenc.input_ids.npu()
    test_ppl(quant_model, testenc)

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

import torch
import torch_npu

from utils import get_loaders, get_qwen2, build_enc, get_calib_dataset, infer_model, test_ppl
import amct_pytorch as amct

if __name__ == '__main__':
    # Phase0: choose model && data
    model, model_path = get_qwen2('7b')
    quant_model = model.eval().npu()
    enc = build_enc(model_path)

    samples = get_calib_dataset(
        data="pileval", tokenizer=enc, n_samples=512, block_size=256
    )
    samples = torch.cat(samples, dim=0)[:256, :]

    # Phase1: quantize model
    cfg = {
        'batch_num': 1,
        'quant_cfg': {
            'weights': {
                'type': 'int4',
                'symmetric': False,
                'strategy': 'group',
                'group_size': 32
            },
        },
        'algorithm': {'gptq'},
        'skip_layers': {'lm_head', 'down_proj'}
    }
    amct.quantize(quant_model, cfg)
    
    # Phase2: inference calibration model to cal quantized factors
    infer_model(quant_model, samples)
    torch_npu.npu.empty_cache()

    # Phase3: convert deploy model
    amct.convert(quant_model)
    torch_npu.npu.empty_cache()

    # Phase4: Test ppl result
    testenc = get_loaders(dataset_name='wikitext2',
                        enc=enc,
                        seqlen=model.seqlen)
    testenc = testenc.input_ids.npu()
    test_ppl(quant_model, testenc)

# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
# ----------------------------------------------------------------------------
"""Qwen 类大模型 —— 剪枝 + INT4 AWQ 量化合成流水线示例。

执行方式：
    python3 src/run_qwen_prune.py --model_path=/data/Qwen2-7b/

演示 PrunePass 与 InsertQuantizeModulePass 在同一 ModelOptimizer 中串接的用法。
"""

import argparse
import torch
import torch_npu  # noqa: F401  实际运行需要 NPU 环境

from utils import get_causal_lm, get_calib_dataset, infer_model

import amct_pytorch as amct
from amct_pytorch.classic.ptq.optimizer.model_optimizer import ModelOptimizer


def main():
    parser = argparse.ArgumentParser(description="Qwen prune + AWQ-int4 example")
    parser.add_argument("--model_path", type=str, required=True, help="Qwen 模型本地路径")
    args = parser.parse_args()

    # Phase0: choose model && data
    model, tokenizer = get_causal_lm(args.model_path)
    quant_model = model.eval().npu()

    samples = get_calib_dataset(tokenizer=tokenizer, n_samples=512, block_size=256)
    samples = torch.cat(samples, dim=1)[:2, :]

    # Phase1: 用一个 ModelOptimizer 把"先剪枝再量化"放在同一条流水线里
    prune_cfg = {
        "methods": {
            "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.30}},
            "moe": {"name": "mass_variance", "kwargs": {"prune_ratio": 0.30, "boundary": 10}},
        },
        "missing_data_policy": "warn_skip",
    }
    opt = ModelOptimizer()
    opt.add_pass(amct.PrunePass(prune_cfg, data=samples))
    opt.do_optimizer(quant_model)

    # Phase2: 在剪枝后的模型上叠加 AWQ-int4 量化
    amct.quantize(quant_model, amct.INT4_AWQ_WEIGHT_QUANT_CFG)
    infer_model(quant_model, samples)
    torch_npu.npu.empty_cache()

    # Phase3: convert deploy model
    amct.convert(quant_model)
    torch_npu.npu.empty_cache()
    print("Done: pruned + INT4-AWQ Qwen ready for NPU deploy.")


if __name__ == "__main__":
    main()

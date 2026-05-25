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
"""DistilBERT (序列分类) + BoolQ 的稠密层 low-variance 剪枝示例。

执行方式：
    python3 src/run_distilbert_boolq_prune.py \\
        --model_path=/data/distilbert-base-uncased-qa-boolq

模型需用户自行下载并通过 --model_path 指定本地路径（与 AMCT awq/minmax 样例约定一致）。
"""

import argparse

from utils import get_seqcls, get_boolq_calib

# 直接从子包导入剪枝 façade，避免 amct_pytorch 顶层 __init__ 触发 torch_npu / 量化栈。
# 该示例在 CPU 环境下也可运行；如果同一脚本需要叠加 amct.quantize/convert，请改为
# `import amct_pytorch as amct`，并在 NPU 环境下执行。
from amct_pytorch.pruning import (
    DENSE_LOWVAR_PRUNE_CFG,
    get_last_report,
    prune,
)


def main():
    parser = argparse.ArgumentParser(description="DistilBERT BoolQ prune example")
    parser.add_argument("--model_path", type=str, required=True, help="HF 模型本地路径")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="./pruned_out")
    args = parser.parse_args()

    # Phase0: choose model && data
    model, tokenizer = get_seqcls(args.model_path)
    calib_batches = get_boolq_calib(tokenizer, n_samples=args.n_samples)

    # Phase1: prune (稠密层 low-variance 剪枝 50%)
    prune(model, DENSE_LOWVAR_PRUNE_CFG, data=calib_batches)
    report = get_last_report()
    print(
        f"Prune done: params {report.params_before} -> {report.params_after} "
        f"(reduction {(1 - report.params_after / max(report.params_before, 1)) * 100:.2f}%)"
    )

    # Phase2: (可选) 与量化合成 —— 在 NPU 环境下取消下面三行注释
    # import amct_pytorch as amct
    # amct.quantize(model, amct.INT8_MINMAX_WEIGHT_QUANT_CFG)
    # amct.convert(model)

    # Phase3: 落盘
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Saved pruned model to {args.save_dir}")


if __name__ == "__main__":
    main()

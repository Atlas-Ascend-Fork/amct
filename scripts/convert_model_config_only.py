#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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

"""Config-only test for infer-repo convert_model.py.

Generates config.json with quantization_config using the SAME logic as
``cann-recipes-infer_glm52/models/glm-5.2/utils/convert_model.py``,
but skips all weight conversion.  Only config.json + tokenizer files are
needed in the model directory.

Usage::

    python scripts/convert_model_config_only.py \
        --input_fp8_hf_path /path/to/GLM-5.2 \
        --output_hf_path /path/to/GLM-5.2-infer-test \
        --quant_type w8a8c16
"""

import json
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 1 — Make convert_model.py importable.
#
# convert_model.py does ``from mx_quantize import ...`` at module level, so we
# must add its directory to sys.path before importing.
# ---------------------------------------------------------------------------
CONVERT_MODEL_DIR = "/mnt/workspace/code/cann-recipes-infer_glm52/models/glm-5.2/utils"
sys.path.insert(0, CONVERT_MODEL_DIR)

from convert_model import (
    generate_ignore_item,
    generate_quant_config,
    copy_py_json_jinja,
)


# ---------------------------------------------------------------------------
# Step 2 — Parse args (same interface as convert_model.py CLI).
# ---------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--input_fp8_hf_path", type=str, required=True)
parser.add_argument("--output_hf_path", type=str, required=True)
parser.add_argument("--quant_type", type=str, default="w8a8c16",
                    choices=["bfloat16", "w8a8c16", "w8a8c8", "w4a8c16", "w4a8c8", "w8a8-mx"])
parser.add_argument("--clip", action="store_true")
parser.add_argument("--quant_param_path", type=str, default=None)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Step 3 — Read config.json and derive ignore list + quantization_config.
#
# This replicates the config-generation portion of convert_model.main(),
# lines 440-465, without the weight-processing loop.
# ---------------------------------------------------------------------------
fp8_path = args.input_fp8_hf_path
output_path = args.output_hf_path
quant_type = args.quant_type

os.makedirs(output_path, exist_ok=True)

config_file = os.path.join(fp8_path, "config.json")
with open(config_file, "r") as f:
    config = json.load(f)

num_hidden_layers = config["num_hidden_layers"]
num_nextn_predict_layers = config["num_nextn_predict_layers"]
first_k_dense_replace = config["first_k_dense_replace"]
num_layers = num_hidden_layers + num_nextn_predict_layers
indexer_types = config.get("indexer_types")

c8 = quant_type.endswith("c8")
w4a8 = quant_type.startswith("w4a8")
w8a8 = quant_type.startswith("w8a8")
is_mx = quant_type.endswith("mx")

print(f"quant_type={quant_type}  c8={c8}  w4a8={w4a8}  w8a8={w8a8}  is_mx={is_mx}")
print(f"num_hidden_layers={num_hidden_layers}  num_nextn={num_nextn_predict_layers}  "
      f"num_layers={num_layers}  first_k_dense_replace={first_k_dense_replace}")

if w8a8 or w4a8:
    quant_ignore_layers = generate_ignore_item(
        num_layers, num_hidden_layers, first_k_dense_replace,
        indexer_types=indexer_types, is_mx=is_mx,
    )
    quantization_config = generate_quant_config(
        c8, quant_ignore_layers, w4a8=w4a8, clip=args.clip, is_mx=is_mx,
    )
    config["quantization_config"] = quantization_config
    print(f"\nGenerated {len(quant_ignore_layers)} ignore entries.")
else:
    quant_ignore_layers = []
    print("\nbfloat16 mode — no quantization_config.")


# ---------------------------------------------------------------------------
# Step 4 — Copy support files (config.json, tokenizer, .py, .jinja).
# ---------------------------------------------------------------------------
copy_py_json_jinja(fp8_path, output_path)

# Overwrite config.json with quantization_config.
new_config_file = os.path.join(output_path, "config.json")
with open(new_config_file, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nconfig.json written to {new_config_file}")

if quant_ignore_layers:
    print(f"\n--- ignore list ({len(quant_ignore_layers)} entries) ---")
    for name in quant_ignore_layers:
        print(f"  {name}")

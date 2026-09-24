#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GLM52_BF16_DIR:?set GLM52_BF16_DIR to the complete original BF16 checkpoint}"
: "${GLM52_ASCEND_DIR:?set GLM52_ASCEND_DIR to a new independent output directory}"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN:-python3}" "${repo_dir}/amct_pytorch/cli/llm/deploy.py" \
  --trust_remote_code \
  --model "${GLM52_BF16_DIR}" \
  --model_name glm5_2 \
  --device cpu \
  --granularity tensor \
  --quant_dtype int \
  --bit_config "${repo_dir}/amct_pytorch/configs/deploy/tensor_w8a8.yaml" \
  --deploy_format ascend \
  --quant_layers_config "${repo_dir}/amct_pytorch/configs/deploy/glm5_2_ascend_w8a8.json" \
  --output_dir "${GLM52_ASCEND_DIR}"

#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${QWEN36_BF16_DIR:?set QWEN36_BF16_DIR to the complete original BF16 checkpoint}"
: "${QWEN36_ASCEND_DIR:?set QWEN36_ASCEND_DIR to a new independent output directory}"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN:-python3}" "${repo_dir}/amct_pytorch/cli/llm/deploy.py" \
  --trust_remote_code \
  --model "${QWEN36_BF16_DIR}" \
  --model_name qwen3_6_moe \
  --device cpu \
  --granularity tensor \
  --quant_dtype int \
  --bit_config "${repo_dir}/amct_pytorch/configs/w8a8.yaml" \
  --deploy_format ascend \
  --quant_layers_config "${repo_dir}/amct_pytorch/configs/deploy/qwen3_6_ascend_w8a8.json" \
  --output_dir "${QWEN36_ASCEND_DIR}"

#!/bin/bash
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

# Resolve the sample directory and repository root from this file's location.
COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="${SAMPLE_DIR:-$(cd -- "${COMMON_DIR}/.." && pwd)}"
AMCT_ROOT="${AMCT_ROOT:-$(cd -- "${SAMPLE_DIR}/../../.." && pwd)}"

# Require the model directory to be provided through an environment variable.
: "${MODEL:?Please export MODEL pointing to the local Qwen3.5-9B directory}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Common runtime settings.
MODEL_NAME="${MODEL_NAME:-qwen3_5}"
DEVICE="${DEVICE:-npu:0}"
GRANULARITY="${GRANULARITY:-block}"
SEQ_LEN="${SEQ_LEN:-4096}"
NSAMPLES="${NSAMPLES:-128}"
QUANT_DTYPE="${QUANT_DTYPE:-int}"
# Quantization settings.
ALGOS="${ALGOS:-autoround}"
EPOCHS="${EPOCHS:-15}"
BASE_LR="${BASE_LR:-1e-5}"
START_BLOCK_IDX="${START_BLOCK_IDX:-0}"
END_BLOCK_IDX="${END_BLOCK_IDX:-32}"

# Configuration file paths.
BF16_BIT_CONFIG="${BF16_BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/bf16.yaml}"
BIT_CONFIG="${BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/w4a4.yaml}"

# Calibration data paths.
OUTPUT_ROOT="${OUTPUT_ROOT:-${SAMPLE_DIR}/outputs}"
DATA_ROOT="${DATA_ROOT:-${OUTPUT_ROOT}/ptq_data}"
ATTN_DATA_DIR="${ATTN_DATA_DIR:-${DATA_ROOT}/attn-linear}"
MLP_DATA_DIR="${MLP_DATA_DIR:-${DATA_ROOT}/mlp}"

# Quantization output and evaluation parameter paths.
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${QUANT_DTYPE}}"
PARAM_ROOT="${PARAM_ROOT:-${OUTPUT_DIR}/ptq_params/${MODEL_NAME}}"
ATTN_LINEAR_PARAM_DIR="${ATTN_LINEAR_PARAM_DIR:-${PARAM_ROOT}/attn-linear}"
MLP_PARAM_DIR="${MLP_PARAM_DIR:-${PARAM_ROOT}/mlp}"

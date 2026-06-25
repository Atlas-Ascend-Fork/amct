#! /usr/bin/env bash
# MiniCPM4-3B PTQ entry (fast-hess SOTA recipe). Device-agnostic: runs on NPU or
# CUDA unchanged (device auto-detected in the Python entries). SOTA: 50.47% /
# PPL 12.40 (5-seed, ±0.07).
#
# Usage: bash scripts/run_3b.sh <stage> [seed]      # stage 1=hess 2=quant+ft 3=hfize
# CUDA:  TORCH_DEVICE=cuda BASE_MODEL=<cuda-3b-path> HARNESS_Q_BATCH=16 \
#        bash scripts/run_3b.sh <stage>
# 3B needs gpu:1 (pipeline is single-card; a 2nd card COW-doubles host RAM → OOM).
# Env switches: see _ptq_stage_runner.sh.

EXP_NAME="MiniCPM4-3B"
BASE_MODEL="${BASE_MODEL:-/user/anjiangze/Models/MiniCPM4-3B}"
WORKSPACE="${WORKSPACE:-/user/anjiangze/Workspaces/ptq-3b-npu-fast-hess-aligned}"
DATASET="${DATASET:-/user/fanruikai/workspace-ptq/pajam.jsonl}"
CODEBOOK="linear_guassian"
QUANT_MODEL_DTYPE="float32"   # 3B MUST be fp32: fp16 hadamard corrupts massive-
                              # activation outliers (memory project_m8_3b_npu_fp32_works)
DEFAULT_Q_BATCH=16            # device knob; HARNESS_Q_BATCH overrides
DEFAULT_HESS_BATCH=4
STAGE="${1:?usage: $0 <stage> [seed]}"
SEED="${2:-}"

source "$(dirname "$0")/_ptq_stage_runner.sh"

#! /usr/bin/env bash
# MiniCPM4-1B PTQ entry (fast-hess SOTA recipe). Device-agnostic: runs on NPU or
# CUDA unchanged (device auto-detected in the Python entries). SOTA: 42.9% /
# PPL 19.55 (5-seed, ±0.03).
#
# Usage: bash scripts/run_1b.sh <stage> [seed]      # stage 1=hess 2=quant+ft 3=hfize
# CUDA:  TORCH_DEVICE=cuda BASE_MODEL=/user/fanruikai/workspace-ptq/MiniCPM4-1B \
#        HARNESS_Q_BATCH=32 bash scripts/run_1b.sh <stage>
# 1B tip: set HARNESS_EMB_LEGACY_COPY=1 (memfix regressed 1B fp16 PPL; see
#         memory project_1b_memfix_ppl_regression). Env switches: see _ptq_stage_runner.sh.

EXP_NAME="MiniCPM4-1B"
BASE_MODEL="${BASE_MODEL:-/user/anjiangze/Models/MiniCPM4-1B}"
WORKSPACE="${WORKSPACE:-/user/anjiangze/Workspaces/ptq-1b-npu-fast-hess-aligned}"
DATASET="${DATASET:-/user/fanruikai/workspace-ptq/pajam.jsonl}"
CODEBOOK="linear_guassian"
QUANT_MODEL_DTYPE="float16"   # 1B fp16 finetune is fine (no massive-act overflow)
DEFAULT_Q_BATCH=32            # device knob; HARNESS_Q_BATCH overrides
DEFAULT_HESS_BATCH=4
STAGE="${1:?usage: $0 <stage> [seed]}"
SEED="${2:-}"

source "$(dirname "$0")/_ptq_stage_runner.sh"

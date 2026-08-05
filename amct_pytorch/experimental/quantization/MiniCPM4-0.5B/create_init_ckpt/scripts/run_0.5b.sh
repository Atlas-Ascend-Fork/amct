#! /usr/bin/env bash
# MiniCPM4-0.5B PTQ entry. Device-agnostic (device auto-detected in the Python
# entries). NOTE: 0.5B uses codebook=identical (its own experiment line); it has
# NOT been validated to the fast-hess SOTA bar like 1B/3B — treat as exploratory.
# Rewritten to match the 1B/3B per-model convention (was a one-off legacy script
# with relative log path, stage default=3, interactive prompt, --ft_epoch typo).
#
# Usage: bash scripts/run_0.5b.sh <stage> [seed]    # stage 1=hess 2=quant+ft 3=hfize
# Env switches: see _ptq_stage_runner.sh.

EXP_NAME="MiniCPM4-0.5B"
BASE_MODEL="${BASE_MODEL:-/user/fanruikai/workspace-ptq/MiniCPM4-0.5B}"
WORKSPACE="${WORKSPACE:-/user/anjiangze/Workspaces/ptq-0.5b-npu-fast-hess-aligned}"
DATASET="${DATASET:-/user/fanruikai/workspace-ptq/pajam.jsonl}"
CODEBOOK="identical"          # 0.5B experiment line (not linear_guassian)
QUANT_MODEL_DTYPE="float16"
DEFAULT_Q_BATCH=16
DEFAULT_HESS_BATCH=4
STAGE="${1:?usage: $0 <stage> [seed]}"
SEED="${2:-}"

source "$(dirname "$0")/_ptq_stage_runner.sh"

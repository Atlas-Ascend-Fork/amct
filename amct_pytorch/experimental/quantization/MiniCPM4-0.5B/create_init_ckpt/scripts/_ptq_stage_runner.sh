#! /usr/bin/env bash
# Shared stage runner for the per-model PTQ entries (run_1b/run_3b/run_0.5b).
# A per-model script sets the DEVICE-AGNOSTIC recipe variables below and then
# sources this file. Device (npu/cuda/cpu) is auto-detected inside the Python
# entries (quantize_finetune.py / hessian_general_model.py) — these scripts and
# this runner never name a device. Per-device knobs (batch sizes) come from env.
#
# Required vars set by the caller before sourcing:
#   EXP_NAME            e.g. MiniCPM4-1B
#   BASE_MODEL          HF base model dir (env-overridable in the caller)
#   WORKSPACE           output root (env-overridable in the caller)
#   DATASET             calibration jsonl (env-overridable in the caller)
#   CODEBOOK            linear_guassian | identical
#   QUANT_MODEL_DTYPE   float16 | float32   (recipe fact for this model)
#   DEFAULT_Q_BATCH     stage2 batch default (env HARNESS_Q_BATCH overrides)
#   DEFAULT_HESS_BATCH  stage1 batch default (env HARNESS_HESS_BATCH overrides)
#   STAGE               1 | 2 | 3
#   SEED                "" => single workspace; set => per-seed sweep
#
# Env switches (apply to all models):
#   HARNESS_SMOKE=1            devset=256/ctx=1024/ft_epochs=1 pipeline smoke (not accuracy)
#   HARNESS_EMB_LEGACY_COPY=1  revert memfix copy_/swap (RECOMMENDED for 1B; see
#                              memory project_1b_memfix_ppl_regression)
#   HARNESS_EMB_MEMMAP=1       memmap-back embedding cache (low RAM)
#   HARNESS_MAX_LAYERS=N       cap layers for profiling (0=all)
#   HARNESS_Q_BATCH / HARNESS_HESS_BATCH   override stage2 / stage1 batch per device
#   TORCH_DEVICE=cuda|npu      force device (else auto-detect npu→cuda→cpu)

# PYTHONPATH APPENDED (not replaced) so the image's CANN/`tbe` path survives.
export PYTHONPATH="/local/apps/910b-ptq/create_init_ckpt:${PYTHONPATH:-}"
# Serial TBE op-compile: lowers compile memory peak, harmless on CUDA. (NOT the
# 3B OOM fix — that was the Stage2 host-memory machinery, commit c9fbe28.)
export TE_PARALLEL_COMPILER=1

set -ueo pipefail

: "${EXP_NAME:?}" "${BASE_MODEL:?}" "${WORKSPACE:?}" "${DATASET:?}" "${CODEBOOK:?}"
: "${QUANT_MODEL_DTYPE:?}" "${DEFAULT_Q_BATCH:?}" "${DEFAULT_HESS_BATCH:?}" "${STAGE:?}"

# Per-seed sweep isolates workspaces so seeds don't clobber each other's ckpt.
if [ -n "${SEED:-}" ]; then
    WORKSPACE="${WORKSPACE%-aligned}-sweep-seed${SEED}"
fi
export WORKSPACE_ROOT="$WORKSPACE"   # devset tokenizer cache lives here
seed_arg=${SEED:-0}

# Batch sizes are a device characteristic, not a recipe value: env wins, else
# the per-model default. The Python entries also honor HARNESS_Q/HESS_BATCH.
q_batch_size=${HARNESS_Q_BATCH:-$DEFAULT_Q_BATCH}
hess_batch_size=${HARNESS_HESS_BATCH:-$DEFAULT_HESS_BATCH}

# --- recipe (device-agnostic) ---
if [ "${HARNESS_SMOKE:-0}" != "0" ]; then
    ctx_size=1024; devset_size=256; ft_epochs=1; scale_search_iters=1
    echo "[smoke] devset=256 ctx=1024 ft_epochs=1 scale_search_iters=1 (pipeline test, not accuracy)"
else
    ctx_size=2048; devset_size=4096; ft_epochs=5; scale_search_iters=3
fi
hess_model_dtype=float32   # Hessian always fp32 (fed the fast-hess SOTA recipe)
ft_bs=4
ft_valid_size=128
scale_override=0.83

exp_dir=${WORKSPACE}/exp/quip_sharp/${EXP_NAME}
hess_path=${exp_dir}/hess_npu_fast_aligned
ckpt_path=${exp_dir}/ckpt_npu_fast_aligned
hf_path=${exp_dir}/hf_npu_fast_aligned
log_path=${WORKSPACE}/logs/${EXP_NAME}_npu_fast_aligned
mkdir -p "$log_path"

ensure_empty_or_wipe() {
    local target_dir="$1"
    mkdir -p "$target_dir"
    if ! [[ $(ls "$target_dir" | wc -w) == "0" ]]; then
        echo "$target_dir already exists -- wiping in non-interactive mode."
        rm "$target_dir"/*
    fi
}

case "$STAGE" in
    1)
        ensure_empty_or_wipe "$hess_path"
        python quantize_llama/hessian_general_model.py \
            --batch_size $hess_batch_size \
            --devset_size $devset_size \
            --ctx_size $ctx_size \
            --base_model "$BASE_MODEL" \
            --save_path "$hess_path" \
            --dataset_path "$DATASET" \
            --model_dtype "$hess_model_dtype" \
            --seed $seed_arg \
            2>&1 | tee "$log_path/step1_hessian.log"
        ;;
    2)
        mkdir -p "$ckpt_path"
        python quantize_llama/quantize_finetune.py \
            --save_path "$ckpt_path" \
            --codebook "$CODEBOOK" \
            --batch_size $q_batch_size \
            --ft_bs $ft_bs \
            --scale_override $scale_override \
            --base_model "$BASE_MODEL" \
            --hessian_path "$hess_path" \
            --dataset_path "$DATASET" \
            --devset_size $devset_size \
            --ctx_size $ctx_size \
            --ft_valid_size $ft_valid_size \
            --scale_search_iters $scale_search_iters \
            --ft_epochs $ft_epochs \
            --model_dtype "$QUANT_MODEL_DTYPE" \
            --seed $seed_arg \
            2>&1 | tee "$log_path/step2_finetune.log"
        ;;
    3)
        ensure_empty_or_wipe "$hf_path"
        python quantize_llama/hfize.py \
            --base_model "$BASE_MODEL" \
            --quantized_path "$ckpt_path" \
            --hf_output_path "$hf_path" \
            2>&1 | tee "$log_path/step3_hfize.log"
        ;;
    *)
        echo "unknown stage: $STAGE (expected 1, 2, or 3)" >&2
        exit 2
        ;;
esac

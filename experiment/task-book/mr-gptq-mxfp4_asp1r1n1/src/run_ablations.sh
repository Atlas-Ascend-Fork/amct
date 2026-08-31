#!/bin/bash
# W4A4 补充消融（Qwen3-8B，全量 73 块，bf16 基准 8.9893）
#
# 用法：bash run_ablations.sh [组名...]   不带参数=全跑
#   s1  阻尼延伸 0.3/0.4/0.6      —— 0.05→0.2 的增量未衰减，曲线可能未触底
#   a2  校准集敏感性 seed 7/2024  —— 达标余量仅 0.0103，需量化对校准集选取的敏感性
#   b1  留一法 ×4                 —— 从最终配置各去掉一项，回答「哪项可以拿掉」
#   b2  大校准集 × 阻尼           —— 验证「样本变多 → 最优阻尼左移」
#
# 断点续跑：日志已存在且含 [RESULT] 的 run 自动跳过。
# 单次耗时 ≈ 59min（校准 ~31 + 评测 ~28）；b2 因校准集 ×4 约 68min。
set -u
cd "$(dirname "$0")" || exit 1

M=/mnt/workspace/models/Qwen3-8B
M35=/mnt/workspace/models/Qwen3.6-35B-A3B
LOG=logs
mkdir -p $LOG

BASE="--model_path $M --mode mr_gptq --seq_len 4096"
INC="--increments hadamard scale_fitting act_quant act_scale_fit joint_comp"
PROT="--act_protect down_proj o_proj"
ROT="--hadamard_k 4096 --hadamard_full"

run () {
  local tag=$1; shift
  local f="$LOG/$tag.log"
  if [ -f "$f" ] && grep -q '\[RESULT\]' "$f"; then
    echo ">>> skip $tag（已完成）"; return
  fi
  echo ">>> [$(date +%H:%M:%S)] $tag"
  echo "    python3 -u run_eval.py $BASE $*"
  python3 -u run_eval.py $BASE "$@" 2>&1 | tee "$f"
  echo ">>> [$(date +%H:%M:%S)] $tag done"
}

# 35B 专用（双卡 device_map；单次约 35min，比 8B 便宜一半）
run35 () {
  local tag=$1; shift
  local f="$LOG/$tag.log"
  if [ -f "$f" ] && grep -q '\[RESULT\]' "$f"; then
    echo ">>> skip $tag（已完成）"; return
  fi
  echo ">>> [$(date +%H:%M:%S)] $tag"
  ASCEND_RT_VISIBLE_DEVICES=0,1 python3 -u run_eval.py       --model_path $M35 --mode mr_gptq --seq_len 4096 --device_map auto       --calib_chunk 32 "$@" 2>&1 | tee "$f"
  echo ">>> [$(date +%H:%M:%S)] $tag done"
}

SEL=${*:-"s1 a2 b1 b2"}

# ── S1：阻尼延伸 ────────────────────────────────────────────────────────
if [[ " $SEL " == *" s1 "* ]]; then
  for D in 0.3 0.4 0.6; do
    run "w4a4_damp${D}_73"  $ROT $PROT $INC --perc_damp $D
  done
fi

# ── A2：校准集敏感性（评测本身确定性，随机性只来自校准语料抽样）────────
if [[ " $SEL " == *" a2 "* ]]; then
  for S in 7 2024; do
    run "w4a4_seed${S}_73"  $ROT $PROT $INC --perc_damp 0.2 --calib_seed $S
  done
fi

# ── B1：留一法（基线 = 最终配置 delta 0.3897）──────────────────────────
if [[ " $SEL " == *" b1 "* ]]; then
  # ① 去 act_scale_fit
  run "w4a4_loo_noscalefit_73" $ROT $PROT --perc_damp 0.2 \
      --increments hadamard scale_fitting act_quant joint_comp
  # ② 去 joint_comp
  run "w4a4_loo_nojoint_73"    $ROT $PROT --perc_damp 0.2 \
      --increments hadamard scale_fitting act_quant act_scale_fit
  # ③ 去保护层 → 全网真 4-bit 激活，真 4-bit 层占比 71.4% → 100%
  run "w4a4_loo_noprotect_73"  $ROT       --perc_damp 0.2 $INC
  # ④ 退回 k=128（部署友好点：旋转在线开销 1.6% vs 全宽的 100%+）
  run "w4a4_loo_k128_73"       --hadamard_k 128 $PROT --perc_damp 0.2 $INC
fi

# ── B2：大校准集 × 阻尼 ─────────────────────────────────────────────────
if [[ " $SEL " == *" b2 "* ]]; then
  for D in 0.1 0.2 0.4; do
    run "w4a4_calib256_damp${D}_73" $ROT $PROT $INC \
        --perc_damp $D --n_calib 256 --calib_chunk 32
  done
fi

# ── C1：35B W4A4 × 3 seed ────────────────────────────────────────────────
# ⚠️ 35B 必须【不加】混合精度保护，故此处不带 $PROT：
#    该模型有 100 个 nn.Linear 因维度不整除 32 而无法量化、却仍计入任务书分母(350)。
#    带保护 → 200/350 = 57.1% 不达标；不带保护 → 250/350 = 71.4% 达标。
#    （另一分母 200/250 = 80% 是"已量化模块"口径，非验收口径，勿混用。）
#    8B 上结论相反——保护不可省。配置不可跨模型继承。
if [[ " $SEL " == *" c1 "* ]]; then
  for S in 42 7 2024; do
    run35 "w4a4_35b_noprot_seed${S}_72" $ROT $INC --perc_damp 0.4 --calib_seed $S
  done
fi

# ── C2：8B W4A16 主线 × 2 seed —— 主线余量 0.127，需确认不在噪声内 ──────
if [[ " $SEL " == *" c2 "* ]]; then
  for S in 7 2024; do
    run "w4a16_8b_seed${S}_73" --increments hadamard scale_fitting --calib_seed $S
  done
fi

# ── 汇总 ────────────────────────────────────────────────────────────────
echo
echo "================ 汇总（基准：8B 8.9893 / 35B 6.311；门槛 delta ≤ 0.40）================"
printf "%-32s %-12s %-10s %s\n" "配置" "PPL" "delta" "判定"
for f in $LOG/w4a4_*_73.log $LOG/w4a4_35b_*_72.log $LOG/w4a16_8b_*_73.log; do
  [ -f "$f" ] || continue
  p=$(grep -oP 'Score: \K[0-9.]+' "$f" | tail -1)
  [ -z "$p" ] && { printf "%-32s %-12s\n" "$(basename $f .log)" "(未完成)"; continue; }
  base=8.9893; case "$f" in *_35b_*) base=6.311;; esac   # 35B 的 bf16 基准不同
  d=$(python3 -c "print(f'{$p-$base:.4f}')")
  ok=$(python3 -c "print('PASS' if $p-$base<=0.40 else 'FAIL')")
  printf "%-32s %-12.4f %-10s %s\n" "$(basename $f .log)" "$p" "$d" "$ok"
done

#!/bin/bash
# 夜间队列（约 7h10m）。断点续跑：日志含 [RESULT] 则跳过。
#
# 排序原则：先「决定验收的」，再「决定结论可信度的」，最后「优化性的」。
# 已砍：留一法里去 act_scale_fit / 去 joint_comp 两项——今日实测本实验分辨率
# 约 0.06（同 seed 阻尼响应面粗糙度）、跨 seed 极差 0.111，而这两项预期效应
# 仅 0.03 量级，测不出来。
set -u
cd "$(dirname "$0")" || exit 1

M8=/mnt/workspace/models/Qwen3-8B
M35=/mnt/workspace/models/Qwen3.6-35B-A3B
LOG=logs
mkdir -p $LOG

INC="--increments hadamard scale_fitting act_quant act_scale_fit joint_comp"
ROT="--hadamard_k 4096 --hadamard_full"

go () {                       # go <tag> <可见卡> <参数...>
  local tag=$1 dev=$2; shift 2
  local f="$LOG/$tag.log"
  if [ -f "$f" ] && grep -q '\[RESULT\]' "$f"; then
    echo ">>> [$(date +%H:%M:%S)] skip $tag（已完成）"; return
  fi
  echo ">>> [$(date +%H:%M:%S)] START $tag"
  ASCEND_RT_VISIBLE_DEVICES=$dev python3 -u run_eval.py --seq_len 4096 "$@" > "$f" 2>&1
  echo ">>> [$(date +%H:%M:%S)] DONE  $tag  ->  $(grep -oP 'Score: \K[0-9.]+' "$f" | tail -1)"
}

B35="--model_path $M35 --mode mr_gptq --device_map auto --calib_chunk 32"
B8="--model_path $M8  --mode mr_gptq"

# ①② 35B W4A4 无保护 × seed 7/2024（验收：seed42 得 delta 0.3242，余量仅 0.0758，
#     而 8B 上跨 seed 极差达 0.111 —— 必须确认 35B 的方差）
go w4a4_35b_noprot_seed7_72    0,1  $B35 $ROT $INC --perc_damp 0.4 --calib_seed 7
go w4a4_35b_noprot_seed2024_72 0,1  $B35 $ROT $INC --perc_damp 0.4 --calib_seed 2024

# ③ 8B W4A4 无保护 ★今晚最值钱：若能过 0.4，真 4-bit 层占比 71.4% → 99.6%
go w4a4_8b_noprot_damp04_73    0    $B8  $ROT $INC --perc_damp 0.4

# ④⑤ 8B W4A16 主线 × seed（主线余量 0.127，需确认不在噪声内）
go w4a16_8b_seed7_73           0    $B8  --increments hadamard scale_fitting --calib_seed 7
go w4a16_8b_seed2024_73        0    $B8  --increments hadamard scale_fitting --calib_seed 2024

# ⑥⑦ 8B W4A4 damp0.4（seed42 得 0.3519）× seed —— 定 worst-case 过线的头条配置
go w4a4_8b_damp04_seed7_73     0    $B8  $ROT --act_protect down_proj o_proj $INC \
                                          --perc_damp 0.4 --calib_seed 7
go w4a4_8b_damp04_seed2024_73  0    $B8  $ROT --act_protect down_proj o_proj $INC \
                                          --perc_damp 0.4 --calib_seed 2024

# ⑧ 8B W4A4 k=128（部署点：旋转在线开销 1.6% vs 全宽的 100%+）
go w4a4_8b_k128_damp04_73      0    $B8  --hadamard_k 128 --act_protect down_proj o_proj \
                                          $INC --perc_damp 0.4

# ── 汇总 ────────────────────────────────────────────────────────────────
echo
echo "========= 汇总（基准：8B 8.9893 / 35B 6.311；门槛 delta ≤ 0.40）========="
printf "%-34s %-11s %-10s %s\n" "配置" "PPL" "delta" "判定"
for f in $LOG/w4a4_*.log $LOG/w4a16_*.log; do
  [ -f "$f" ] || continue
  p=$(grep -oP 'Score: \K[0-9.]+' "$f" | tail -1)
  [ -z "$p" ] && { printf "%-34s %s\n" "$(basename $f .log)" "(未完成)"; continue; }
  base=8.9893; case "$f" in *_35b_*) base=6.311;; esac
  printf "%-34s %-11.4f %-10s %s\n" "$(basename $f .log)" "$p" \
    "$(python3 -c "print(f'{$p-$base:.4f}')")" \
    "$(python3 -c "print('PASS' if $p-$base<=0.40 else 'FAIL')")"
done
echo
echo "—— 真 4-bit 激活层占比（任务书口径）——"
grep -H "全部 nn.Linear" $LOG/w4a4_*.log 2>/dev/null | sed 's|.*/||'

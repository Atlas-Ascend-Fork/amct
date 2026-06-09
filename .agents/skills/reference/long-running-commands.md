# 长命令防超时（eval / extract_ptq_data / ptq / deploy）

agent runtime 的单条 Bash 调用有超时上限（如 opencode 默认约 600s）。大模型 eval、PTQ 训练、deploy 常超过它——一条**阻塞式**命令会被运行时直接杀掉（已踩坑：8B/35B 的 ptq 在 ~10min 处被砍，只训完前 ~10 层）。执行下列任一长命令时按本页做。

## 标准做法：后台 + 轮询（不要阻塞式单跑）

1. 后台启动并重定向日志：
   ```bash
   nohup <完整命令> > run.log 2>&1 &
   ```
   这条 Bash 调用立即返回，不会计入运行时的命令超时。
2. 用**多次短** Bash 调用轮询，直到结束：
   ```bash
   grep -E "PPL:|Error|Traceback|Saved PTQ|Block Processing" run.log | tail
   ps -p <pid>   # 进程是否还在
   ```
3. 进程结束后从 run.log 取结果（如 `PPL:` 行 / 报错）。

> 关键：把"启动"和"等待"拆成两次以上 Bash 调用；不要让一次 Bash 调用既启动又阻塞等到长命令结束——那正是被超时杀掉的原因。

## ptq 可断点续跑（被中断也不白跑）

- ptq 逐层保存参数到 `<output_dir>/ptq_params/<model>/<unit>/layer_<i>_<unit>.pt`。
- 若被中断，用 `--start_block_idx <下一未完成层>` 续跑，已训层不重复。
- 也可主动用 `--start_block_idx/--end_block_idx` 分块，每块控制在超时窗口内（与 PR #141 的分块 eval 思路一致：每块 ≤ 超时上限，再合并）。

## 经验时长（判断要不要后台）

- eval：~1B 几十秒；8B ~3min；30B+/35B ~20min/段 → **必须**后台+轮询
- extract_ptq_data：随 nsamples / 层数，几分钟到十几分钟
- ptq（autoround/omniquant 等）：~1min/层 × 层数 × 算法，大模型几十分钟到数小时 → **必须**后台+轮询或 `--start_block_idx` 分块
- deploy：blockwise 导出，30B ~1min、35B 数分钟

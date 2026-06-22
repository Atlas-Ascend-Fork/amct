---
name: quant-run
description: 量化执行 skill（implementer 用）。按已确认方案跑全部实际量化命令——直转评测 / 校准数据提取 / PTQ 算法训练 / PTQ 结果评测，按 algo 参数化（直转无 algo；PTQ 走 autoround/lac/lwc/omniquant 等）。只执行 + 把结果写 progress，不做达标判读（判读归 quant-reviewer），不设计方案（归 quant-analyzer）。
---

# 量化执行（quant-run）

implementer 的**统一执行 skill**：把"跑命令拿数 / 出产物"的所有重活收在这里——直转评测、校准数据提取、PTQ 训练、PTQ 结果评测，按 `--algos` 参数化覆盖直转与各 PTQ 算法。

**只执行与记录，不判达标**：`delta` 是否达标、PTQ 相对直转是否真改善，由 `quant-reviewer` 判读；第一轮方案 / 算法选择由 `quant-analyzer` 设计；本 skill 不改 adapter 逻辑、不改方案。

按需先读：[../references/direct-quant.md](../references/direct-quant.md)；PTQ 升级读 [../references/ptq-escalation.md](../references/ptq-escalation.md)。

## 命令模板

以 `examples/eval.sh` 为权威评测模板，不要凭空拼命令。四类命令：

```bash
# 1) 直转评测（无 algo）
python -m amct_pytorch.eval --model <path> --model_name <name> --device npu:N \
  --granularity block --eval_mode quant --quant_target <mlp|moe|attn-linear|attn-cache> \
  --quant_dtype <int|mxfp> --bit_config amct_pytorch/configs/<wXaY>.yaml --seq_len 4096

# 2) 校准数据提取（PTQ 前置）
python -m amct_pytorch.extract_ptq_data --model <path> --model_name <name> \
  --quant_target <target> --nsamples 128 --data_dir <dir>

# 3) PTQ 算法训练
python -m amct_pytorch.ptq --model <path> --model_name <name> --quant_target <target> \
  --quant_dtype <int|mxfp> --bit_config <yaml> --algos <algo> \
  --data_dir <dir> --<target>_param_dir <param_dir> --granularity block

# 4) PTQ 结果评测（加载训练参数）
python -m amct_pytorch.eval --model <path> --model_name <name> --device npu:N \
  --granularity block --eval_mode quant --quant_target <target> --quant_dtype <int|mxfp> \
  --bit_config <yaml> --algos <同 ptq> --<target>_param_dir <param_dir> --seq_len 4096
```

bf16 baseline = 命令 1 把 `--eval_mode` 换 `bf16`。

## 硬规则

- `--model_name` 必填（已注册适配器名；缺失会默认 deepseek 误用）。`--granularity block` 必填（默认 `model` 不真量化、给假"无掉点"）。`--quant_dtype` 在 quant 模式必填（缺报 `KeyError: '' is not registered in 'dtype'`）。
- **`--algos` 一致性**：加载 PTQ 参数（**评测 与 deploy 都**）必带与 ptq 训练一致的 `--algos`，否则加载侧 quant 模块不构建 `weight_quantizer.algorithms.<algo>` 子模块、`load_module` 报 `KeyError: Submodule '...algorithms.<algo>' is not found`。
- **算法只认 new-path（LLM）`ALGO_REGISTRY`**：当前 = `autoround / lac / lwc / omniquant`（gptq/awq/mxfp 视分支移植）；跑前确认算法在 new-path（忽略 classic 图量化名册，详见 L1 `cross-model-pitfalls.md`）。
- 一次 `--quant_target` 聚焦一个角色；多角色分别跑、分别给 `--*_param_dir`。
- 离线容器用 `cli_wrapper.py` 包一层（patch get_wikitext2/get_pileval 到本地数据）。
- **extract↔ptq 数据目录一致**：`extract_ptq_data --output_dir <D>` 与对应 `ptq --data_dir <D>` 必须**同一目录**（每个 quant_target 一份）；产物名 `block_<idx>_<target>_in.pkl`。报 `PTQ input file not found: <dir>/block_*_<target>_in.pkl` → 先核对 ptq 的 `--data_dir` 是否就是 extract 的 `--output_dir`（路径不一致最常见），用 `find -name 'block_*_in.pkl'` 确认真实位置，**不要直接判 extract 未生成 / 有 bug**（同 `--algos`：漏检自身用法别甩锅框架）。
- 引用仓库内路径一律相对路径。

## 长命令防超时（必须）

大模型 eval（30B+/35B 单段 ~20min+）、PTQ 训练（~1min/层）会超过 agent 单条 Bash 超时上限（如 ~600s）被杀。**必须后台跑 + 轮询**：`nohup <命令> > run.log 2>&1 &` 启动（Bash 立即返回、不计超时），再用多次短 Bash 调用 `grep -E "PPL:|Error|Saved" run.log` + `ps -p <pid>` 轮询到结束；**不要一条命令既启动又阻塞等到结束**。PTQ 被中断用 `--start_block_idx <下一未完成层>` 续跑（逐层存参不重复）。

## 产物与内循环自审

- eval：拿到 `PPL:` 数值，写 progress（**不判达标**）。
- extract：校准数据落 `--data_dir`。
- ptq：逐层 `layer_*_<target>.pt`，核对层数 ×（MoE 时 expert 数）齐全；缺层则该层 deploy 会 fallback 直转。
- 跑通命令 → 看 PPL/产物 → 基础校验；CLI crash（KeyError/shape 不匹配/文件缺失）先自定位修复，不盲目重试；方案本身有问题则停止并报 `quant-analyzer`。

## 完成后写 progress

把每步命令摘要、产物路径、`ppl_bf16 / ppl_quant / ppl_algo`、是否 crash 写入 progress 工作区，并更新顶部状态块 `STATUS`（达标与否、是否交付由 `quant-reviewer` 判读后定）。

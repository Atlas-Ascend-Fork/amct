---
name: direct-quant-eval
description: 评测用户已指定的直转量化方案。适用于用户已经明确给出量化模块、bit、dtype 或 algos，只想验证当前直转方案的 Wikitext PPL、delta、是否达标，以及下一步应保持、缩范围还是转入后续验证流程。
---

# 直转量化评测

## 用户调用格式

- 如果用户明确点名调用这个 skill，先返回这个 skill 对应的固定输入模板，再开始执行。
- 不替用户改成别的 skill；只有当用户没有指定 skill、只是描述目标时，才建议是否回到 `$quant-workflow`。
- 固定输入模板见：[../../reference/skill-input-template.md](../../reference/skill-input-template.md)

这个 skill 只用于**评测用户已指定的直转量化方案**。

它回答的问题是：

- 当前指定方案能不能跑通
- `ppl_bf16 / ppl_quant / delta` 分别是多少
- 当前结果是否达标
- 结果异常时优先该怀疑什么
- 下一步是保持方案、缩范围，还是转入后续验证流程

它不负责：

- 推荐第一轮量化方案
- 重新设计候选方案
- 做粗粒度误差定位
- 进入 PTQ
- 验证某个 PTQ / 量化算法是否能把精度拉回来

按下面顺序阅读：

1. [../../quant-workflow/references/metrics-and-thresholds.md](../../quant-workflow/references/metrics-and-thresholds.md)
2. [../../quant-workflow/references/direct-quant.md](../../quant-workflow/references/direct-quant.md)
3. 如需确认当前能力边界，再读 [../../../docs/repo-map.md](../../../docs/repo-map.md)

## 命令模板

评测命令以 `examples/eval.sh` 为唯一权威模板（含 bf16 baseline 与 quant 两段），不要凭空拼命令。按当前方案替换：

- `--model`：模型权重目录
- `--model_name`：**必填**，已注册适配器名（dense Qwen3=`qwen3`、30B MoE=`qwen3_moe`，其余见 `.agents/docs/repo-map.md`）；缺失会默认 deepseek 而误用
- `--granularity block`：**必填**。默认 `model` 走 `_run_modelwise` 用原始浮点模型**完全不量化**，PPL 会与 bf16 基线一致 → 假的"无掉点"；只有 `block` 才真正量化
- `--eval_mode`：`bf16`（基线）/ `quant`（直转评测）
- `--quant_target`：量化模块（如 `mlp attn-linear`）
- `--quant_dtype`：**quant 模式必填**（`int`/`mxfp`），缺失报 `KeyError: '' is not registered in 'dtype'`
- `--bit_config`：`amct_pytorch/configs/` 下 yaml（schema 顶层 `w_bits/a_bits`）
- 离线容器：用 `cli_wrapper.py` 包一层（patch get_wikitext2/get_pileval 到本地数据），裸 `-m amct_pytorch.eval` 离线会在加载数据处失败

## 固定流程

1. 先确认用户已经给出了直转方案，至少明确：
   - 量化模块
   - `quant_dtype`
   - `a_bits / w_bits`
   - `algos`
   - `granularity`
2. 如果用户没有给定方案，停止评测，转到 `$scheme-recommendation`。
3. 先查现有结果：
   - `outputs/`、日志或脚本目录里是否已有同模型、同口径、同方案结果
   - 当前代码、模型版本、评测口径是否与旧结果一致
4. 先检查前置条件：
   - 模型已经接进框架
   - BF16 baseline 已拿到
   - 已验证关闭量化后保持浮点等价
5. 跑当前指定的直转量化方案，拿到 `ppl_quant`。
6. 计算 `delta = ppl_quant - ppl_bf16`。
7. 根据结果做第一轮判读：
   - `delta <= 0.2`：当前方案可接受
   - `delta > 0.2`：当前方案不可接受，建议后续缩范围或转后续验证
   - 如果结果异常：先检查链路，不直接下结论
8. 输出“直转量化评测卡”。

## 边界规则

- 这里只评测**用户已指定的直转方案**，不负责发明新方案。
- 如果用户只说“帮我给个方案”，回到 `$scheme-recommendation`。
- 如果模型还没适配完成，回到 `$model-adapter`。
- 如果用户想验证某个 PTQ / 量化算法能否改善掉点，转到后续算法验证流程。
- 这里默认只给出直转量化的评测结论和下一步建议，不做误差定位和 PTQ 升级。
- 当前主判断指标是 Wikitext PPL 和 `delta <= 0.2`。
- 如果量化后 PPL 明显优于 BF16 很多，先检查链路，不直接当成功。
- 引用仓库内代码、脚本和文档路径时，一律使用仓库相对路径；不要写绝对路径。

## 输出要求

结束时必须给出一张“直转量化评测卡”，至少包含下面字段：

- `当前方案`
- `评测口径`
- `ppl_bf16`
- `ppl_quant`
- `delta`
- `结果判读`
- `异常检查`
- `下一步建议`
- `结果复用情况`

## 直转量化评测卡

```md
# 直转量化评测卡

## 当前方案
- 量化模块：
- `quant_dtype`：
- `a_bits / w_bits`：
- `algos`：
- `granularity`：

## 评测口径
- 数据集：
- 指标：
- `seq_len`：
- 模型版本 / 代码口径：

## 结果
- `ppl_bf16`：
- `ppl_quant`：
- `delta`：

## 结果判读
- 是否达标：
- 判断依据：
- 是否异常：

## 异常检查
- 优先检查 1：
- 优先检查 2：
- 优先检查 3：

## 下一步建议
- 保持当前方案：
- 缩小量化范围：
- 转入后续验证流程：

## 结果复用情况
- 是否复用旧结果：
- 如果没有复用，原因：
```

## 长命令防超时

大模型（30B+/35B）单段 eval ~20min+，会超过 agent runtime 单条 Bash 超时上限（如 ~600s）被杀。务必**后台跑 + 轮询**：`nohup <命令> > run.log 2>&1 &` 启动后用多次短 Bash 调用 `grep PPL:/报错 run.log` + `ps -p <pid>` 轮询到结束，不要阻塞式单跑。

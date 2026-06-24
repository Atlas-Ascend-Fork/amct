---
name: quant-workflow
description: amct 大模型量化的唯一编排入口（多 agent 集成只需对接此入口）：输入模型路径与工作目录（可选 quant_target / bits / device），按阶段完成 适配→量化实验→deploy 交付→归档 全生命周期，重入安全、复用已有结果不重跑。触发场景：端到端量化任意 amct 模型 / 单段需求（只适配 / 只方案 / 只评测 / 只导出，自动分流到对应叶子 skill）。约束：方案选择 / PTQ 升级 / 是否导出等关键决策停下与用户确认（human-in-the-loop）。
---

# amct 量化编排入口

## 概述

唯一的端到端编排入口。**不重写**叶子 skill 的规则，只负责：查现状 → 判阶段 → 串联/分流 → 复用已有结果 → deploy 后补交付文档。

全生命周期按阶段推进，每段达标确认后再进下一段：

```text
阶段 1  未适配（无模型注册 / block wrapper / 最小 PTQ 单元闭环）
阶段 2  已适配，未完成 BF16 闭环（baseline / 浮点等价 / 最小 PTQ smoke 未齐）
阶段 3  已适配，准备量化（BF16 闭环完成，准备直转 / 算法验证 / PTQ 升级）
阶段 4  已有量化结果，准备 deploy
阶段 5  deploy 已完成，待补交付文档（deploy_quantization.md 未补齐）
阶段 6  已有完整结果，准备归档或复核
```

**编排为 supervisor**：分析类委派 `quant-analyzer`（只读、不改代码/方案）、实施类委派 `quant-implementer`（改 adapter / 跑 `$quant-run` 执行全部命令 / 导出）、审查类委派 `quant-reviewer`（只读 quant-run 结果判读，不跑评测/ptq、不改方案）；按各自 `description` 匹配。编排自身只做判阶段 / 路由 / 汇总 / 交互确认，不直接改 adapter 代码或量化方案。

叶子 / 子任务：`$model-adapter`、`quant-tools/`（**执行** `$quant-run`；**推荐** `$scheme-recommendation` / `$algorithm-recommendation`；**判读** `$direct-quant-eval` / `$algorithm-validation`；**导出** `$deploy-export`）。

---

## 重要原则

- **编排不动手**：编排只调度与确认，不直接改 adapter 代码或量化方案；该派 implementer 的不自己改。
- **重入安全**：再次进入先读 progress.md 机读状态块，复用已有结果，不重跑。
- **环境 fail-fast**：环境是调用方责任，编排只暴露不排障；前置自检任一不满足即写 `BLOCKED` 并停止（见「状态协议与交付」）。
- **评测口径统一**：所有通过 skill 触发的 Wikitext PPL 统一 `seq_len=4096`，用户未显式指定即按 4096；历史结果非 4096 默认不能与当前 `delta` 直接横比或当同口径复用；仅当用户明确要求其他 `seq_len` 才允许偏离，并在结论里标注本次口径不同。
- **融合算子兼容性作参考输入**：`graph_fusion` 报告 JSON 仅作方案选择的参考，不单独评估、不增加确认环节——Pass 生效率 >80% 可按标准方案、50%–80% 适当保守、<50% 优先保守方案。

---

## 交互门（硬约束）

> 以下每一处都是**必须停下与用户确认的硬门**。无人机 / 批处理模式无法交互确认时，一律 **fail-fast 写 `BLOCKED`**，不得用 default 默默推进。

1. **方案确认契约（未确认绝不进实施层）**
   - 用户已显式指定方案（`quant_target / bits / quant_dtype / algos` 任一组合）→ 视为「已确认」，直接执行。
   - 用户未指定 → **必须先 `$scheme-recommendation` 产出「方案推荐卡」、停下让用户确认后**才委派 implementer；CLI 的 `quant_target` 等 default 只能作为推荐卡里的建议值，**不得当作自动执行值**默默起跑。
   - 无头 / 批处理且方案未经用户确认 → **fail-fast 写 `BLOCKED: 方案未确认，拒绝用 default 执行`**。
2. **复用确认**：查到已有 BF16 / quant / deploy 结果，先汇报；足够回答则不默认重跑，先确认。
3. **升级确认**：第一轮直转结果出来后，不默认升级，确认 接受 / 改方案重测 / 升级。
4. **直转 → PTQ 确认**：先说明为何不能停在直转，确认后再升级。
5. **deploy 精度门**：进 deploy 前**必须已产出 `ppl_bf16 / ppl_quant / delta` 精度结论并由用户确认接受**；缺 PPL 精度结论不得进 deploy。“实验结束”不等同“自动 deploy”。

---

## 工作流程

### 阶段判断

先把当前任务归入「概述」的阶段 1–6 其一，再按下方固定流程推进。

### 固定流程

1. **先查现有结果**：casebook 按三层读——L1 `.agents/docs/casebook/cross-model-pitfalls.md`（跨网络通用，先通读）→ 按 config/checkpoint 触发信号读 L2 `.agents/docs/casebook/structure-family-pitfalls.md` 命中的结构家族 → L3 `.agents/docs/casebook/<series>/<case>.md` 同模型/系列结论；再查 `outputs/`、日志、脚本目录的 BF16 / quant / deploy 结果；当前代码、模型版本、量化目标、口径是否一致。足够则复用并说明口径，仅在变化时重跑。
2. 阶段 1 / 2 → `$model-adapter`。
3. 阶段 3 → 进入**量化实验环**（见下）。
4. 阶段 4 → `$deploy-export`。
5. 阶段 5 → `$deploy-export`，目标改为：复核 deploy 产物 + 按模板生成 `deploy_quantization.md` + 与导出目录一一绑定。
6. 阶段 6 → 先整理复用已有结论，再决定是否还要调子 skill。
7. 跨阶段按序推进：`$model-adapter` → 量化实验环 → `$deploy-export`。
8. 任一次进入 deploy，**权重导出 ≠ 全流程结束**；只有 `deploy_quantization.md` 已补齐并足以给 infer 仓使用，deploy 才算闭环。

#### 量化实验环（阶段 3）

先读 `../quant-tools/references/metrics-and-thresholds.md` → `../quant-tools/references/direct-quant.md`；需升级 PTQ 再读 `../quant-tools/references/ptq-escalation.md`。

1. 无可执行的第一轮直转方案 → `$scheme-recommendation` 产出「量化方案推荐卡」。
2. implementer 跑 `$quant-run` 出第一轮直转 `ppl_bf16` / `ppl_quant`，reviewer 用 `$direct-quant-eval` 判 `delta`。
3. `delta <= 0.2` → 可接受、可停止升级；`delta > 0.2` → 未选算法走 `$algorithm-recommendation`，已选算法由 implementer 跑 `$quant-run`（含 ptq + 带 `--algos` 的结果评测）、reviewer 用 `$algorithm-validation` 判收益。
4. 直转 + 算法验证后仍不达标 → 做一轮**粗粒度**误差定位；只有粗定位之后才允许在**最小范围**引入 PTQ。
5. 每轮结束必须输出下一步决策：停止 / 补定位 / 小幅升级。
6. 默认：先直转再 PTQ；先整网方案再缩到 block / unit；误差定位只做粗粒度；PTQ 先最小范围。
7. 量化后 PPL 明显优于 BF16 很多时，先查链路，不直接当成功。

### 单段分流

用户只覆盖其中一段时，直接转对应叶子 skill，不默认走完整流程：

- 只要方案 → `$scheme-recommendation`
- 已指定方案、只看 delta → implementer 跑 `$quant-run` + reviewer `$direct-quant-eval` 判读
- 只导出 → `$deploy-export`（不在此重做评测 / PTQ / 精度判定）
- 只推荐算法 → `$algorithm-recommendation`
- 只验证算法 → implementer 跑 `$quant-run`（含 ptq）+ reviewer `$algorithm-validation` 判读

---

## 状态协议与交付

### progress.md 机读状态块（供轮询 / 多 agent 集成）

progress.md 顶部维护一个机读状态块：编排入口负责初始化与收尾更新，子代理在各自轮次更新。**上游 / 多 agent 集成只需轮询此块判断进度，无需读全文。** 格式（`key: value` 逐行可解析）：

```text
STAGE: <1-6 阶段号>
STATUS: IN_PROGRESS | DONE | BLOCKED
DELTA: ppl_bf16=<> ppl_quant=<> delta=<>     # 拿到量化结果后
ARTIFACTS: <deploy 目录 / 关键产物路径>        # DONE 时
BLOCKED: <原因> — <一行回退 hint>              # 仅 BLOCKED 时
UPDATED_BY: orchestrator | analyzer | implementer | reviewer
```

- **前置自检（任一子代理启动即做）**：`amct_pytorch` 可导入、NPU device 由调用方 `--device` 指定且可用、模型与评测数据可达；任一不满足 → 写 `STATUS: BLOCKED` + `BLOCKED: <缺失项> — <回退 hint>`（如数据不可达：`set HF_ENDPOINT 镜像 / 用 modelscope / 指本地路径`）**并停止**，不臆造继续。
- **终态**：达标可交付写 `STATUS: DONE` + `DELTA` + `ARTIFACTS`。

### 两段式结构（防上下文膨胀）

机读状态块之下分两段，用标记分隔：

```text
<!-- ===== 以上为常驻区，不清除 ===== -->
<!-- ===== 以下为工作区，阶段推进时归档并清空 ===== -->
```

- **常驻区**（机读状态块 + 产物契约结论 + 阶段概览）：编排维护，只追加不清空。
- **工作区**（各子代理按角色追加：方案分析 / 实施记录 / 验证·判读 / 诊断）：阶段推进时由编排跑本 skill 自带的归档脚本（`scripts/archive_progress.py`，相对**本 skill 目录**、非仓库根）把工作区归档到 `progress_history.md` 并清空。
- **历史只 Grep**：`progress_history.md` 禁止全文 Read，按关键字 Grep 取历史。

### deploy 交付边界

- `amct_pytorch/cli/llm/deploy.py` 与 `amct_pytorch/workflows/llm_deploy.py` 只负责导出权重产物。
- `deploy_quantization.md` 不由 deploy 代码自动生成，由 `$deploy-export` 按模板生成、与本次导出目录绑定。
- deploy 阶段分两步：① 代码导出权重目录 ② skill 补齐交付说明文档；二者不混。

### 文档写回触发

触发条件与目标（系列 casebook README / 个案 / L1·L2 经验库、repo-map、Agent Docs，及默认不写）统一见 [../../docs/README.md](../../docs/README.md) 的「文档写回触发」；编排在阶段收尾据此决定是否写回、写哪层。

### 输出要求

结束时至少说明：当前模型阶段；本轮复用了哪些已有结果；调用了哪些子 skill；本轮量化方案与 `ppl_bf16 / ppl_quant / delta`；是否达标、是否粗定位、是否进 PTQ；下一步决策；若重跑为何旧结果不可复用；若进 deploy：权重是否导出、`deploy_quantization.md` 是否补齐；是否更新 casebook / repo-map / Agent Docs（不更新需说明理由）。

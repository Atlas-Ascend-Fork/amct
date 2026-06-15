---
name: quant-workflow
description: amct 量化编排入口。从“模型导入”到“适配→量化实验→deploy 交付→归档”的全生命周期总控：先查现状判阶段、复用 casebook、串联叶子 skill，单段需求直接分流。量化方案与升级决策需用户确认。
---

# amct 量化编排入口

唯一的端到端编排入口。**不重写**叶子 skill 的规则，只负责：查现状 → 判阶段 → 串联/分流 → 复用已有结果 → deploy 后补交付文档。

叶子 / 子任务：`$model-adapter`、`quant-tools/`（`$scheme-recommendation` / `$direct-quant-eval` / `$algorithm-recommendation` / `$algorithm-validation` / `$deploy-export`）。

**编排为 supervisor**：分析类委派 `quant-analyzer`（只读、不改代码/方案）、实施类委派 `quant-implementer`（改 adapter / 跑命令 / 导出）、审查类委派 `quant-reviewer`（只跑评测不改方案），按各自 `description` 匹配；编排自身只做判阶段 / 路由 / 汇总 / 交互确认，不直接改 adapter 代码或量化方案。

## 评测口径强约束

- 所有通过 skill 触发的 Wikitext PPL 统一 `seq_len=4096`；用户未显式指定即按 4096。
- 历史结果非 4096 默认不能与当前 `ppl_bf16 / ppl_quant / delta` 直接横比或当同口径复用。
- 仅当用户明确要求其他 `seq_len` 时才允许偏离，且必须在结论里标注本次口径不同。

## 阶段判断（先把当前任务归入其一）

1. 未适配（无模型注册 / block wrapper / 最小 PTQ 单元闭环）
2. 已适配，未完成 BF16 闭环（baseline / 浮点等价 / 最小 PTQ smoke 未齐）
3. 已适配，准备量化（BF16 闭环完成，准备直转 / 算法验证 / PTQ 升级）
4. 已有量化结果，准备 deploy
5. deploy 已完成，待补交付文档（`deploy_quantization.md` 未补齐）
6. 已有完整结果，准备归档或复核

## 固定流程

1. **先查现有结果**：casebook 按三层读——L1 `docs/casebook/common-pitfalls.md`（跨网络通用，先通读）→ 按 config/checkpoint 触发信号读 L2 `docs/casebook/family-pitfalls.md` 命中的结构家族 → L3 `docs/casebook/<series>/<case>.md` 同模型/系列结论；再查 `outputs/`、日志、脚本目录的 BF16 / quant / deploy 结果；当前代码、模型版本、量化目标、口径是否一致。足够则复用并说明口径，仅在变化时重跑。
2. 阶段 1 / 2 → `$model-adapter`。
3. 阶段 3 → 进入**量化实验环**（见下节）。
4. 阶段 4 → `$deploy-export`。
5. 阶段 5 → `$deploy-export`，目标改为：复核 deploy 产物 + 按模板生成 `deploy_quantization.md` + 与导出目录一一绑定。
6. 阶段 6 → 先整理复用已有结论，再决定是否还要调子 skill。
7. 跨阶段按序推进：`$model-adapter` → 量化实验环 → `$deploy-export`。
8. 任一次进入 deploy，**权重导出 ≠ 全流程结束**；只有 `deploy_quantization.md` 已补齐并足以给 infer 仓使用，deploy 才算闭环。

### 量化实验环（阶段 3）

先读 `references/metrics-and-thresholds.md` → `references/direct-quant.md`；需升级 PTQ 再读 `references/ptq-escalation.md`；网络结构特殊再读 `references/network-strategies.md`。

1. 无可执行的第一轮直转方案 → `$scheme-recommendation` 产出“量化方案推荐卡”（融合算子兼容性作参考输入，见下节）。
2. `$direct-quant-eval` 跑第一轮直转 → `ppl_bf16` / `ppl_quant` / `delta`。
3. `delta <= 0.2` → 可接受、可停止升级；`delta > 0.2` → 未选算法走 `$algorithm-recommendation`，已选算法走 `$algorithm-validation`。
4. 直转 + 算法验证后仍不达标 → 做一轮**粗粒度**误差定位；只有粗定位之后才允许在**最小范围**引入 PTQ。
5. 每轮结束必须输出下一步决策：停止 / 补定位 / 小幅升级。
6. 默认：先直转再 PTQ；先整网方案再缩到 block / unit；误差定位只做粗粒度；PTQ 先最小范围。
7. 量化后 PPL 明显优于 BF16 很多时，先查链路，不直接当成功。

## 单段分流

用户只覆盖其中一段时，直接转对应叶子 skill，不默认走完整流程：
- 只要方案 → `$scheme-recommendation`
- 已指定方案、只看 delta → `$direct-quant-eval`
- 只导出 → `$deploy-export`（不在此重做评测/PTQ/精度判定）
- 只推荐算法 → `$algorithm-recommendation`
- 只验证算法 → `$algorithm-validation`

## 融合算子兼容性（参考输入）

融合算子分析报告作为**方案选择的参考输入**，不单独评估、不增加确认环节。方案推荐时综合：Pass 生效率 >80% 可按标准方案；50%–80% 适当保守；<50% 优先保守方案。输入为 `graph_fusion` 报告 JSON。

## 交互节点（必须停下与用户确认）

1. 从 `$model-adapter` 切到量化前：说明拟用方案；未锁定则问是否改 `quant_target / bits / quant_dtype / algos`。
2. 查到已有 BF16 / quant / deploy 结果：先汇报；足够回答则不默认重跑，先确认。
3. 第一轮直转结果出来后：不默认升级，确认 接受 / 改方案重测 / 升级。
4. 直转 → PTQ 前：先说明为何不能停在直转，确认后再升级。
5. 进 deploy 前：确认已接受当前方案；“实验结束”不等同“自动 deploy”。

## deploy 交付边界

- `amct_pytorch/cli/llm/deploy.py` 与 `amct_pytorch/workflows/llm_deploy.py` 只负责导出权重产物。
- `deploy_quantization.md` 不由 deploy 代码自动生成，由 `$deploy-export` 按模板生成、与本次导出目录绑定。
- deploy 阶段分两步：① 代码导出权重目录 ② skill 补齐交付说明文档；二者不混。

## 文档写回触发

- 系列 `casebook README`：某模型首次拿到稳定 BF16 baseline + 第一轮可复现结论；默认起点 / 敏感模块 / 升级策略 / 常见问题需修订；新结果推翻系列默认经验。
- 个案 L3 `<series>/<case>.md`：新直转或 PTQ 升级路线；代表性问题 / 排障结论；明显偏离系列经验；已有个案需按新结果修订；**仅本模型独有、不可迁移**的坑留这层。
- 经验库 L1/L2（`common-pitfalls.md` / `family-pitfalls.md`）：发现跨网络通用、或某结构家族通用的可复用坑/陷阱时写回。**晋升护栏：单模型新发现先留 L3，同家族 ≥2 模型复现才升 L2、跨家族复现才升 L1**（防单例过度泛化）；新增 L2 家族须配触发信号。
- `.agents/docs/repo-map.md`：实际改动 workflow / solver / 数据链路 / 导出加载语义 / 主注册链，或现状已与代码不一致。
- `.agents/docs/README.md`：文档层级 / 职责边界变化，或记录的目录结构与真实不一致。
- 默认不写：单次失败 / 不稳定结果；BF16 baseline 自身不稳定；本轮无新增可复用信息（需说明无需更新）。

## 输出要求

结束时至少说明：当前模型阶段；本轮复用了哪些已有结果；调用了哪些子 skill；本轮量化方案与 `ppl_bf16 / ppl_quant / delta`；是否达标、是否粗定位、是否进 PTQ；下一步决策；若重跑为何旧结果不可复用；若进 deploy：权重是否导出、`deploy_quantization.md` 是否补齐；是否更新 casebook / repo-map / Agent Docs（不更新需说明理由）。

## 使用示例

```bash
# 直转评测
python -m amct_pytorch.eval --model /path/to/model --model_name qwen3_5 \
  --quant_target mlp --quant_dtype int --bit_config amct_pytorch/configs/w8a8.yaml --eval_mode quant
# PTQ 训练
python -m amct_pytorch.ptq --model /path/to/model --model_name qwen3_5 \
  --quant_target mlp --quant_dtype int --bit_config amct_pytorch/configs/w8a8.yaml
# Deploy 导出
python -m amct_pytorch.deploy --model /path/to/model --model_name qwen3_5 \
  --quant_target mlp --quant_dtype int --bit_config amct_pytorch/configs/w8a8.yaml --output_dir ./deploy_out
```

# 委派规则

## 先判断任务类型

`amct-coder` 是编码总入口，但不是所有 AMCT 任务都应该在这里执行。任务实质不是编码，或已有更专用 skill 时，按本文件委派。

先区分用户要的是“实现”还是“评审拟议落点”：

| 动作 | amct-coder 的处理 |
| --- | --- |
| 新增或结构性实现外部 skill 职责 | 立即进入外部 skill，不预读 amct-coder 的 repo-map/playbook |
| review 指定 workflow/solver/quant module 等落点是否合理 | 用最小资料审查当前落点；确认替代实现属于外部职责后再委派 |
| 局部 bug 报告，根因尚未确认 | 先沿实际数据链核验；根据确认后的改动实质决定保留或委派 |

## 委派表

| 用户需求 | 使用 skill | 处理方式 |
| --- | --- | --- |
| 新增模型 adapter、wrapper 类、attention/MLP/MoE 结构、注册新模型、改 unit 拆分策略 | `model-adapter` | 模型层实现全部转交；`amct-coder` 只做契约对齐和 review，见下节 |
| 新增/修改 `amct_ops/` NPU 算子、Ascend C kernel、PyTorch C++ extension、Meta kernel、`ops_build.sh` | `amct-ops-dev` | `amct-coder` 只判断调用边界和构建入口；硬件实现转交 |
| 跑直转 eval、提取校准数据、PTQ 训练、PTQ 结果评测命令 | `quant-run` | 本 skill 不执行实验调参 |
| 端到端模型量化，从适配到方案、实验、deploy、归档 | `quant-workflow` | 由主编排 skill 分阶段处理 |
| 直转量化结果是否达标、ppl_bf16/ppl_quant delta 判读 | `direct-quant-eval` | 只读结果并给下一步建议 |
| PTQ 算法结果相对直转是否改善 | `algorithm-validation` | 读取 ppl_algo 与 ppl_direct 后判读 |
| 为已适配模型推荐第一轮量化方案 | `scheme-recommendation` | 不在编码 skill 中下方案结论 |
| 为已适配模型推荐下一步 PTQ 算法 | `algorithm-recommendation` | 不在编码 skill 中判算法优先级 |
| 导出 deploy-ready safetensors 目录与交付文档 | `deploy-export` | 不在编码 skill 中执行交付导出 |
| 创建/查看 GitCode PR、读取/删除 PR 评论、查看 PR diff | `gitcode-pr` | 必须使用 GitCode API skill |
| 查看 GitCode issue 或评论 | `gitcode-issue` | 必须使用 GitCode API skill |

## 与 model-adapter 的关系

`model-adapter` 拥有模型层实现，位置是 `.agents/skills/quant-tools/model-adapter/`。这是硬规则，不是可选项：

**凡是新增或结构性修改 `amct_pytorch/common/models/llm/...`，一律转 `model-adapter`。** 包括新增模型族、新增模型 adapter、注册新模型、新增 attention/MLP/MoE wrapper 类、改 wrapper 结构、改 `iter_ptq_units()` 的 unit 拆分策略。

amct-coder 在模型层只保留三件事，见 [llm-model-unit.md](llm-model-unit.md)：

1. 契约对齐检查：`PtqUnit.kind`、hook 名、deploy binding 与下游消费口径是否一致。
2. review：越层、注册缺失、关量化不等价、验证缺失。
3. 局部 bug 修复：不改结构、不新增 wrapper、不动注册链。

任务同时含模型层实现和其它层改动时，拆开：模型层那部分转 `model-adapter`，dtype / 算法 / workflow / solver / deploy / public API 那部分留在 amct-coder，最后由 amct-coder 做契约对齐检查收口。

### 转交时要带上的信息

`model-adapter` 自己会读 `repo-map` 和 casebook（其固定流程第 1、2 步），不要替它读、也不要复制它的适配步骤。但转交前把下面三项写进委派说明，省掉它重新定位的成本：

1. 落点路径，精确到目录：例如 `amct_pytorch/common/models/llm/qwen/qwen3_next/`。
2. 命中的 casebook 个案路径（存在才写）：`.agents/docs/casebook/<family>/<model>.md`。已有个案的模型族包括 qwen（`qwen3-dense` / `qwen3-moe` / `qwen3.5-3.6` / `qwen3-next-80b-a3b-instruct`）、deepseek（`deepseek-v3.2` / `deepseekv4`）、glm（`glm-5.1`）、hunyuan（`hy3-preview`）、longcat（`longcat`）。
3. 本次任务里非模型层的部分（如果有），以及回来做契约对齐时要核对的契约项，见 [llm-model-unit.md](llm-model-unit.md) §契约清单。

判不清时，先答 [llm-model-unit.md](llm-model-unit.md) §适配清单 的 7 问：全部落在模型层就转 `model-adapter`。

## 不委派的编码场景

这些任务继续由 `amct-coder` 处理：

- 修 `amct_pytorch/common/models/llm/...` 的局部 bug：权重 key 拼错、kwargs 漏传、命名不一致等不改结构的修复。
- 修复算法注册、`targets`、导入注册链。
- 修复 bit policy / dtype 策略代码。
- 修复 PTQ 参数保存/加载的小范围代码问题。
- 做 AMCT Python 代码 review，重点查越层、注册缺失、验证缺失；模型层代码和 `model-adapter` 的产出也由 amct-coder review。

## 委派时的回复要求

委派前先简短说明原因，例如：

- “这是 NPU 算子开发，涉及 Ascend C kernel 和 torch.ops 注册，应使用 `amct-ops-dev`。”
- “这是跑 PTQ 实验，不是编码改动，应使用 `quant-run`。”
- “这是直转结果判读，应使用 `direct-quant-eval`。”

如果用户的请求同时包含编码和实验，先完成编码边界与实现，再把实验阶段明确交给对应 skill。

## 委派完成标准

同时满足下面条件才算完成委派：

1. 明确说明为什么当前动作属于外部 skill。
2. 实际加载并遵循外部 skill 的固定流程，而不是只给出 skill 名。
3. 由外部 skill 形成其职责内的计划或实现；amct-coder 不越权代写模型结构、kernel 或实验方案。
4. 混合任务完成后，由 amct-coder 只对跨层契约做收口 review。

同一会话已经读取过外部 skill 所需且未变化的资料时可以复用，但要说明复用内容，并重新核对当前任务的输入和范围。

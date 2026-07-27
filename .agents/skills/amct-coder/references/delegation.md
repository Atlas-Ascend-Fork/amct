# 委派规则

## 先判断任务类型

`amct-coder` 是编码总入口，但不是所有 AMCT 任务都应该在这里执行。任务实质不是编码，或已有更专用 skill 时，按本文件委派。

## 委派表

| 用户需求 | 使用 skill | 处理方式 |
| --- | --- | --- |
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

`model-adapter` 是更窄的模型适配实现 skill。用户明确要求“新增模型 adapter / 补 wrapper / 注册模型 / 定义 PTQ unit”时：

1. 先用 `amct-coder` 判断任务是否只属于模型层，是否触及 dtype/algorithm/workflow/deploy。
2. 若确实是纯模型适配，进入 `model-adapter` 固定流程。
3. 若同时涉及 dtype、算法、数据、solver、deploy，继续使用 `amct-coder` 的多个子能力 playbook。

## 不委派的编码场景

这些任务继续由 `amct-coder` 处理：

- 小范围修复 `amct_pytorch/common/models/llm/...` 的 adapter bug。
- 修复算法注册、`targets`、导入注册链。
- 修复 bit policy / dtype 策略代码。
- 修复 PTQ 参数保存/加载的小范围代码问题。
- 做 AMCT Python 代码 review，重点查越层、注册缺失、验证缺失。

## 委派时的回复要求

委派前先简短说明原因，例如：

- “这是 NPU 算子开发，涉及 Ascend C kernel 和 torch.ops 注册，应使用 `amct-ops-dev`。”
- “这是跑 PTQ 实验，不是编码改动，应使用 `quant-run`。”
- “这是直转结果判读，应使用 `direct-quant-eval`。”

如果用户的请求同时包含编码和实验，先完成编码边界与实现，再把实验阶段明确交给对应 skill。

# 职责交界

## 使用原则

`amct-coder` 只承接 AMCT 源码编码、调试和 review。开始任务时以当前会话已发现 skill 的 `name + description` 为外部职责事实源；本文件不复制完整 skill 清单，只说明最容易与编码任务混淆的稳定交界。

先区分用户要的是实现代码、评审拟议落点，还是实际执行操作：

| 动作 | 处理方式 |
| --- | --- |
| 任务实质已被更专门 skill 完整覆盖 | 直接加载该 skill，不预读 amct-coder 的 repo-map 或场景手册 |
| review 指定 workflow/solver/quant module 等落点是否合理 | 用最小资料审查当前落点；正确职责明确后继续对应流程 |
| 局部 bug 报告，根因尚未确认 | 先沿实际数据链核验；根据确认后的改动实质决定职责 |
| 同时包含编码和操作流程 | 按职责拆开，各自完成后只核对共享契约和结果接口 |

## 高歧义交界

| 交界 | amct-coder 承接 | 更专门职责 |
| --- | --- | --- |
| 模型层 | 已有 adapter/wrapper 的局部非结构 bug、契约 review | 新模型注册、`PtqUnit`/quant block、attention/MLP/MoE wrapper 的新增或结构性修改由 `model-adapter` 承接 |
| NPU 算子 | `amct_pytorch` 调用、可选 import、fallback 和报错时机 | Ascend C kernel、C++ extension、schema、Meta、构建和 wheel 由 `amct-ops-dev` 承接 |
| eval/extract/PTQ | workflow、data I/O、PPL 等实现代码 | 实际运行命令、生成实验结果和结果判读由当前发现的执行或 reviewer skill 承接 |
| deploy | deploy workflow、safetensors/index/config 等实现代码 | 实际生成 deploy-ready 目录和交付文档由当前发现的导出 skill 承接 |

上述名称只用于说明与编码层直接相邻的稳定边界。其它能力直接按当前已发现 skill 的 description 判断；新增或删除外部 skill 不要求同步修改本文件。

## 模型层契约

新增或结构性修改 `amct_pytorch/common/models/llm/...` 时进入 `model-adapter`。amct-coder 在模型层只保留：

1. 核对 `PtqUnit.kind`、hook 名、deploy binding 与下游消费口径。
2. review 越层、注册缺失、关量化不等价和验证缺失。
3. 修复不改结构、不新增 wrapper、不动注册链的局部 bug。

转交模型层任务时提供已确认的目标、候选落点和需要对齐的上下游契约。模型经验按 [casebook 索引](../../../docs/casebook/README.md) 或目录动态检索，不在这里维护模型族和个案快照。

## 继续执行与询问

BLOCKER 只停止错误落点或错误方案。正确替代路径明确、仍属于用户目标且不需要新授权时，直接进入对应流程继续完成任务。只有输入输出、数值语义、schema、兼容范围或用户控制方式等关键契约缺失，且不同答案会改变实现方案时，才询问用户；询问前先按 SKILL.md「契约缺失时的处理形态」确认该问题不能由源码、注册入口或测试消歧。

进入更专门 skill 后要实际加载并遵循其流程，不能由 amct-coder 模仿实现后只口头说明“已经委派”。混合任务完成后，只对共享签名、数据格式、保存/加载和验证证据做收口，不重新接管其它 skill 的内部职责。

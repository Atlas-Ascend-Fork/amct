# Agent Docs

`docs/agent/` 用来保存仓库里相对稳定、适合长期复用的知识，不直接承担执行型约束。

## 内容组成

- `repo-map.md`
  - 记录代码结构、关键模块和锚点文件
- `amct-onepage.md`
  - 只保留仓内已确认事实的单页架构说明，适合新人速读和对接前快速统一口径
- `amct-to-infer-execution-flow.md`
  - 将 `amct_llm -> infer` 对接部分单独抽成执行流图，方便按步骤落地接入
- `amct-10min-quickstart.md`
  - 面向新人的 10 分钟速读版，帮助快速建立对主链路、核心抽象和导出物的第一印象
- `amct-quantization-framework-guide.md`
  - 面向新人和下游推理框架的量化框架原理、导出契约与对接指南
- `infer-integration-checklist.md`
  - 面向推理框架接入方的执行清单，帮助快速判断量化工件是否可接、如何接、缺什么
- `casebook/`
  - 记录模型适配经验、量化经验和典型问题

## 使用原则

- 这里放稳定知识，不放一次性的实验过程
- 执行型规则、步骤和分流逻辑放在 `skills/`
- 如果某类经验已经足够稳定，适合长期复用，再从 `skills/references/` 沉淀到这里

## 维护边界

- 代码结构、模块职责或目录边界发生变化时，更新 `repo-map.md`
- 某个模型系列形成稳定经验后，更新 `casebook/`
- 单次失败实验或暂未复现的现象，不默认写入这里

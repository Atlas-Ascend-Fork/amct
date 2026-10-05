---
name: amct-coder
description: |
  AMCT 仓源码级编码辅助入口。用于实现规划、代码修改、源码行为诊断和代码 review，覆盖 LLM PTQ CLI 及相关 workflow/solver/data、量化模块（dtype/quant module）与 LLM PTQ 算法、deploy、public API 及其测试/文档的修改位置、调用链和跨端约定检查。不用于量化实验执行、精度/算法收益判读、模型结构适配、classic/graph_based 或 NPU 算子实现。
---

# AMCT Coder

## 文件索引

| 类型 | Reference | 用途与加载时机 |
| --- | --- | --- |
| 项目规范 | [coding-rules.md](references/coding-rules.md) | 编码任务与规范合规 review 都整份读（含「贡献门槛对编码的约束」）；只做规划、诊断、代码行为 review 时不读 |
| 主场景 | [quant-dtype-module.md](references/quant-dtype-module.md) | 量化 dtype 与通用 quant module 开发/review |
| 主场景 | [llm-algorithm.md](references/llm-algorithm.md) | LLM 量化算法开发/review |
| 主场景 | [model-adapter-review.md](references/model-adapter-review.md) | 已有 adapter 局部非结构修复与跨端约定 review |
| 主场景 | [llm-cli-dataflow.md](references/llm-cli-dataflow.md) | LLM CLI 架构与 eval/extract/ptq/deploy 数据流开发/review |
| 主场景 | [llm-solving-lifecycle.md](references/llm-solving-lifecycle.md) | PTQ 求解侧生命周期：solver、granularity/unit 前置检查、参数保存回载 |
| 主场景 | [llm-deploy.md](references/llm-deploy.md) | deploy 实现代码：blockwise/tensorwise 导出、分片、index 与 config |
| 主场景 | [public-api-packaging.md](references/public-api-packaging.md) | public API 导出/懒加载、packaging 清单、experimental 开关与文档承诺的开发/review |
| 交付验证 | [validation.md](references/validation.md) | 改完读并执行最小验证（含验证失败的处理）；诊断/review 的结论定级读「证据等级」 |
| 示范 | [walkthrough.md](references/walkthrough.md) | 诊断/review/规划首次进入时读（示范什么时候停止查资料、开始交付）；编码任务不默认读，只在读了很多资料仍未收敛时读一次校准 |
| 仓内资料 | [casebook](../../docs/casebook/README.md) | 模型结构、checkpoint/config 或模型相关精度异常按 L1 -> L2 -> L3 读取 |

## 选主场景手册

多个主场景同时命中时只加载一本；只有真实跨端影响已经成立，才把第二个主场景作为辅助手册加载。

命中后按节读取，不要求整份全文：

- **每次都读**：职责边界、禁止项（全局约束不随场景豁免）。
- **命中才读**：调用链、实现定位、当前事实、实现规则、前置检查、测试矩阵、自验，以及各手册的场景专属节——按命中的场景或动作取；诊断类任务可只读目标节。

需要核对跨端约定时，把对应主场景手册当辅助资料读取。

未命中现有手册时不要强行套入最相近项：以 repo-map 判断改动所属层，据此形成任务规划。

场景手册里的路径只是当前工作区的提示，可能已失效。

## 按需查阅的细则

| 命中条件 | 读哪一节 |
| --- | --- |
| 要定 repo-map 读取范围，或判断改动面是否已被需求点名到具体文件或函数 | [仓库边界](references/conditional-rules.md#仓库边界) |
| 命中重点核对对象或跨端两端，要决定核对做到多重 | [跨端约定核对的做法](references/conditional-rules.md#跨端约定核对的做法)（重点核对对象定义、比例原则） |
| 要把手册「当前事实」当判断依据，或核对命令与断言不符 | [当前事实与核对命令纪律](references/conditional-rules.md#当前事实与核对命令纪律) |

## 交付说明

按有无代码改动选一类写，只写命中项：

- **代码有改动**：改动落在哪一层、改动文件、复用了哪些既有抽象、跑了/没跑哪些验证（没跑的写阻塞原因）、事实与待确认项、docs 要不要更新。
- **代码无改动**：边界、事实、方案、已核对的证据与缺失的证据；开头一句话说清**核对到哪条链路、哪一环没验到**。报链路，不报读了哪些文件名的清单。规范合规检查只写检查项、发现、证据，不报链路环点。

**不许沉默**：跑了/没跑哪些验证、有没有波及本次改动范围之外的共用代码（波及就点名到具体层）、有哪些待确认项——没有内容也要显式写一句。其余项不涉及就不写，不用 `n/a` 占行（跨端约定状态表例外，按 [conditional-rules「比例原则」](references/conditional-rules.md#比例原则) 写 `n/a` + 原因）。

**写结论，不写动作**：命中跨端约定核对时写兼容性影响、写入端和读取端是否一致，不写「本次叠加了跨端约定核对」。

## 禁止项

- 不在 CLI、workflow、solver 或通用 quant module 中加入模型专属分支。

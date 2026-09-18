# 职责交界：delegation

## 使用原则

`amct-coder` 只承接 AMCT 源码编码、实现规划、调试和 review。开始任务时以当前会话已发现 skill 的 `name + description` 为外部职责事实源（完整优先级：用户指定 > 专门 skill description > 本 skill 流程，见 [SKILL.md](../SKILL.md) 职责归属节）；本文件不复制完整 skill 清单，只说明最容易与编码任务混淆的稳定交界。

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
| 模型层 | 已有 adapter/wrapper 的局部非结构 bug、契约 review | 新模型注册、`PtqUnit`/quant block、attention/MLP/MoE wrapper 的新增或结构性修改由 `model-adapter` skill 承接 |
| NPU 算子 | `amct_pytorch` 调用、可选 import、fallback 和报错时机 | Ascend C kernel、C++ extension、schema、Meta、构建和 wheel 由 `amct-ops-dev` 承接 |
| classic / graph_based | 其 **public 面**：导出与懒加载（`__getattr__`）、packaging 清单、文档承诺的一致性（主手册 `public-api-packaging`） | 具体量化实现（classic custom algorithm/config、graph pass、`quantize`/`convert` 的算法语义）无对应 skill，本 skill 也不承接——frontmatter 的「不用于 classic/graph_based」指实现侧，不指 public 面 |
| eval/extract/PTQ | workflow、data I/O、PPL 等实现代码 | 实际运行命令、生成实验结果和结果判读由当前发现的执行或 reviewer skill 承接 |
| deploy | deploy workflow、safetensors/index/config 等实现代码 | 实际生成 deploy-ready 目录和交付文档由当前发现的导出 skill 承接 |

上述名称只用于说明与编码层直接相邻的稳定边界。其它能力直接按当前已发现 skill 的 description 判断；新增或删除外部 skill 不要求同步修改本文件。

## 专门 skill 路由例子

以下是高频请求形态，用来帮助判断“继续用 amct-coder”还是“切到专门 skill”。具体以当前会话发现的 skill 描述为准。

| 用户说法或目标 | 处理方式 | amct-coder 保留职责 |
| --- | --- | --- |
| “跑一下直转评测 / extract_ptq_data / PTQ / 算法评测” | 转执行类 skill，例如 `quant-run` | 若命令失败暴露实现 bug，再回源码诊断 |
| “这些 ppl 是否达标 / 直转掉点多少” | 转结果判读类 skill，例如 `direct-quant-eval` | 只在口径或日志字段来自代码问题时诊断 |
| “这个 PTQ 算法相对直转有没有收益” | 转算法收益判读类 skill，例如 `algorithm-validation` | 只核对算法结果文件或参数回载代码是否可信 |
| “给这个已适配模型推荐下一步算法” | 转推荐类 skill，例如 `algorithm-recommendation` | 只核对仓库算法注册和适用范围事实 |
| “端到端量化这个模型 / 只适配 / 只导出” | 转编排类 skill，例如 `quant-workflow` | 只处理编排中发现的源码缺陷或契约 review |
| “新增某模型 adapter / attention wrapper / MoE wrapper” | 转 `model-adapter` skill | review 分层、`PtqUnit`、关量化等价和下游契约 |
| “写 NPU kernel / schema / Meta / CMake / wheel” | 转 `amct-ops-dev` | 只处理 `amct_pytorch` Python 侧调用和 fallback |
| “导出 deploy-ready 目录并写交付文档” | 转 `deploy-export` | 只处理 deploy 实现代码、配置生成和 public API 问题 |

切换不是结束任务。若专门 skill 的输出需要代码修复、落点 review 或契约收口，回到 amct-coder 只处理这些源码级部分。

## 模型层契约

新增或结构性修改 `amct_pytorch/common/models/llm/...` 时进入 `model-adapter` skill。amct-coder 在模型层只保留：

1. 核对 `PtqUnit.kind`、hook 名、deploy binding 与下游消费口径。
2. review 越层、注册缺失、关量化不等价和验证缺失。
3. 修复不改结构、不新增 wrapper、不动注册链的局部 bug。

转交模型层任务时提供已确认的目标、候选落点和需要对齐的上下游契约。模型经验按 [casebook 索引](../../../docs/casebook/README.md) 或目录动态检索，不在这里维护模型族和个案快照。

## 继续执行与询问

BLOCKER 的处理形态与何时询问用户以 [process-rules「全局 BLOCKER」与「契约缺失的处理形态」](process-rules.md#契约缺失的处理形态) 为唯一出处。本文件的补充只有一条：询问用户前，先确认该问题不能由源码、注册入口或测试消歧（与 SKILL.md 核心流程「事实纪律」的固定消歧顺序一致），且只问编码与操作流程交界处的契约，不把 amct-coder 内部可自查的问题带出。

进入更专门 skill 后要实际加载并遵循其流程，不能由 amct-coder 模仿实现后只口头说明“已经委派”。混合任务完成后，只对共享签名、数据格式、保存/加载和验证证据做收口，不重新接管其它 skill 的内部职责。

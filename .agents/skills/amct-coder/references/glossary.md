# 术语表：glossary

本文件是本 skill 自造词的唯一出处。各 reference 与 [SKILL.md](../SKILL.md) 直接使用这些词、不再解释；遇到不明术语回这里查。本表是查阅资料，按需查单行、不整读。

| 术语 | 含义 |
| --- | --- |
| 落点层 | 改动应落在哪一层：CLI / workflow / solver / quant module / dtype / algorithm / adapter / deploy / public API。落点错层是本仓最主要的错误来源，SKILL.md 轻级判据②即约束它。 |
| 主手册 | 本次任务唯一的主场景 reference（SKILL.md「场景 Reference 导航」的「主场景」行）。一个任务只选一个，第二个只能作辅助；豁免情形见 SKILL.md 首段直达卡。 |
| 门禁对象 | 命中即强制升重级的对象：新 dtype / 新算法 / 新 solver / 新 granularity、复合 unit、skip/partial result。共同点是「注册成功不等于可用」。 |
| 门禁表 / 契约状态模型 | 重级任务逐项填写的表，把「当前仓库事实」与「目标契约」分账，未冻结项标 `pending`。填法受 [process-rules「重级的比例原则」](process-rules.md#追加判据与各级流程) 约束。 |
| 快照 / 核对命令 | 手册里对当前源码状态的断言，附一条可执行的 `rg`/`grep` 命令。冲突时以命令结果为准，不以文档为准；引用快照事实作落点依据前先跑它。 |
| 事实纪律 Q1/Q2/Q3 | 待核实契约的消歧顺序：源码能查就查（Q1）→ 不影响公开契约的局部选择自行决定（Q2）→ 查不到且不同答案产出不兼容实现才问用户（Q3）。第一个「是」即停。规则正文以 SKILL.md「各级都要做」为唯一出处。 |
| 契约缺失形态 | Q3 触发后向用户提问的规定形态：封闭选项、一次问完、无依据不伪造推荐。操作细则见 [process-rules「契约缺失的处理形态」](process-rules.md#契约缺失的处理形态)。 |
| 双端验证 | producer 与 consumer 两侧都验（保存侧与回载侧、CLI 侧与 registry 侧）。只验一侧会漏掉「静态可写、运行不可读」。 |
| 消费面 | 一个能力的下游接入点集合（registry / CLI / bit policy / quantizer / deploy / algorithm / model 分支）。新增能力要逐面收口，不能只证明定义存在。 |
| 敏感层 | 改动影响面超出本次落点的层，交付说明须单独点明；枚举以 SKILL.md「交付格式」为准。 |
| BLOCKER | 手册中标注的硬约束。语义是「这个方案错了」，不是「任务停止」：替代路径明确且在用户目标内就直接继续。全局清单以 [process-rules「全局 BLOCKER」](process-rules.md#全局-blocker) 为唯一出处，轻级可违反的三条内联在 SKILL.md「各级都要做」。 |
| 收束 | 停止扩张上下文、开始交付的时刻。判据是停止条件（见 [process-rules「渐进披露与读取预算」](process-rules.md#渐进披露与读取预算)），不是读了多少资料。 |

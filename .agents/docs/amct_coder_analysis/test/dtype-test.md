# dtype 子能力有无 Skill 规划行为对比

## 1. 测试说明

本轮比较 Agent 在 dtype 任务中的规划路线，重点观察：

- 是否先判断职责和代码体系；
- 是否区分仓库事实与目标契约；
- 未知项是否保持 `pending`；
- 是否核对 registry、CLI、bit policy、quantizer 和 deploy 消费面；
- 是否在证据充分后停止扩张。

共完成 5 组有 / 无 `amct-coder` 的成对测试。原始长回答归档于 [dtype-test-raw.md](dtype-test-raw.md)。

### 证据限制

- 本轮是同会话 warm 对比，有 Skill 回答能够看到或复用无 Skill 阶段的结论，不是严格 cold A/B。
- 每题只有一次结果，未记录统一的模型版本、commit、token、时间和完整工具轨迹。
- 原计划中用于验证“信息充分时能否正常收敛”的 `q6s` 尚未测试。

因此，本轮只能判断规划行为是否方向性改善，不能认定 dtype 子能力已经通过稳定回归。

## 2. 两种情况总结

### 无 Skill

典型路线：

```text
题面关键词
  -> 搜索相似 dtype 和硬编码分支
  -> 默认需要新增 dtype
  -> 展开文件、重构和端到端检查清单
```

优点：能发现 registry import、CLI 白名单、bit policy、quantizer 和 deploy 静默错分等真实问题。

主要问题：经常在数值语义、role、粒度和 deploy 范围未确定时，就锁定新类、文件名、注册键、算子骨架或跨层重构；搜索命中也容易被当成必改清单。

### 有 Skill

典型路线：

```text
判断职责和代码体系
  -> 读取 dtype 主手册与最小源码锚点
  -> 先判断复用、别名、迁移还是新增
  -> 分离当前事实与目标契约
  -> 未知项保持 pending，按条件分支规划
```

优点：规划起点从“怎么新增 dtype”前移到“是否需要新增 dtype”；能更稳定地区分注册存在与实际消费闭环，并在 NPU 场景中划清 Python 语义和 kernel 实现边界。

仍有问题：答案依然偏长；少数题目会把 `pending` 项在后文提前收口；当前记录不足以判断 Reference 读取预算是否真的改善。

## 3. 逐题精炼对比

### D1：只有产品名称的新格式

测试语：

> AMCT 准备支持暂定名为 `nova-float` 的新量化格式，但目前没有更详细的格式说明。请给出接入前规划，只读分析。

无 Skill：

- 识别出数值语义缺失和 CLI/deploy 的硬编码风险。
- 仍默认需要新增 dtype，并提前确定命名映射、dtype 骨架、NPU 名称及多处全局重构。
- 未知契约虽然被列出，但没有阻止方案形成唯一主路线。

有 Skill：

- 先判断它是现有语义别名、classic 能力迁移，还是全新 dtype。
- role、bit、粒度、数值语义、deploy 和 NPU 范围保持 `pending`。
- 只给出三条条件路线，不把候选文件和注册名写成最终落点。

结论：有 Skill 明显改善，dtype 状态机和既有语义复用门生效。

### D2：错误的复制复用建议

测试语：

> 有人建议新增 `orbit4` 时直接复制现有 4 bit 实现并替换注册名，认为 CLI 和部署会自动支持。请评审方案。

无 Skill：

- 正确指出 CLI 白名单和 deploy 精确分支不会随 registry 自动更新。
- 能识别 INT4 与 MXFP4 的编码、packing 和下游语义不同。
- 但仍建议先建新文件骨架并实施跨 CLI/deploy 的去硬编码重构。

有 Skill：

- 先比较 `orbit4` 与现有格式的真实数值语义，再决定别名、复用或新增。
- 使用消费面表区分 registry、CLI、bit policy、quantizer 和两种 deploy 路径。
- 将算法和模型侧搜索命中视为兼容性信号，不直接列为必改项。

结论：两组都能否决错误方案，有 Skill 对修改范围和未知契约的控制更好。

### D3：部分契约已经确定

测试语：

> 新增 `quark4`，已确认仅用于权重、固定 4 bit，不量化 activation 和 cache；编码、粒度和 deploy 范围未知。请给出实现前规划。

无 Skill：

- 正确采用 weight-only 和 4 bit，并发现 `is_act=True` 构造及非目标层 16 bit 直通约束。
- 能区分数值和 deploy 的阻塞项。
- 仍提前规划新文件、CLI 接入以及可选 NPU 路径。

有 Skill：

- 已知 role/bit 正常收口，编码、粒度和 deploy 保持 `pending`。
- 将 activation 构造兼容、16 bit 直通和 deploy 参与度分开处理。
- 但根据命名惯例建议把注册名改成 `quark`，证据不足以作为默认选择。

结论：有 Skill 有改善，但仍存在从仓库惯例推导目标命名的过早决策。

### D4：只有类、注册和数值单测

测试语：

> `luma` 方案只有新的 dtype 类、一次注册和数值单测，作者认为已经完成框架接入。请判断结论并给出检查计划。

无 Skill：

- 正确追到 `register_dtype()` import、CLI、quantizer 和 deploy 活跃链，否决“已经完成”。
- 能说明直接 import 的数值单测可能掩盖生产注册链缺失。
- 检查面过大，混入题面未要求的 NPU、模型分支和全局重构。

有 Skill：

- 明确区分 registry 已接通、CLI 不可达、quantizer 契约未覆盖和 deploy 是否参与仍未知。
- 使用当前事实 / 目标契约表保留 role、bit、粒度和 deploy 的 `pending` 状态。
- 但一边把 CLI 是否公开标为 `pending`，一边又把添加 CLI choice 列为可立即修改，前后不完全一致。

结论：有 Skill 的证据分层更清楚，但 `pending` 状态在后续计划中的一致性仍需加强。

### D5：dtype 与 NPU 加速的职责交界

测试语：

> 为一种新量化格式同时提供 Python 参考实现和昇腾 NPU 加速，但编码、支持对象和部署格式仍在讨论。请规划当前工作和职责划分。

无 Skill：

- 直接按 LLM `DTYPE_REGISTRY + CLI + deploy + amct_ops` 路线展开。
- 提前锁定文件、算子名、schema/Meta 骨架和两项通用重构。
- 没有先判断目标属于 LLM new-path、classic helper，还是两者。

有 Skill：

- 先识别 LLM registry dtype 与 classic HiFloat helper 两种现存形态，并要求先选择目标体系。
- 将 Python 参考实现作为数值基线，把 kernel、C++ schema、Meta、构建和 NPU 测试交给 `amct-ops-dev`。
- 在数值、role 和 shape 未冻结前不确定 schema；ops 目录结构仅能作为候选布局。

结论：有 Skill 明显改善了代码体系判断和跨 Skill 职责边界。

## 4. 横向结论

| 观察项 | 无 Skill | 有 Skill |
| --- | --- | --- |
| 规划起点 | 默认新增 dtype | 先判断复用、别名、迁移或新增 |
| 未知契约 | 会列出，但常被后续方案隐式冻结 | 多数保持 `pending` 并形成条件分支 |
| 消费面 | 依赖搜索结果补清单 | 按 registry、CLI、quantizer、deploy 逐项取证 |
| 修改范围 | 容易将搜索命中列为必改项 | 更强调 owner 和真实跨契约影响 |
| NPU 边界 | 容易提前展开算子骨架 | 先冻结 Python 语义，再路由 `amct-ops-dev` |
| 证据等级 | 能说明未运行，但表达不稳定 | 较稳定地区分源码、单测、集成和真实环境 |
| 输出效率 | 长 | 仍然长，尚无读取轨迹证明改善 |

整体上，D1、D2、D5 的路线改善最明显；D3、D4 仍暴露了“未知项在后续计划中被提前收口”的残余问题。

## 5. 下一步

1. 补测 `q6s`，验证信息充分时 Agent 能否直接收敛到最小改动，而不是机械保留 `pending`。
2. 在物理隔离的新会话中重跑 D1、D3、D4，每题至少三次。
3. 分别验证：D1 未知项全程保持 `pending`；D3 不推导题面未给出的命名；D4 CLI/deploy 严格按目标参与范围决定。
4. 记录模型、commit、实际读取资料、token、时间和 reviewer 评分，再判断 dtype 子能力是否通过回归。

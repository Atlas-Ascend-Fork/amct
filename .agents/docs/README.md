# Agent Docs

`.agents/docs/` 用来保存仓库里相对稳定、适合长期复用的知识，不直接承担执行型约束。

## 内容组成

- `repo-map.md`
  - 记录代码结构、关键模块和锚点文件
- `casebook/`
  - 记录模型适配经验、量化经验和典型问题

## 使用原则

- 这里放稳定知识，不放一次性的实验过程
- 执行型规则、步骤和分流逻辑放在 `skills/`
- 如果某类经验已经足够稳定，适合长期复用，再从 `skills/quant-tools/references/` 沉淀到这里

## 文档写回触发

agent 完成一轮工作后据此维护本 docs（执行型流程在 `skills/`，这里只放稳定知识与维护策略）：

- **`repo-map.md`**：`register_llm_models()` 注册集 / solver 主注册链 / workflow 主路径语义 / `BaseModel·PtqUnit·quant_apply·quant_base` 复用边界变化，或现状已与代码不一致。
- **系列 `casebook/<vendor>/README.md`**：某模型首次拿到稳定闭环（BF16 baseline + 关闭量化等价 + 最小 PTQ smoke），或系列默认起点/敏感模块/升级策略/常见问题需补订，或新结果推翻系列默认经验。
- **个案 `casebook/<vendor>/<case>.md`**：新结构模式 / 新量化或 PTQ 路线 / 代表性可复用问题 / 明显偏离系列经验 / 已有个案需按新结论修订；**仅本模型独有、不可迁移**的坑留这层。
- **经验库 `casebook/{cross-model,structure-family}-pitfalls.md`**：跨网络通用 / 结构家族通用的可复用坑写回；**三层晋升护栏见 [casebook/README.md](casebook/README.md) 维护规则**（单例先 L3、同家族 ≥2 升 L2、跨家族升 L1，防过度泛化）。
- **`README.md`（本文件）**：docs 层级 / 职责边界变化，或目录结构与真实不一致。
- **默认不写**：单次失败 / 不稳定结果；BF16 baseline 自身不稳；只修局部 wrapper bug 不重写 repo-map；本轮无新增可复用信息（需说明无需更新）。

# Casebook

`casebook/` 沉淀模型适配与量化中**可复用、且不易解决**的经验：后续其他网络会再遇到的 hard bug、适配重点、关键精度，而不是一次性小问题或 agent 环境/过度推断类问题。

## 三层抽象（agent 适配新模型的检索主轴）

经验按"普适程度"分三层，agent 适配新模型时**自外向内**取用：

1. **L1 跨网络通用** [`cross-model-pitfalls.md`](cross-model-pitfalls.md) —— 与具体结构无关，**任何新模型都先通读**。
2. **L2 结构家族通用** [`structure-family-pitfalls.md`](structure-family-pitfalls.md) —— 仅对某类结构成立（MoE / 混合 attention / 自定义 modeling / MLA）。按**触发信号**判断新模型属哪几类（可多属），**叠加** L1 读对应小节。
3. **L3 模型专属** —— 各 `<vendor>/<model>.md` 个案，只放该模型/家族独有、不可迁移的坑。

> agent 用法：新模型进来 → 读 L1 → 按 config/checkpoint 信号选中的 L2 家族（取并集）→ 找结构最近的 L3 个案起步。没有专属个案时，L1+L2 仍给强先验。

## 组织方式

L3 个案**按厂商系列**组织，与源码 `amct_pytorch/common/models/llm/<vendor>/` 对齐。**同一源类、或纯子类（零/单方法 delta）的不同尺寸模型合并为一个个案**（同结构归一）。

```text
casebook/
  cross-model-pitfalls.md   # L1
  structure-family-pitfalls.md   # L2
  case-template.md     # L3 模板
  qwen/      deepseek/      glm/      longcat/    # L3 + 各系列 README
```

### L3 个案导航（按系列）

- **qwen/**：`qwen3-dense`(4B/8B)、`qwen3-moe`(30B-A3B/235B-A22B)、`qwen3.5-3.6`(35B-A3B/35B)、`qwen3-next-80b-a3b-instruct`
- **deepseek/**：`deepseekv4`、`deepseek-v3.2`
- **glm/**：`glm-5.1`（继承 DeepseekV32，瘦案例）
- **longcat/**：`longcat`(Flash-Lite/Next)

> 个案均对应源码 `register_llm_models()` 已注册的 adapter（casebook ⊆ 已适配）。

## 维护规则

- 只记**可复用、难解**的结论；一次性小问题、agent 环境/过度推断不入库。
- **分层归档**：新坑先判普适程度——普适→L1、某结构家族→L2、仅此模型→L3。
- **≥2 例才晋升**：单模型新发现默认先留 L3（标"待复现"）；同家族 ≥2 模型复现才升 L2，跨家族复现才升 L1。**防单例过度泛化**（别从 1 个 MoE 断言"所有 MoE"）。
- **清单非全集**：L1/L2 的自检清单是先验、不是闸门；清单外现象照常据证据诊断 + 归档（必要时新增家族/个案）。
- **开放分类**：L2 家族不锁死枚举，出现新结构模式（如 SSM/新稀疏 attention）就新增小节 + 触发信号。
- **合并依据 = 继承 + override 审计**（`wc -l`/grep defs/读非透传方法），不是继承声明本身：`pass` 子类零 delta 直接合；只 override 个别方法的（如 LongCat-Next 仅 `empty_weights_model`）合并并把那一处写成 delta 段；override 体量大、语义偏移多的不合。

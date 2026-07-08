# <模型名> · 系列 <vendor> · 结构 <dense/moe/mla-moe/vl-moe/混合>

> **速览**：<adapter 路径 / 是否纯复用主链 / 首推方案+delta / 核心坑或状态——一句话>。<同源类/纯子类的不同尺寸合并到一个文件，多尺寸分小节>。
> **触发信号**：<config/checkpoint 上可检测的特征——据此判断属哪些 L2 家族>。读 L2 `../structure-family-pitfalls.md`（命中家族）+ L1 `../cross-model-pitfalls.md`。（路径以个案实际所在 `<vendor>/` 子目录为基准，即 `../`。）

## 结构与适配要点

- <架构判断：decoder 类型、是否 MoE、attention 形态、关键 config、特别之处（vs 系列默认路径）>。
- **参考与差异**：最近参考 <案例>；关键差异 <本案不同点>。
- **复用**：<复用的仓内抽象>。**新增**：<模型专属实现 / “无（纯复用）”>。
- **起步复用清单**（下一条同系列/同结构）：<从哪些文件/类起步 + 首轮起步方案>。

## 适配验证结论

- 标准三步闭环（BF16 baseline → 关闭量化浮点等价 → 最小 PTQ smoke）<是否全部通过>。
- **模型特定结论**：<哪些部分对齐/敏感、关闭量化等价结果、最小 PTQ smoke 用的单元>。

## 关键陷阱（L3 模型专属；通用见 L1/L2）

> 只记**本模型独有、不可迁移**的坑（L3）。跨网络通用的引 L1 `../cross-model-pitfalls.md`、结构家族通用的引 L2 `../structure-family-pitfalls.md`（本案是否为其首遇例）；同家族 ≥2 模型复现的坑应上抽到 L2，别留在这里重复。agent 环境配置类、一次性小问题不入库。

- **<L3 陷阱一句话>** —— 现象：<报错/异常>。根因：<本模型独有原因>。处理：<修法>。教训：<一句>。
- <通用坑> → 见 L1 `../cross-model-pitfalls.md` · <条目> / L2 `../structure-family-pitfalls.md` · <家族>。

## 量化结论（+ 性能注意，可选）

- BF16 baseline（seq_len=4096）<ppl>；首轮直转 <方案> W?A?-<dtype>（<粒度>，<是否 PTQ>）`delta=<值>`（<是否达标>）。已落地粒度 / 升级路径。
- 性能注意（可选，无则删）：<如 MoE per-expert 小 M_eff 存疑、须 infer MoEGMM 实测>。

## 适配建议（下次同系列/同结构）

- 先参考：<本案 + 最近参考案例>。
- 先做什么：<抓 config.json / index.json、小 config 空载 smoke、逐层等价…>。
- **不建议**：<反模式，如靠 architectures 命名推断 checkpoint 结构、直接抄前代 quant 模块、一上来裁 kv_cache…>。

## 精度速查表

> ppl 口径 seq_len=4096。MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 |  |

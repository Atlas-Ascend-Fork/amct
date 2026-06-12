# Casebook

`casebook/` 用来沉淀模型适配和量化过程中的可复用经验。

## 记录内容

- 模型适配路径
- 量化起步方案与结果
- 典型问题与排查结论
- 成功路径和失败路径

## 组织方式

**主轴 = 网络结构**（量化策略由结构决定：dense→mlp/attn-linear；MoE→moe.routed/shared；MLA→attn-cache；VL→language_model scope）。厂商专属适配经验单独收在 `_vendor-notes/`。

```text
casebook/
  README.md
  dense/          # 标准 dense decoder（attn-linear + mlp）
  moe/            # 标准 MoE（FFN experts，routed/shared）
  mla-moe/        # MLA attention + MoE（deepseek 式）
  vl-moe/         # 多模态 VL + MoE（language_model + visual + mtp）
  diffusion/      # 视频扩散模型（attention 量化，指标体系不同）
  _vendor-notes/  # 厂商系列适配经验（adapter + 系列量化经验）
```

### 个案导航（按结构）

- **dense/**：`qwen3-4b`、`qwen3-8b`
- **moe/**：`qwen3-30b-a3b`、`qwen3-235b-a22b`、`qwen3-next-80b-a3b-instruct`、`longcat-flash-lite`、`longcat-next`、`glm-5.1`
- **mla-moe/**：`deepseekv4`、`deepseek-v2-lite`、`deepseek-r1`、`deepseek-v3.1`、`deepseek-v3.2`、`longcat-flash`、`kimi-k2-thinking`
- **vl-moe/**：`qwen3.5-35b-a3b`、`qwen3.6-35b`
- **diffusion/**：`HunyuanVideo`、`Wan2.2-t2v`（视频，指标体系不同）
- **_vendor-notes/**：`qwen`、`deepseek`、`glm`、`longcat`

## 维护规则

- 新结果与同结构已有经验一致时，优先更新对应结构目录下的个案；厂商通用经验更新 `_vendor-notes/<vendor>.md`
- 出现新的结构模式、量化路径或高价值问题时，再新增 `<structure>/<case>.md`
- 只记录可复用结论，不记录一次性的临时过程

## 精度横比总览

> 每模型一行：主指标基线（有 ppl 用 ppl，否则 mmlu 等）+ 已记录方案数 + 个案链接。完整 dtype×config sweep 见各个案的「精度速查表」。Diffusion 视频模型见 `diffusion/`。

| 结构 | 模型 | 基线 | 方案数 | 个案 |
| --- | --- | --- | --- | --- |
| dense | Qwen3-4b | ppl 12.377 | 9 | [↗](dense/qwen3-4b.md) |
| moe | Qwen3-235B-A22B | ppl 5.09 | 13 | [↗](moe/qwen3-235b-a22b.md) |
| moe | GLM-5.1 | ppl 2.5658 | 2 | [↗](moe/glm-5.1.md) |
| moe | Qwen3-Next | ppl 5.4483 | 12 | [↗](moe/qwen3-next-80b-a3b-instruct.md) |
| moe | longcat-flash-lite | ppl 14.6133 | 9 | [↗](moe/longcat-flash-lite.md) |
| moe | longcat-Next | ppl 17.518 | 9 | [↗](moe/longcat-next.md) |
| mla-moe | Deepseekv2-lite | ppl 6.31 | 6 | [↗](mla-moe/deepseek-v2-lite.md) |
| mla-moe | Deepseek-R1 | ppl 3.3303 | 3 | [↗](mla-moe/deepseek-r1.md) |
| mla-moe | Deepseek-v3.1 | mmlu 87.8 | 4 | [↗](mla-moe/deepseek-v3.1.md) |
| mla-moe | Deepseek-v3.2 | ppl 3.2133 | 3 | [↗](mla-moe/deepseek-v3.2.md) |
| mla-moe | longcat-flash | ppl 3.1214 | 2 | [↗](mla-moe/longcat-flash.md) |
| mla-moe | Kimi-K2 Thinking | ppl 2.1896 | 2 | [↗](mla-moe/kimi-k2-thinking.md) |
| vl-moe | Qwen3.5-35b | ppl 6.2486 | 9 | [↗](vl-moe/qwen3.5-35b-a3b.md) |
| vl-moe | Qwen3.6-35b | ppl 6.2799 | 9 | [↗](vl-moe/qwen3.6-35b.md) |

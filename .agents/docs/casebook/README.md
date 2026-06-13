# Casebook

`casebook/` 用来沉淀模型适配与量化中**可复用、且不易解决**的经验：主要是后续其他网络会再遇到的 hard bug 与适配陷阱，而不是一次性的小问题。

## 记录内容

- 适配中难解的 bug 与陷阱（结构 / 权重 / attention / MoE 等），后续同类网络可复用
- 量化起步方案与关键精度结论
- 典型问题的排查路径与失败教训

## 组织方式

**主轴 = 厂商系列**，与源码 `amct_pytorch/common/models/llm/<vendor>/` 对齐。

```text
casebook/
  README.md
  qwen/        # qwen3 / qwen3_5 / qwen3_6 / qwen3_next
  deepseek/    # deepseek_v3_2 / deepseek_v4
  glm/         # glm5
  longcat/     # longcat_lite / longcat_next
```

每个 `<vendor>/README.md` 是该系列的适配经验总览，`<vendor>/<model>.md` 是个案。

### 个案导航（按系列）

- **qwen/**：`qwen3-4b`、`qwen3-8b`、`qwen3-30b-a3b`、`qwen3-235b-a22b`、`qwen3-next-80b-a3b-instruct`、`qwen3.5-35b-a3b`、`qwen3.6-35b`
- **deepseek/**：`deepseekv4`、`deepseek-v3.2`
- **glm/**：`glm-5.1`
- **longcat/**：`longcat-flash-lite`、`longcat-next`

## 维护规则

- 只记录**可复用、难解**的结论：后续其他网络会用到、且不是顺手就能解的 bug；一次性小问题不入库。
- 新结果与同系列已有经验一致时，更新对应 `<vendor>/` 下个案或系列 `README.md`。
- 出现新的难解陷阱、量化路径或失败教训时，新增 `<vendor>/<case>.md`。

## 精度横比总览

> 每模型一行：主指标基线 + 已记录方案数 + 个案链接。完整 sweep 见各个案的「精度速查表」。

| 系列 | 模型 | 基线 | 方案数 | 个案 |
| --- | --- | --- | --- | --- |
| qwen | Qwen3-4b | ppl 12.377 | 9 | [↗](qwen/qwen3-4b.md) |
| qwen | Qwen3-235B-A22B | ppl 5.09 | 13 | [↗](qwen/qwen3-235b-a22b.md) |
| qwen | Qwen3-Next | ppl 5.4483 | 12 | [↗](qwen/qwen3-next-80b-a3b-instruct.md) |
| qwen | Qwen3.5-35b | ppl 6.2486 | 9 | [↗](qwen/qwen3.5-35b-a3b.md) |
| qwen | Qwen3.6-35b | ppl 6.2799 | 9 | [↗](qwen/qwen3.6-35b.md) |
| deepseek | Deepseek-v3.2 | ppl 3.2133 | 3 | [↗](deepseek/deepseek-v3.2.md) |
| glm | GLM-5.1 | ppl 2.5658 | 2 | [↗](glm/glm-5.1.md) |
| longcat | longcat-flash-lite | ppl 14.6133 | 9 | [↗](longcat/longcat-flash-lite.md) |
| longcat | longcat-Next | ppl 17.518 | 9 | [↗](longcat/longcat-next.md) |

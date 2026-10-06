# Qwen3.5-35B-A3B / Qwen3.6-35B / Qwen3.6-27B · 系列 qwen · 结构 moe（混合 attention）/ dense

> **速览**：`Qwen3_5Moe` 混合 `linear`+`full` attention、checkpoint MoE 已 packed（无需重组）。`Qwen3_6Moe(Qwen3_5Moe): pass` —— **3.6 与 3.5 结构完全相同（零 override）**，仅注册名/权重不同。Qwen3.6-27B Dense 复用 `qwen3_5` adapter，64 层的 `attn-linear` / `mlp` A4W4 MXFP4 已完成 PPL、Deploy 与 clean reload 验证。35B MoE 首推 `attn-linear + moe` W8A8-int 直转（达标）。核心坑：误用 DeepSeek MLA 接口（L3）。
> **触发信号**：35B MoE 的 `layer_types` 含 `linear_attention`+`full_attention`，checkpoint experts 已 packed；27B Dense 的 `model_type=qwen3_5`、`architectures=Qwen3_5ForConditionalGeneration`。读 [L2 · 混合 attention 类 + MoE 类](../structure-family-pitfalls.md) + [L1](../cross-model-pitfalls.md)。

## 结构与适配要点

- 40 层 decoder，`layer_types` 重复 `linear×3 + full`；checkpoint MoE 已 packed（不像 235B 需重组）。
- **参考与差异**：3.5 参考 `qwen3_5` attention/blockwise + `qwen3_moe` MoE 路由；3.6 = `Qwen3_5Moe` 纯子类（`pass`），暂无显著结构差异。
- **复用**：`BaseModel` blockwise、`PtqUnit`、`QuantLinear`、`apply_quant_attn`、`apply_quant_moe_mlp`。Qwen3.6-27B 的 attention 复用 Qwen3.5 实现，MLP 复用标准 Dense MLP 实现。**新增**：`qwen3_5_moe` 的 `build_quant_block()` 路由、`QuantPackedExperts`、`PackedExpertView`。
- **起步复用清单**（下一条 Qwen3.5/3.6 MoE）：从 `qwen3_5_moe.py` 起步。

## 适配验证结论

### Qwen3.5-35B-A3B
- 三步闭环通过；关闭量化 `attn`(3/19/39) 接近、`moe`(0/20/39) 对齐；`layer0/shared_expert` 最小 PTQ smoke 打通。

### Qwen3.6-35B（= Qwen3_5Moe，`pass`，零 delta）
- 三步闭环**复测通过**（real weights, `/mnt/data/models/Qwen/Qwen3.6-35B-A3B`，系统 python）：
  - BF16 baseline：**6.2731**（与精度表 6.2799 同量级，口径/版本微差）。
  - 关闭量化等价：`attn-linear+moe` @ `bf16.yaml`（空配置=全 16-bit）= **6.2751** ≈ BF16 → rebuilt wrapper 前向语义正确。
  - 最小 PTQ smoke：`extract_ptq_data` 16 样本 → `ptq --algos autoround` → 落盘 `layer_0_linear_attn.pt`（`attn-linear` 的 `linear_attn` 投影单元）✓。
- **运行注记**（env，非模型缺陷）：`linear_attention` 无 `flash-linear-attention` 包时走 torch fallback（功能正常）；autoround 触 `No module named 'triton'`（缺包，非致命，PTQ 参数已落盘）。

### Qwen3.6-27B Dense

- PTQ calibration：PileVal，`nsamples=16`，`seq_len=1024`；PPL evaluation：WikiText2 test，`seq_len=4096`。
- `attn-linear` 与 `mlp` 的 Direct、LWC、LAC、LWC+LAC、OmniQuant、AutoRound、FlatQuant 均完成 Deploy 与 clean reload；无缺失键、非预期键、形状不匹配或加载错误。
- clean reload 识别 `attn-linear` 304 个量化模块（第 0 层 5 个）、`mlp` 192 个量化模块（第 0 层 3 个）；Vision 分支和 `lm_head` 未被错误量化。

## 关键陷阱（L3 模型/家族专属；通用见 L1/L2）

- **误用 DeepSeek 的 MLA 接口**（L3，Qwen3_5/3_6 专属） —— 现象：进 quant block 构建即调到不存在的 `apply_quant_mla()` 报错。根因：`qwen3_5moe` 继承 `Qwen3_5`，应复用 Qwen 公共接口而非 DeepSeek MLA 专用入口。处理：统一收回 `apply_quant_attn()`/`apply_quant_moe_mlp()`。**教训：Qwen 分支先看 dense adapter 公共接口，勿直接用 DeepSeek MLA 入口。**
- 混合 attention causal mask 路由 / `linear` vs `full` 分开处理（`linear_attention` kernel 未就绪先留 BF16，仍可量化其投影 Linear）→ 见 [L2 · 混合 attention 类](../structure-family-pitfalls.md)。

## 量化结论

- 首推 `attn-linear + moe` W8A8-int8 直转，3.5/3.6 均 delta 小达标（见精度速查表）。`linear_attention` kernel 量化未就绪时先留 BF16，仅量化其投影 Linear（3.6 smoke 已验该单元可 PTQ）。
- Qwen3.6-27B A4W4 MXFP4 `attn-linear` 和 `mlp` 结果见精度速查表。

## 适配建议（下次同系列/同结构）

- 先参考：本案 + `qwen3-next`（混合 attention）+ Qwen README。
- 先做什么：先确认 experts packed/展开；BF16 离谱先查混合 attention mask 路由。
- **不建议**：Qwen 分支误用 DeepSeek MLA 入口；`linear`/`full` attention 一把梭。

## 精度速查表

> ppl 口径 seq_len=4096。MXFP 双值为两次评测口径 a/b。

### Qwen3.5-35B-A3B（BF16 6.2486）

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 6.2486 |
| INT | A4w4: moe, attn-linear | 无 | 3839.2734 |
| INT | A8w4: moe, attn-linear | 无 | 7.1928 |
| INT | A8w8: moe, attn-linear | 无 | 6.3781 |
| MXFP | A4w4: moe, attn-linear | 无 | 6.9814/6.826 |
| MXFP | A8w4: moe, attn-linear | 无 | 6.508/6.436 |
| MXFP | A8w8: moe, attn-linear | 无 | 6.3168/6.285 |

### Qwen3.6-35B（BF16 6.2799 / 复测 6.2731）

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 6.2799 |
| INT | A4w4: moe, attn-linear | 无 | 3962.8643 |
| INT | A8w4: moe, attn-linear | 无 | 7.3586 |
| INT | A8w8: moe, attn-linear | 无 | 6.3788 |
| MXFP | A4w4: moe, attn-linear | 无 | 7.024/6.931 |
| MXFP | A8w4: moe, attn-linear | 无 | 6.599/6.536 |
| MXFP | A8w8: moe, attn-linear | 无 | 6.3168/6.308 |

### Qwen3.6-27B（BF16 7.8214）

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 7.8214 |
| MXFP | A4w4: attn-linear | 无 | 8.0828 |
| MXFP | A4w4: attn-linear | lwc | 8.0008 |
| MXFP | A4w4: attn-linear | lac | 7.6261 |
| MXFP | A4w4: attn-linear | lwc lac | 7.5939 |
| MXFP | A4w4: attn-linear | omniquant | 7.8375 |
| MXFP | A4w4: attn-linear | autoround | 8.1834 |
| MXFP | A4w4: attn-linear | flatquant | 7.2257 |
| MXFP | A4w4: mlp | 无 | 7.9196 |
| MXFP | A4w4: mlp | lwc | 8.0670 |
| MXFP | A4w4: mlp | lac | 7.8648 |
| MXFP | A4w4: mlp | lwc lac | 8.0103 |
| MXFP | A4w4: mlp | omniquant | 7.8741 |
| MXFP | A4w4: mlp | autoround | 7.9661 |
| MXFP | A4w4: mlp | flatquant | 7.7327 |

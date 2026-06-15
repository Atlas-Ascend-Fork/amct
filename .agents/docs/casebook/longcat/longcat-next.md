# LongCat-Next · 系列 longcat · 结构 moe

> **速览**：第二个 LongCat 适配，拓扑近 Flash-Lite。核心坑：`trust_remote_code` 顶层走多模态入口，需切 text-only backbone。

## 结构与适配要点

- **参考与差异**：参考 `longcat_lite`（block 拓扑近同）；差异：顶层 `trust_remote_code` 走多模态、依赖 `flash_attn`，需切 text-only backbone。
- 结构同 Flash-Lite：`ngram_embeddings` 输入、双 self-attn + 双 dense MLP + MoE shortcut；`PtqUnit` 拆 `attn→self_attn_0/1`、`mlp→mlp_0/1`、`moe→expert_<i>`；checkpoint 逐 expert，运行时重组 packed。
- **复用**：`LongcatLite` blockwise 工作流与 PTQ unit 路由 + `QuantLinear`/`QuantGatedMLP`。**新增（唯一兼容点）**：adapter 内实例化 `modeling_longcat_ngram.LongcatFlashNgramForCausalLM`，`lm_head` 对齐 `text_vocab_plus_multimodal_special_token_size`。

## 适配验证结论

- 三步闭环通过；关闭量化 attn/mlp/moe 代表层 `max_abs=0` 完全对齐；`layer0/mlp_0` 最小 PTQ smoke（flatquant 1-epoch，回载 `max_abs=0`）打通。

## 关键陷阱（模型/框架内在，可复用）

- **`trust_remote_code` 顶层模型空载失败** —— 现象：`AutoModelForCausalLM.from_config(..., trust_remote_code=True)` 被 `flash_attn` 依赖阻塞。根因：顶层远端代码默认走多模态入口，而 blockwise PTQ 只需 text-only backbone。处理：adapter 内切到 `modeling_longcat_ngram.LongcatFlashNgramForCausalLM` 并修 `lm_head` 形状。**教训：`trust_remote_code` 模型先看 blockwise PTQ 真正需要的最小 backbone，再决定是否完整加载顶层模型。**

## 适配建议（下次同系列/同结构）

- 先参考：`longcat_lite`。
- 先做什么：先看能否直接复用 `longcat_lite` 的 block 拓扑；兼容收在 adapter 内。
- **不建议**：直接空载顶层多模态模型。

## 精度速查表

> ppl 口径 seq_len=4096（2048→4096 后精度反而变差）。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 17.518 |
| INT | A4w4: mlp, moe, attn-linear | 无 | 3302.094 |
| INT | A8w4: mlp, moe, attn-linear | 无 | 24.8581 |
| INT | A8w8: mlp, moe, attn-linear | 无 | 15.718 |
| MXFP | A4w4: mlp, moe, attn-linear | 无 | 19.0125/18.266 |
| MXFP | A8w4: mlp, moe, attn-linear | 无 | 17.5067/16.899 |
| MXFP | A8w8: mlp, moe, attn-linear | 无 | 18.3902/19.2 |
| HiFP | A4w4: mlp, moe, attn-linear | 无 | 16.9826 |
| HiFP | A8w4: mlp, moe, attn-linear | 无 |  |
| HiFP | A8w8: mlp, moe, attn-linear | 无 |  |

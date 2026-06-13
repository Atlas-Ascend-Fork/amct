# LongCat-Next

第二个 LongCat 系列适配案例。adapter：`amct_pytorch/common/models/llm/longcat/longcat_next/longcat_next.py`（+ `llm/__init__.py`）。

## 适配重点

- 结构同 `LongCat-Flash-Lite`：`ngram_embeddings` 输入、双 self-attn + 双 dense MLP + MoE shortcut；`PtqUnit` 拆 `attn→self_attn_0/1`、`mlp→mlp_0/1`、`moe→expert_<idx>`；checkpoint 逐 expert，运行时重组 packed。
- 复用 `LongcatLite` 的 blockwise 工作流与 PTQ unit 路由 + `QuantLinear`/`QuantGatedMLP`；唯一新增兼容点：adapter 内直接实例化 `modeling_longcat_ngram.LongcatFlashNgramForCausalLM`，并把 `lm_head` 对齐到 `text_vocab_plus_multimodal_special_token_size`。

## 关键陷阱（可复用）

- **`trust_remote_code` 顶层模型空载失败**：`AutoModelForCausalLM.from_config(..., trust_remote_code=True)` 被 `flash_attn` 依赖阻塞，因顶层远端代码默认走多模态入口，而 blockwise PTQ 只需 text-only backbone。修复：adapter 内切到 `modeling_longcat_ngram.LongcatFlashNgramForCausalLM` 并修 `lm_head` 形状。**教训：`trust_remote_code` 模型先看 blockwise PTQ 真正需要的最小 backbone，再决定是否完整加载顶层模型。**

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

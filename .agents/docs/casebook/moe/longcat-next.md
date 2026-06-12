# LongCat-Next

## 为什么这个案例重要

- 这是仓库里第二个 LongCat 系列适配案例。
- 这个模型的 text block 拓扑和 `LongCat-Flash-Lite` 接近，但顶层 `trust_remote_code` 模型额外依赖 `flash_attn`，不能直接按默认路径空载。
- 这个案例验证了：在不改 workflow、solver 和通用 quant module 的前提下，只在 adapter 内切到 text-only backbone，也能把 BF16、step8 和最小 PTQ smoke 全部打通。

## 参考关系

- 所属系列：LongCat
- 复用的仓内抽象：`BaseModel`、`PtqUnit`、`QuantLinear`、`QuantGatedMLP`
- 最接近的参考思路：`longcat_lite.py`
- 新增的最小模型专属兼容：`longcat_next.py`

## 结构判断

- 输入路径：`ngram_embeddings`
- decoder block：双 `self_attn` 分支、双 dense `mlps` 分支，以及一条 `mlp` MoE shortcut 路径
- MoE 形态：checkpoint 侧是逐 expert 权重，运行时仍需在 block 内重组成 packed expert 张量
- `PtqUnit` 拆分：
  - `attn` -> `self_attn_0`、`self_attn_1`
  - `mlp` -> `mlp_0`、`mlp_1`
  - `moe` -> `expert_<idx>`

## 适配结论

- 实际改动范围没有越界，只改了：
  - `amct_pytorch/common/models/llm/longcat/longcat_next/longcat_next.py`
  - `amct_pytorch/common/models/llm/__init__.py`
- 复用的已有抽象：
  - `LongcatLite` 的 blockwise 工作流和 PTQ unit 路由
  - `BaseModel` 的参数导出/回载链路
  - `QuantLinear` 和 `QuantGatedMLP`
- 新增的模型专属兼容点只有一个：
  - 在 adapter 内直接实例化 `modeling_longcat_ngram.LongcatFlashNgramForCausalLM`
  - 并把 `lm_head` 形状对齐到 `text_vocab_plus_multimodal_special_token_size`

## BF16 与 step8 验证

- BF16 baseline：
  - Wikitext PPL，blockwise，`seq_len=2048`
  - `ppl_bf16 = 13.807648658752441`
- 关闭量化后的 wrapper 等价：
  - `attn` 代表层 `0 / 7 / 13`：`max_abs = 0`
  - `mlp` 代表层 `0 / 7 / 13`：`max_abs = 0`
  - `moe` 代表层 `0 / 7 / 13`：`max_abs = 0`
- 挂上 quant wrapper 但关闭量化后的整链验证：
  - `wrapper_disabled_ppl = 13.807648658752441`

## 最小 PTQ smoke

- 目标路径：`quant_target = mlp`
- smoke 目标 unit：`layer 0 / mlp_0`
- 实际闭环内容：
  - 已提取 `mlp_0 / mlp_1` 的 unit 输入
  - 已跑 `flatquant`，`epochs = 1`
  - 已导出参数：`layer_0_mlp_0.pt`、`layer_0_mlp_1.pt`
  - 已回载验证：`reload_max_abs = 0.0`
- 训练日志：
  - `mlp_0`：`avg_loss = 0.000004`
  - `mlp_1`：`avg_loss = 0.000005`

## 典型问题

- 问题：
  - 用 `AutoModelForCausalLM.from_config(..., trust_remote_code=True)` 直接空载失败
- 现象：
  - 动态模块加载时被 `flash_attn` 依赖阻塞
- 根因：
  - 顶层 `LongCat-Next` 远端代码默认走多模态模型入口，而当前 blockwise PTQ 实际只需要 text-only backbone
- 修复：
  - adapter 内切到 `modeling_longcat_ngram.LongcatFlashNgramForCausalLM`
  - 并在同一处修正 `lm_head` 的 checkpoint 形状兼容
- 后续建议：
  - 遇到这类 `trust_remote_code` 模型，先看 blockwise PTQ 真正需要的最小 backbone，再决定是否必须完整加载顶层模型

## 最终建议

- 下一个 LongCat 系列模型，优先先看它能不能直接复用 `longcat_lite.py` 的 block 拓扑。
- 如果顶层模型只是在外层多包了一层依赖，但 text-only backbone 已经足够支持 PTQ 主链，优先把兼容收在 adapter 内。
- `LongCat-Next` 这一条目前已经完成 model-adapter skill 要求的稳定闭环；量化实验应单独走 quantization skill。

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

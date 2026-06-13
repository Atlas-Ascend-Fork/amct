# Qwen3-4B

## 结论

- 这是仓库里第一条稳定接通的 `qwen3` dense 适配案例。
- 当前模型路径：`<path-to>/Qwen3-4B`
- 当前 adapter：`amct_pytorch/common/models/llm/qwen/qwen3/qwen3.py`
- 已完成：
  - BF16 blockwise baseline
  - 关闭量化后的 step8 浮点等价
  - `block0/mlp` 最小 PTQ smoke
  - 第一轮直转量化评测
  - deploy 导出

## 结构判断

- 这是标准 dense Qwen3 decoder，不带 MoE。
- `lm_head` 不单独落盘，和 `embed_tokens` tied，需要在 adapter 里把 `lm_head` 绑定回 `model.embed_tokens.`
- attention 路径沿用官方 `Qwen3Attention`，但 quant wrapper 里会拆成 `attn-linear` 和 `attn-cache` 两类目标。

## 适配要点

- `qwen3.py` 只用于 dense Qwen3；`qwen3_moe.py` 保持独立 adapter，不共享目录语义。
- `Catcher` 需要兼容 `attention_type`，否则 Qwen3 官方 forward 在 embedding 阶段会因为换掉 `layer0` 后访问不到该字段而报错。
- `attn-linear` 的 wrapper 不能在只做 `q/k/v/o` 量化时把 attention kernel 换成自定义 matmul 路径；需要继续复用官方 `attention_interface`，否则 step8 会明显漂。

## 适配闭环结果

- BF16 baseline：
  - Wikitext PPL，`seq_len=2048`
  - `13.5441`
- step8：
  - `mlp` only：`13.5441`
  - `attn-linear` only：`13.5441`
  - 说明当前 `attn-linear + mlp` 的 rebuilt quant modules 在 `bits=16` 下与 BF16 对齐
- 最小 PTQ smoke：
  - 目标：`block0/mlp`
  - 提取：`extract_ptq_data` 已打通
  - 训练：`a8w8-int + lwc/lac + epochs=1`
  - 保存：`layer_0_mlp.pt`
  - 回载：`load_selected_layer_ptq_params(0, ...) -> {'loaded': ['mlp'], 'missing': []}`

## 第一轮量化结果

- 量化目标：`attn-linear + mlp`
- 数据类型：`int`
- 配置：`a8w8`
- 评测口径：Wikitext PPL，`seq_len=2048`
- 结果：
  - `ppl_bf16 = 13.5441`
  - `ppl_quant = 13.7177`
  - `delta = 0.1736`
- 结论：
  - 当前结果在量化 skill 的默认接受阈值 `delta <= 0.2` 内
  - 可以先接受这一轮直转方案，不需要继续升级到 PTQ

## Deploy 结果

- 导出目录：`<deploy artifact: A8W8 INT>`
- 当前导出产物：
  - `layer_000.safetensors` 到 `layer_035.safetensors`
  - `rest_00000.safetensors`
  - `model.safetensors.index.json`
- 当前导出是直转量化导出，没有额外 PTQ 参数目录参与。

## 复用建议

- 下一条 dense Qwen3 模型，优先复用：
  - `amct_pytorch/common/models/llm/qwen/qwen3/qwen3.py`
  - `amct_pytorch/common/models/llm/qwen/qwen3/quant_module.py`
- 第一轮建议顺序：
  1. BF16 baseline
  2. step8 先拆 `mlp-only` 和 `attn-only`
  3. 再合并跑 `attn-linear + mlp`
  4. 如果 `delta <= 0.2`，优先直接 deploy

## 常见坑

- `lm_head` tied 到 `embed_tokens`，不要按独立 `lm_head.weight` 去找 index。
- 如果 `attn-linear` 的 step8 爆掉，优先检查 wrapper 是否把官方 attention interface 提前替换成了自定义 attention kernel。
- `quant_target=moe` 对 dense `qwen3` 不应该允许进入。

## 精度速查表

> ppl 口径 seq_len=4096。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 12.377 |
| INT | A4w4: mlp, attn-linear | 无 | 9942.9922 |
| INT | A8w4: mlp, attn-linear | 无 | 19.9143 |
| INT | A8w8: mlp, attn-linear | 无 | 12.4327 |
| MXFP | A4w4: mlp, attn-linear | 无 | 16.043/14.1 |
| MXFP | A8w4: mlp, attn-linear | 无 | 13.808/13.066 |
| MXFP | A8w8: mlp, attn-linear | 无 | 12.6976/12.365 |
| HiFP | A4w4: mlp, attn-linear | 无 |  |
| HiFP | A8w4: mlp, attn-linear | 无 |  |
| HiFP | A8w8: mlp, attn-linear | 无 |  |

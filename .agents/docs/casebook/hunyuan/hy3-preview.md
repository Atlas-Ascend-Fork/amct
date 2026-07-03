# Hy3-preview（Tencent Hunyuan V3）量化适配个案

> **触发信号**：`config.json` 中 `model_type=hy_v3` / `architectures=["HYV3ForCausalLM"]`；`num_experts=192` + `num_experts_per_tok=8` + `num_shared_experts=1` + `moe_router_use_sigmoid=true` + `moe_router_enable_expert_bias=true`；`first_k_dense_replace=1`；`qk_norm=true`；`num_nextn_predict_layers=1`（MTP）。读 L2 `../structure-family-pitfalls.md`（MoE 类 + 自定义 modeling 类）+ L1 `../cross-model-pitfalls.md`。

## 结构与适配要点

- **架构**：80 层 decoder；`first_k_dense_replace=1`（layer0 = dense GatedMLP，intermediate 13312；layer1-79 = MoE，moe_intermediate 1536）；192 routed experts top-8 + 1 shared expert；sigmoid 路由 + expert_bias + route_norm + router_scaling 2.826；GQA 64/8、head_dim 128、qk_norm；hidden 4096、vocab 120832；1 层 MTP；`enable_lm_head_fp32`。packed experts（`gate_up_proj` [192,3072,4096] + `down_proj` [192,4096,1536]）。
- **参考与差异**：最近参考 `deepseek_v3_2`（MoE + shared expert + sigmoid 路由）与 `qwen3_moe`（packed experts）；关键差异 = sigmoid+expert_bias 路由组合、router_scaling、qk_norm、MTP。
- **复用**：`BaseModel` blockwise 加载、`QuantGatedMLP`（dense + 每 expert）、`QuantGatedExperts`/`GatedExpertView`（moe packed 解包 + PTQ 实体化）、`scaled_dot_product_attention`、`apply_quant_to_attn`。**新增**：`hyv3/`（HyV3 adapter + QuantHYV3Attn/MLP/MoE wrapper + weight packing/key mapping）。
- **起步复用清单**（下一条同结构）：从 `qwen3_moe` 的 `QuantGatedExperts` 起步接 MoE（自带 materialize PTQ 实体化路径），attn wrapper 抄 qwen3 的 sdpa 调用（`is_causal=True, attn_mask=None`）；首轮起步方案 W8A8-int / moe+attn-linear / 直转。

## 适配验证结论

- 标准三步闭环（BF16 baseline → 关闭量化浮点等价 → 最小 PTQ smoke）：BF16 baseline + PTQ smoke（lwc，前 2-3 block）通过。
- **模型特定结论**：routed/shared/attn-linear 量化通路全部识别正确；shared_experts 走直转（不 yield PTQ，见下）；直转 W8A8 与 mxfp8 PPL 均近无损（见量化结论）。

## 关键陷阱（L3 模型专属；通用见 L1/L2）

- **`eval_mode=bf16` 需带 `--bit_config`（全 ≥16bit 的 yaml，如 configs/bf16.yaml 空配置=全16）** —— 现象：不带或带 w8a8 报 `eval_mode=bf16 requires a bit_config with no <16-bit entries`。根因：bf16 校验强制非量化策略。处理：用 `configs/bf16.yaml`（空配=全 16bit）。教训：BF16 baseline 也必须显式给 bit_config。
- **shared_experts PTQ 策略矛盾（L3，已修复）** —— 现象：PTQ 报 `PTQ input file not found: block_*_moe.shared_in.pkl`；训练 loss 恒为 0；虽有参数保存但无实际优化。根因：quant_module 用 `build_no_algo_args(args)`（清空 algos），但 adapter `iter_ptq_units` 错误 yield shared_experts → 两层设计矛盾。处理：注释掉 adapter 的 yield（hyv3.py:133-136），保持与 quant_module 一致；shared_experts 使用直接量化（密集激活，效果已够好）。教训：MoE adapter 必检查 `build_no_algo_args` 与 `iter_ptq_units` 的一致性，优先用通用函数 `apply_quant_to_moe_mlp`（自动处理）；**训练 loss 恒为 0 时立即停止检查适配逻辑**。见 L2 `../structure-family-pitfalls.md` · MoE 类 · shared_experts PTQ 策略不一致。
- packed-expert device 迁移 / attn sdpa is_causal / "冷 expert 无输入"误诊更正（实为 shared_experts 误 yield）→ 见 L2 `../structure-family-pitfalls.md` · MoE 类（本案为这几条的复现例）。

## 量化结论

- BF16 baseline（seq_len=4096）= **3.9036**；首轮直转 **moe+attn-linear**，全 80 层，granularity block，无 PTQ：
  - **int W8A8**：ppl 3.9203，`delta=+0.0167`（达标，略优）
  - **mxfp8**：ppl 3.9455，`delta=+0.0419`（达标）
  - **mxfp W4A8**（routed W4A8 / shared+attn W8A8）：ppl 4.1163，`delta=+0.2127`（routed 专家降至 4bit 精度损耗明显，可接受性视业务而定）
- 性能注意：MoE per-expert 路径在小 M_eff 下收益存疑，须 infer 侧 packed `MoEGMM` 实测确认（见 L2 MoE 类）。mxfp8 为 amct 伪量化 eval（A3 可行）；真 MX GEMM 推理需 A5（ascend910_95）。

## 适配建议（下次同系列/同结构）

- 先参考：本案 + `qwen3_moe` / `deepseek_v3_2`。
- 先做什么：抓 `config.json`（确认 first_k_dense_replace / num_experts / sigmoid 路由 / qk_norm / MTP）；MoE wrapper 优先用 `apply_quant_to_moe_mlp`（自动处理 shared_experts）；若自写 wrapper，必须检查 `build_no_algo_args` 与 `iter_ptq_units` 一致性；attn wrapper 用 `is_causal=True, attn_mask=None`；gate/router/expert_bias/lm_head 不量化。
- **不建议**：自写 experts 量化类只做 `materialize=False` 惰性视图（PTQ 时 weight 留 CPU → device 不一致）；attn sdpa 同时传 `attn_mask` 和 `is_causal=True`（撞 assert）；靠 architectures 命名推断而不读 config；quant_module 对 shared_experts 用 `build_no_algo_args` 但 adapter 仍 yield shared_experts 作为 PTQ unit。

## 精度速查表

> ppl 口径 seq_len=4096。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 3.9036 |
| int W8A8 | moe+attn-linear | 直转 | 3.9203 |
| mxfp8 | moe+attn-linear | 直转 | 3.9455 |
| mxfp W4A8 | moe+attn-linear（routed W4A8 / shared+attn W8A8） | 直转 | 4.1163 |

# Qwen3.6 Ascend Tensorwise Deploy Design

## Goal

在 `amct_qwen_deploy` 分支中，为 Qwen3.6 MoE 增加基于 JSON 模糊匹配的
Ascend tensorwise W8A8_DYNAMIC 导出，生成 vLLM-Ascend 0.23.0 可消费的
`quant_model_description.json`，并保留现有 `legacy` 路径。

## Confirmed Scheme

- 输入是 Qwen3.6 BF16 checkpoint，配置前缀为 `model.language_model.layers.*`。
- 40 个语言层的 256 个 routed experts 全部量化为 W8A8_DYNAMIC。
- full attention 层（3、7、11、...、39）只量化 `self_attn.q_proj`、`k_proj`、`v_proj`。
- linear attention、shared expert、router、`self_attn.o_proj`、embedding、norm、lm_head 保持 FLOAT。
- source index 的 `mlp.experts.gate_up_proj` 和 `down_proj` 是 packed tensor；导出时展开为
  `mlp.experts.<expert>.gate_proj/up_proj/down_proj.weight`。

## Architecture

复用现有 `Qwen3_6Moe` 注册适配器和共享 Ascend payload/description/validation 流程。
适配器提供虚拟 module inventory、packed expert 展开和角色校验；workflow 只负责在导出前调用
这些模型钩子，不承载 Qwen 专属命名逻辑。JSON 选择文件使用展开后的 vLLM module 名称。

## Compatibility

`--deploy_format legacy` 及其原有代码路径保持不变；Ascend 仅允许
`--model_name qwen3_6_moe`，`--granularity tensor`、`--quant_dtype int` 和 W8A8 默认 bit policy。
目标格式遵循 vLLM-Ascend 0.23.0 的 `weight` INT8、`weight_scale`/`weight_offset` BF16 约定。

## Validation

单元测试覆盖 packed expert 展开、选择 JSON 命中、Qwen3.6 角色策略和 Ascend 参数门禁；
使用真实 BF16/目标 description 的 index 做静态命中统计。实际 NPU 服务启动仍需在用户机器验证。

# Pangu-7B

## 为什么这个案例重要

- 这是仓库里第一条 Pangu 系列适配案例。
- 模型结构本身是标准 dense decoder，但模型实现完全来自 `trust_remote_code` 的自定义文件。
- 真正的难点不是 block 结构，而是模型导出时代码依赖的 `transformers` 接口，比远端运行环境更新。

## 参考关系

- 所属系列：Pangu
- 复用的仓内抽象：`BaseModel`、`PtqUnit`、`QuantLinear`、`QuantGatedMLP`
- 最接近的参考思路：Qwen dense 的 blockwise 路径，以及通用 `QuantGatedMLP` / `apply_quant_to_attn` / `apply_quant_to_moe_mlp`

## 适配结论

- 结构判断：
  - 34 层 dense decoder
  - 标准 `self_attn + mlp`
  - attention 是 GQA，不是 MoE
- 实际改动范围：
  - 新增 `amct_pytorch/common/models/llm/pangu/pangu_7b/pangu_7b.py`
  - 更新 `amct_pytorch/common/models/llm/__init__.py`
  - 没有改 workflow、solver 和 quant module 主链
- 复用了这些抽象：
  - `BaseModel` 的 blockwise 工作流
  - `PtqUnit` 对 `attn` 和 `mlp` 的路由
  - `QuantLinear`
  - `QuantGatedMLP`
  - `apply_quant_to_attn`
  - `apply_quant_to_moe_mlp`
- 新增的模型专属实现：
  - `QuantPanguAttn`
  - `QuantPanguMLP`
  - adapter 内的 `transformers` 兼容补丁
  - adapter 内关闭模型自带 NPU fused attention 的逻辑
- 最终验证状态：
  - BF16 baseline：通过
  - 关闭量化后的浮点等价：通过
  - 最小 PTQ 集成 smoke：通过，使用 `layer 0 / mlp`

## 精度与验证

- BF16 baseline：
  - Wikitext PPL，blockwise，`seq_len=2048`
  - `ppl_bf16 = 13.9502`
- 代表层关闭量化验证：
  - `attn-only` 在 `0 / 17 / 33` 层上的测试样本精确对齐
  - `mlp-only` 在 `0 / 17 / 33` 层上的测试样本精确对齐
- 挂上 Quant wrapper 但关闭量化后的整链验证：
  - 全量 Wikitext PPL 为 `13.9502`
- 最小 PTQ smoke：
  - `quant_target = mlp`
  - `layer 0 / mlp`
  - 输入提取、GT 构造、1 epoch `flatquant`、参数导出和参数回载全部通过
  - 参数文件：`layer_0_mlp.pt`

## 第一轮量化结论

- 量化方案：
  - blockwise Wikitext PPL
  - `seq_len=2048`
  - `a8w8-int8`
  - 第一轮整网直转量化目标为 `attn + mlp`
- 结果：
  - `ppl_bf16 = 13.9502`
  - `ppl_quant(attn+mlp) = 13.5534`
  - `delta = -0.3968`
- 路径级粗拆分：
  - `attn-only = 13.1908`
  - `mlp-only = 14.4075`
- 判读：
  - 当前整网 `attn + mlp` 方案达标
  - 这轮量化没有进入 PTQ
  - 从粗拆分结果看，当前收益主要来自 attention 路径，`mlp-only` 单独量化反而会掉点
- 对后续实验的建议：
  - 这一模型后续优先沿 `attn` 或 `attn + mlp` 路线继续
  - 不建议把 `mlp-only` 当成默认首选起点

## 典型问题

- 问题：
  - `trust_remote_code` 模型文件依赖的 `transformers` 接口高于远端环境版本
- 现象：
  - 初始化模型时报缺少 `LossKwargs`、`auto_docstring`、`can_return_tuple` 或 `ROPE_INIT_FUNCTIONS['default']`
- 根因：
  - 模型代码导出时依赖了比当前远端 `transformers` 更高版本的接口
- 修复：
  - 在 adapter 内补最小兼容层，再继续用现有框架主链
- 后续建议：
  - 自定义模型先看 `config.py` / `modeling.py` 的 imports，再决定是升级环境还是在 adapter 内做兼容补丁

- 问题：
  - 模型自带 NPU fused attention 路径在当前环境下不稳定
- 现象：
  - 原生整网参考路径会出现 `nan` 或明显异常的 loss / PPL
- 根因：
  - 模型代码里的 fused attention 分支与当前远端环境不完全兼容
- 修复：
  - 在 adapter 内显式关闭模型自带 fused attention，并固定使用 eager attention
- 后续建议：
  - 只要 BF16 baseline 明显异常，先检查模型自己的 fused attention 开关，而不是直接怀疑 quant wrapper

- 问题：
  - 这个模型的最小 PTQ smoke 更适合先走 `mlp`
- 现象：
  - Dense 模型里 `mlp` 更容易直接接上现有 `QuantGatedMLP + flatquant` 路径
- 当前处理：
  - `layer 0 / mlp` 完成了最小 PTQ 集成 smoke
- 后续建议：
  - 最小 PTQ 集成 smoke 仍然适合从 `mlp` 起步
  - 但第一轮直转量化的默认起点更适合优先看 `attn` 或 `attn + mlp`

## 最终建议

- 下一个 Pangu 系列模型，优先从这个案例和 Pangu 系列 README 开始。
- 先确认自定义模型代码与远端 `transformers` 的兼容关系，再做 BF16 baseline。
- 如果模型自带 fused attention 路径不稳定，优先在 adapter 层固定回 eager 路径。
- 对这一系列，最小 PTQ smoke 可以先做 `mlp`，但第一轮直转量化更适合先做 `attn`。

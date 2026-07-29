# Attention 适配规则

## 使用时机

当任务涉及 attention wrapper、`attn-linear`、`attn-cache`、mask、position、RoPE、SDPA、MLA、混合 attention 或 cache/generate 分支时读取本文件。

**新增或结构性修改 attention wrapper 转 `model-adapter`**（见 [delegation.md](delegation.md)）。amct-coder 用本文件做三件事：review `model-adapter` 的产出、委派时交待必查项、修不改结构的局部 bug。下面的执行步骤同样是 review 时的核对顺序。

## 事实来源

适配原则和已知模型个案不在本文件维护：

- 原则：`.agents/docs/repo-map.md` §Attention 适配原则。
- 个案：`.agents/docs/casebook/structure-family-pitfalls.md` 与命中模型族的 L3 个案。

本文件只写编码动作、必查项和禁止模式。原则与本文件冲突时以 repo-map 为准。

## 输入输出契约

- 上游输入：原模型 attention forward、config、mask/position/RoPE 生成方式、adapter 捕获到的 blockwise kwargs。
- 本层输出：可被 `build_quant_block()` 替换的 attention wrapper，以及关闭量化后与原始 block 足够接近的 forward 行为。
- 下游消费：PPL eval、extract/PTQ 的 `PtqUnit`、deploy binding 和可能的 `attn-linear` / `attn-cache` target。
- 不负责：算法选择、cache 量化收益判断、NPU kernel 实现。

## 执行步骤

1. 先用真实 blockwise forward 确认哪些参数实际参与计算，不按源码形参表推断。
2. 按顺序复用：同族 attention wrapper -> `common/models/llm/common/attention_forward.py` helper -> 新写。
3. 只替换量化目标内的投影层，保留官方 attention 主语义。
4. 显式处理 mask、position、RoPE 和混合 attention 差异，删除当前路径不用的 cache 分支。
5. 关闭量化验证浮点等价，再进入 extract、PTQ 或 deploy。

## 必查清单

- 当前任务是否真的用到 `past_key_values`、cache 或 decode-only 分支；不用就不要保留。
- `forward` 形参在 blockwise / PPL / PTQ 路径下是否恒为 `None` 或固定值；若不参与计算，不要继续向下传。
- 是否必须传 `attention_mask`；不要机械照搬源码参数表。
- 是否需要 `position_ids` 或 `position_embeddings`；以真实 blockwise forward 为准。
- 同系列 dense / MoE attention 初始化和 forward 是否一致；一致则复用一份实现。
- `attn-linear` 是否只 wrap q/k/v/o 投影 Linear；不得替换官方 attention kernel。
- `scaled_dot_product_attention` 是否同时传了 `attn_mask` 和 `is_causal=True`；二者在当前 helper 中可能互斥。因果 LM 无 padding 时优先 `attn_mask=None, is_causal=True`；显式传 causal mask 时设 `is_causal=False`。
- 混合 attention 是否区分 `linear_attention` 与 `full_attention` 的 mask；不能只复用 layer0 捕获到的 mask。
- MLA 是否区分 `attn-linear` 与 `attn-cache` 两条量化路径；编码任务只实现必要路径，不替用户推断最终量化方案。
- 逐个 `forward` 形参和每段 cache/generate 代码过一遍：在当前 blockwise / PPL / PTQ / deploy 路径下是否真的参与计算。判断依据是 debug 时看到的真实运行时传参，不是源码形参表。不参与的一律删，不要为了贴源码保留。`past_key_values.update(...)`、恒为 `None` 的 `attention_mask` 是最常见的两个。

## 禁止模式

- 为了“贴源码”保留当前流程不用的 cache 更新逻辑。
- 为了局部少改，把 dense 和 MoE 的完全重复 attention wrapper 长期拆成两份。
- 在 workflow 中给某个模型补 attention 计算。
- 在通用 SDPA helper 里写模型名分支。
- 只因源码形参存在，就把恒为 `None` 的参数继续向下传递。

## 验证

attention 改动最低验证：

- 原始 float block forward 正常。
- quant block forward 正常。
- 关闭 weight/activation quant 后输出与原始 block 足够接近；漂移时优先检查是否替换了官方 attention kernel。
- BF16 PPL 异常时第一优先级检查 mask/position 传递链。

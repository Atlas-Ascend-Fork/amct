# Model Adapter Workflow

## 目标

以**最小改动范围**把新模型接入现有 amct PTQ 主流程。

## Step 1：先判断结构

写代码前，先回答这几个问题：

1. block class 是什么？
2. attention class 是什么？
3. MLP class 是什么？
4. 模型是 dense 还是 MoE？
5. 如果是 MoE，experts 是独立模块还是 packed tensor？
6. 哪些子结构要变成 `PtqUnit`？

如果这些问题还答不清，就先不要开改。

## Step 2：先定写入范围

新模型默认只改：

- `amct_pytorch/common/models/llm/<family>/<model>/...`

只有在有明确理由时才改其他层：

- `amct_pytorch/common/models/llm/common/...`：现有共享抽象确实不够时
- `amct_pytorch/quantization/modules/...`：通用 wrapper 真的表达不了该模型路径时
- `amct_pytorch/workflows/...`：workflow 本身与该模型结构不兼容时

## Step 3：按顺序复用

优先顺序：

1. 复用 `PtqUnit`
2. 复用 `QuantLinear`
3. 复用 `QuantGatedMLP`
4. 复用 `apply_quant_to_attn()` / `apply_quant_to_moe_mlp()`
5. 只有真正不同的部分再写模型专属 wrapper

除非同一种模式已经在至少两个模型族里重复出现，否则不要急着抽新基类。

## Step 4：先打通最小路径

不要一口气适配整个模型。

推荐顺序：

1. 先做一个 block loader
2. 再做一个 quant block builder
3. 再做一个 PTQ unit enumerator
4. 再打通一条 wrapper 路径
5. 最后补保存 / 加载路径

复杂模型优先顺序：

- MLP first
- then attention
- then MoE / packed experts

## Step 5：先拿到 BF16 baseline

在正式进入量化前，先确保：

1. 模型的 BF16 推理路径能跑通
2. baseline 指标能被记录

如果当前因为环境或路径原因暂时拿不到 baseline，也要明确记录阻塞点，不能静默跳过。

## Step 6：先过浮点等价检查

在做 PTQ 前先验证：

1. 原始 float block forward 正常
2. quant block forward 正常
3. 关闭量化后，quant block 与原始 float block 足够接近

如果关闭量化后仍和原始 float block 不一致，先修 wrapper 等价性，不要继续往 PTQ 走。

## Step 7：再接最小 PTQ 闭环

float 等价过了之后，再做：

1. 抽一条 unit 输入路径
2. 为该 unit 生成 GT
3. 跑一次 unit 级 solver
4. 保存 PTQ 参数
5. 回载 PTQ 参数

做完这些，再考虑更大范围的评测。

## Step 8：最后再总结

结束前至少输出：

- 这次模型适配复用了什么
- 哪些路径已接通
- 哪些路径仍是风险
- 后续最优先该补什么

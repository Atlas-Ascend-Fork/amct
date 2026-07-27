# Wrapper 合并与复用规则

## 使用时机

当任务涉及 MLP、MoE-MLP、experts、shared experts、`QuantGatedMLP`、`QuantLinear`、模型专属 `quant_module.py` 或 wrapper 类新增/修改时读取本文件。

## 输入输出契约

- 上游输入：原模型 MLP/MoE 模块结构、checkpoint expert 布局、已有 family helper、quant module 能力。
- 本层输出：adapter 可替换的 wrapper、稳定的 state_dict 映射、可枚举的 `PtqUnit` 和 deploy binding。
- 下游消费：extract 的 unit 输入文件、solver 的 trainable params、deploy 的权重导出和 ignore/shared expert 策略。
- 不负责：dtype 数值语义、算法收益判断、workflow 中的模型特例。

## 执行步骤

1. 先比较同族 dense、MoE、shared expert wrapper 的初始化、状态和 forward 是否真的不同。
2. 按复用顺序选择已有抽象，只有结构或导出语义真实不同才新增 wrapper。
3. 对 packed experts，确认是否需要 materialized Parameter 才能进入 PTQ。
4. 对 shared experts，同时检查 wrapper 是否清空算法以及 adapter 是否 yield 对应 `PtqUnit`。
5. 确认 `unit.kind`、extract 文件名、solver 参数名和 deploy binding 一致。
6. 关闭量化验证 wrapper 等价，再验证 PTQ 参数导出/回载。

## 合并原则

- 如果同系列两个 wrapper 的初始化、状态和 `forward` 一样，优先合并为一份实现。
- 不要为了保留旧类名或减少 import 改动，长期保留行为完全重复的 wrapper。
- 只有模块结构、导出语义、PTQ unit 边界或量化路径真实不同，才拆成两份。
- “dense”和“moe”标签本身不构成拆分类理由；看实现是否真的不同。

## 复用顺序

1. `QuantLinear`
2. `QuantGatedMLP`
3. `apply_quant_to_attn()`
4. `apply_quant_to_moe_mlp()`
5. family 级 `moe_common.py`
6. 已有同族 `quant_module.py`
7. 新增模型专属 wrapper

## MoE wrapper 检查

- checkpoint 是否逐 expert 展开，运行时是否 packed。
- 是否已有 family 级 packed expert helper，例如 Qwen/LongCat 的 `moe_common.py`。
- 是否错误地用宽泛 hook 命中 gate/router，导致捕获 tuple。
- per-expert PTQ 是否需要 materialized Parameter；不要用只保留 CPU 惰性视图的实现进入 PTQ。
- shared experts 是否和 `build_no_algo_args(args)` 策略一致。

## shared experts 策略

`apply_quant_to_moe_mlp()` 对 `shared_experts` 默认使用 `build_no_algo_args(args)` 清空算法，表示直接量化但跳过 PTQ 算法。

因此 adapter 中必须保持一致：

- 如果 wrapper 用 `build_no_algo_args(args)`，不要在 `iter_ptq_units()` 中 yield `moe.shared`。
- 如果确实要 PTQ shared experts，wrapper 和 adapter 都不能跳过算法，并且必须证明有对应输入文件和训练收益。

训练 loss 恒为 0 或 NaN 是强诊断信号，必须立即停下检查适配逻辑，不要继续扩大实验。

## PTQ unit 命名

- `unit.kind` 必须和 extract 阶段产出的 `block_<layer>_<quant_target>_in.pkl` 命名一致。
- `inps is None` 表示该 `(layer, quant_target)` 输入文件不存在，不能用 skip 掩盖。
- 同层 experts 通常共用同一 target 输入文件，不存在按 expert 文件区分的“冷 expert 无输入”。

## 禁止模式

- 自写一个只适用于单模型的 experts 类，却放到通用 quant modules。
- 给同一行为保留 dense wrapper 和 moe wrapper 两份实现。
- 跳过 `QuantGatedMLP` 直接复制大段 MLP forward，除非结构确实不同。
- 在 adapter 中 yield 一个 extract 从不产出的 `unit.kind`。
- PTQ 参数能保存但训练没有真实优化，还继续声称 smoke 成功。

## 验证

wrapper 改动最低验证：

- 构建至少一层 quant block。
- 枚举至少一个 `PtqUnit`。
- 关闭量化时 wrapper 与原始 float 模块前向等价或足够接近。
- PTQ unit 至少能导出参数；推荐回载后 forward 一次。

# Repo Map

## 作用

这是一份**稳定架构说明**，不是全仓逐文件说明，也不是永远实时的真相。

先读它，再抽查下面的**锚点文件**。
如果锚点文件仍然支持这里的关键结论，就不要重扫全仓。

## 校验信息

| 项目 | 内容 |
| --- | --- |
| 最近校验日期 | `2026-06-16` |
| 使用方式 | `先读 repo-map -> 抽查锚点 -> 只更新受影响 section` |
| 适用场景 | `新模型适配与 PTQ 流程定位` |

## 锚点文件

开始新任务前，至少抽查这些文件：

- `README.md`
- `amct_pytorch/cli/llm/args.py`
- `amct_pytorch/workflows/llm_ptq.py`
- `amct_pytorch/common/models/llm/__init__.py`
- `amct_pytorch/common/models/llm/common/base.py`
- `amct_pytorch/common/models/llm/common/quant_apply.py`
- `amct_pytorch/quantization/modules/quant_base.py`
- `amct_pytorch/quantization/modules/quant_linear.py`
- `amct_pytorch/common/optimization/blockwise_solver.py`

## 分层边界

| 层 | 主要位置 | 职责 | 默认规则 |
| --- | --- | --- | --- |
| CLI | `amct_pytorch/cli/llm/` | 解析参数并启动 workflow | 不放模型逻辑 |
| Workflow | `amct_pytorch/workflows/` | 编排 eval、PTQ 数据提取、PTQ 训练 | 不放模型细节 |
| Model adapter | `amct_pytorch/common/models/llm/...` | 构建 block、quant block、拆 `PtqUnit`、加载 unit 输入和参数 | 新模型适配主入口 |
| Quant modules | `amct_pytorch/quantization/modules/` | 提供通用量化 wrapper，如 `ActivationQuantizer`、`WeightQuantizer`、`QuantLinear` | 保持通用 |
| Algorithms | `amct_pytorch/algorithms/` | 实现 activation / weight / structure 算法 | 通过 `targets` 注册 |
| Optimization | `amct_pytorch/common/optimization/` | 管 optimizer、epoch 循环、重建训练 | 不混入模型分支 |
| Data | `amct_pytorch/common/datasets/` | 提供 PTQ 输入、GT 生成、dataloader | 优先复用现有 provider |

## PTQ 主路径

### 1. Workflow

`amct_pytorch/workflows/llm_ptq.py` 是当前 PTQ 主流程的编排中心：

1. 注册 algorithms、dtype、models、solvers
2. 从 model registry 构建 pipeline
3. 遍历 layer
4. 为目标层构建 quant block
5. 将 block 拆成 `PtqUnit`
6. 准备离线输入和 GT
7. 调 `BlockwiseSolver`
8. 保存 unit PTQ 参数

## 当前接通状态

这一节只记录**当前已经接通的主链**，避免把目录存在但尚未真正纳入主流程的内容写得过满。

- 当前已注册的 LLM 模型适配器：
  - `deepseek_v3_2`
  - `deepseek_v4`
  - `longcat_lite`
  - `longcat_next`
  - `glm5`
  - `qwen3`
  - `qwen3_moe`
  - `qwen3_5`
  - `qwen3_5_moe`
  - `qwen3_6_moe`
  - `qwen3_next`
- 当前已注册的 solver：
  - `block`
- `amct_pytorch/common/optimization/global_solver.py` 文件存在，但当前没有注册到 solver 主链。
- `amct_pytorch/workflows/llm_deploy.py` 当前已接到主流程，支持按 block 写 `layer_xxx.safetensors`、补 `rest_xxxxx.safetensors` 并刷新 `model.safetensors.index.json`。

### 2. 核心模型抽象

`amct_pytorch/common/models/llm/common/base.py` 目前提供最关键的复用抽象：

- `PtqUnit`（定义在 `amct_pytorch/common/models/llm/common/ptq_units.py`，base.py import 并驱动）
- `BaseModel.build_quant_block()`
- `BaseModel.iter_ptq_units()`
- `BaseModel.ptq_param_handler`
- `BaseModel.ptq_param_store`
- `BaseModel.do_block_forward()` 已支持 per-sample state:除已有 `position_ids / position_embeddings / attention_mask`(整批共用)外,新增 `self.input_ids`(per-sample list,默认 None);adapter 在 `do_embedding_forward` 末尾设 `self.input_ids = list(samples)`,`do_block_forward` 的 forward loop 会在每次迭代把对应 entry 注入 kwargs(用于 v4 这类 MoE hash routing 需要 token-id 路由的场景)。其它 adapter 不设 `self.input_ids` 时行为完全不变
- `QuantGatedMLP`（实际定义在 `amct_pytorch/common/models/llm/common/quant_apply.py`）

### 3. 当前 quant wrapper 形态

`amct_pytorch/quantization/modules/` 提供当前通用的底层 wrapper：

- `ActivationQuantizer`
- `WeightQuantizer`
- `QuantLinear`

这些模块本来就是给 `amct_pytorch/common/models/llm/...` 里的模型专属 wrapper 复用的。

### 4. 当前算法分流

算法通过 `--algos` 选择，再根据 registry `targets` 分流：

- `activation`
- `weight`
- `structure`

分流逻辑目前在 `amct_pytorch/quantization/modules/quant_base.py`。

## 当前模型适配模式

`amct_pytorch/common/models/llm/qwen/qwen3/qwen3.py`、`amct_pytorch/common/models/llm/qwen/qwen3/qwen3_moe.py`、`amct_pytorch/common/models/llm/qwen/qwen3_5/qwen3_5.py` 和 `amct_pytorch/common/models/llm/qwen/qwen3_5/qwen3_5_moe.py` 体现了当前 Qwen 系列的适配模式：

1. 在 `MODEL_REGISTRY` 中注册模型适配类
2. 能按 `layer_idx` 加载单层 block
3. 实现 `build_quant_block(layer_idx)`
4. 实现 `iter_ptq_units(layer_idx, block)`
5. 将模型专属 wrapper 放在模型目录下维护

其中 `amct_pytorch/common/models/llm/qwen/qwen3/qwen3.py` 说明了当前 dense Qwen3 的一个额外适配点：

- `lm_head` 可能与 `embed_tokens` tied，不一定单独落盘
- `attn-linear` 的 step8 必须继续复用官方 `attention_interface`，不能因为只量化 `q/k/v/o` 就切到自定义 attention kernel

## Attention 适配原则

新模型适配时，attention 必须优先遵守下面这条原则：

1. 当前框架默认目标是 `blockwise` 路径上的 BF16 / PPL / PTQ / deploy。
2. 不要为了“贴源码”而默认保留 generate / decode / cache 的完整分支。
3. 如果当前任务不会用到 `past_key_values`、cache 或其它生成态路径，就不要先把这些复杂分支保留进 wrapper。
4. `forward` 里的参数如果在当前路径下恒为 `None`，先判断它是否真的参与计算；如果不参与，就不要为了对齐源码形式而强行保留。
5. attention wrapper 的目标是“足够正确且尽量简单”，不是最大程度复刻上游源码全部枝杈。
6. 同系列模型如果 attention 初始化和 `forward` 完全一样，优先共用一套实现，不要 dense / moe 机械拆两份。
7. `scaled_dot_product_attention` 的调用参数要按当前真实跑通路的传参来收敛，不要默认照搬源码里的 `attention_mask`、cache 或其它生成态参数。
8. 参数是否保留，要按 debug 时看到的真实运行时传参来判断，而不是按源码形参表机械保留。
9. 如果某个参数在当前 blockwise / PPL 路径下恒为 `None`、恒为固定值，或根本不参与实际计算，就不要继续把它当成有效输入向下传递。
10. `attention_mask` 只是其中一个典型例子；后续其它参数如果也是同样情况，也按这个规则处理，优先去掉无效传参。
11. 如果当前流程根本不用 cache，就不要保留 `past_key_values.update(...)` 这类只服务生成态的步骤。
12. `qwen3` 这一类就是明确例子：当前 PPL/PTQ/deploy 路径应优先复用一份 attention 实现，并只保留当前真实执行所需的 `position_embeddings` 和 `scaled_dot_product_attention` 调用参数。

## Wrapper 合并原则

这个原则不只适用于 attention，也适用于 MLP / MoE-MLP：

1. 如果同系列两个 wrapper 的初始化、状态和 `forward` 一样，就优先合并成一份实现。
2. 不要为了保留旧类名、减少局部改动，而长期保留行为完全重复的 wrapper。
3. 只有在模块结构、导出语义、PTQ unit 边界或量化路径真实不同的时候，才拆成两份。
4. “dense”和“moe”标签本身不构成拆分类的理由；关键看实现是否真的不同。

`amct_pytorch/common/models/llm/deepseek/deepseek_v3_2/deepseekv3_2.py` 说明了另一种适配形态：

- attention 侧和 MoE 侧可以走不同的 target 路径
- 但最终仍应复用 `BaseModel`、`PtqUnit` 和通用 quant wrapper 主链

`amct_pytorch/common/models/llm/longcat/longcat_lite/longcat_lite.py` 说明了第三种适配形态：

- 输入路径和 decoder block 拓扑可以很特殊
- 但模型专属的输入重放、quant wrapper 注入和 PTQ unit 路由仍应优先放在 `amct_pytorch/common/models/llm/...`

`amct_pytorch/common/models/llm/qwen/qwen3_next/qwen3_next.py` 说明了第四种适配形态：

- 同一模型家族内部同时混有 `linear_attention` 和 `full_attention`
- blockwise 路径不能只复用 layer0 捕获到的 mask，需要在 adapter 内分别维护 linear-attn mask 和 full-attn causal mask
- 当前 `transformers==5.3.0` 环境下，checkpoint 仍是展开的 per-expert 权重，而运行时 MoE 模块是 packed experts；这类重组优先留在模型 adapter 内

`amct_pytorch/common/models/llm/longcat/longcat_next/longcat_next.py` 说明了第五种适配形态：

- 顶层 `trust_remote_code` 模型可能因为额外依赖（例如 `flash_attn`）而无法直接空载
- 如果 blockwise PTQ 只需要 text-only backbone，可以在 adapter 内直接实例化动态模块中的文本模型类
- 当 checkpoint 的 `lm_head` 形状与默认 config 构造不一致时，可以把最小 shape patch 留在 adapter 内，不要扩散到 workflow 或通用模块

因此，repo-map 记录的应是**适配边界和复用点**，不是把某一个模型族的局部命名当成全仓统一规范。

## 数据与导出链路

当前 `amct_pytorch/common/datasets/ptq_provider.py` 负责：

- `load_unit_inputs()`
- `materialize_gt()`
- `build_unit_batch()`

当前 `BaseSolver.finalize()` 的默认语义是：

- 优先调用模块自己的 `export_ptq_params()`
- 否则回退到导出 `requires_grad=True` 的参数

因此：

- 训练型算法依赖 `trainable_params()`
- 统计型或 buffer 型 PTQ 参数应通过 `export_ptq_params()` 进入保存链路

## 更新规则

只有在**框架边界或核心流程变化**时才更新这份 map，例如：

- `llm_ptq.py` 改变了主流程编排语义
- `amct_pytorch/common/models/llm/__init__.py` 改变了实际注册的模型集合
- `BaseModel` 改变了 block / unit / 导出加载语义
- `quant_base.py` 改变了算法分流方式
- `blockwise_solver.py` 改变了优化职责边界

下面这些情况**不要**重写 repo-map：

- 新增一个尚未接入主注册链的模型目录
- 新增一个算法文件
- 修一个局部 wrapper bug
- 改一个不影响复用边界的模型专属实现细节

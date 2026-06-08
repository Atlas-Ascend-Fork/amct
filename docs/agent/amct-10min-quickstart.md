# amct_llm 新人 10 分钟速读

这份文档只回答三个问题：

1. `amct_llm` 到底是什么。
2. 我第一次读仓库时先看什么。
3. 我要做量化或给 infer 对接时，最短路径是什么。

如果你需要完整原理、方法论、主流方案对比，请再看 `amct-quantization-framework-guide.md`。

---

## 1. 一句话定位

`amct_llm` 不是推理 runtime，而是一个面向大模型的离线量化研发与导出框架。

它主要负责四件事：

- 跑 `BF16` 基线和直转量化基线
- 提取 `PTQ` 所需输入
- 对最小量化单元做 `PTQ`
- 导出下游可消费的量化权重、量化配置和部署说明

和 infer 仓的关系可以简单理解成：

- `amct_llm` 负责“产出什么量化工件”
- `infer` 负责“怎么高效加载并跑起来”

---

## 2. 先记住主链路

```text
eval(bf16) -> eval(quant) -> extract_ptq_data -> ptq -> deploy
```

这条链路背后的思路是：

- 先拿浮点基线
- 先看直转量化够不够
- 不够再做 PTQ
- 最后再把确认可接受的方案导出

也就是说，`amct_llm` 默认不是“一上来就重型 PTQ”，而是先用最小成本判断方案值不值得继续。

---

## 3. 先理解三个核心抽象

### 3.1 `BaseModel`

它定义模型适配器边界。

你可以把它理解成：

- 负责按层加载模型
- 负责告诉 workflow “这一层的量化 block 怎么构”
- 负责告诉 deploy “最终该怎么导”

新模型适配，通常先落在 `amct_pytorch/common/models/llm/...` 这一层。

### 3.2 `PtqUnit`

它是最小 PTQ 单元，不一定是一整层。

常见可以是：

- 一个 attention 子模块
- 一个 MLP
- 一个 expert

这让 PTQ 成本更低，也更容易定位问题。

### 3.3 `deploy export`

它负责把量化结果变成下游可消费的工件，而不是继续做实验决策。

这一步的重点不是“再判断一次精度”，而是“把当前接受的量化方案稳定交付出去”。

---

## 4. 为什么这个框架对新模型友好

`amct_llm` 有几个很实用的设计：

- `blockwise`：默认按 block 分段处理，不要求整模型一次性完整驻留
- `quant_target` 驱动：围绕 `mlp`、`moe`、`attn-linear`、`attn-cache` 这类运行对象，而不是围绕某个模型名硬编码
- 算法和 `dtype` 解耦：`AWQ / GPTQ / SmoothQuant / FlatQuant` 与 `int / mxfp` 这类导出格式不是死绑的
- 实验和导出分离：实验阶段回答“量化值不值得做”，导出阶段回答“如何交付下游”

这些特性让它非常适合：

- 快速试一个新模型
- 只对一个 target 先做验证
- 在不改 runtime 的前提下先把工件做出来

---

## 5. 你第一次读仓库，建议按这个顺序

1. `README.md`
2. `docs/agent/repo-map.md`
3. `amct_pytorch/cli/llm/args.py`
4. `amct_pytorch/workflows/llm_eval.py`
5. `amct_pytorch/workflows/llm_extract_ptq_data.py`
6. `amct_pytorch/workflows/llm_ptq.py`
7. `amct_pytorch/workflows/llm_deploy.py`
8. `amct_pytorch/common/models/llm/common/base.py`
9. `amct_pytorch/common/models/llm/common/quant_apply.py`
10. `amct_pytorch/common/models/llm/common/deploy_export.py`
11. `amct_pytorch/quantization/modules/quant_base.py`
12. 任意一个具体模型 adapter，例如 `amct_pytorch/common/models/llm/deepseek/...` 或 `amct_pytorch/common/models/llm/qwen/...`

最容易上手的阅读方式是：

- 先看 workflow 怎么串起来
- 再看模型 adapter 怎么挂接进去
- 再看 quant module 和算法/dtype 怎么分层
- 最后看 deploy 怎么导出

不要一开始就扎进单个模型实现细节，不然很容易只看到 patch，看不到主线。

---

## 6. 输入、产物、导出，最需要记住什么

### 6.1 输入

最重要的输入可以分三层：

- 模型层：`--model`、`--model_name`
- 量化策略层：`--quant_target`、`--quant_dtype`、各种 bits、算法开关
- 实验与导出层：`--data_dir`、`--output_dir`、已有 PTQ 参数目录

### 6.2 中间产物

最重要的是两类：

- `extract_ptq_data` 导出的 block 输入缓存
- `ptq` 阶段按 unit 保存的参数，例如 `layer_xxx_*.pt`

### 6.3 最终导出物

真正给下游用的，建议永远只盯住这四类：

1. `config.json`
2. `model.safetensors.index.json`
3. `*.safetensors`
4. `deploy_quantization.md`

如果一个下游框架接入时还要反向研究上游 Python 才知道怎么加载，通常说明导出契约还不够稳。

---

## 7. infer 仓最短对接路径

对 infer 来说，最快的消费方式不是“再做一套量化”，而是：

1. 读取 `config.json`
2. 解析 `quantization_config`
3. 建立 `target -> runtime object` 映射
4. 按 `model.safetensors.index.json` 找到对应 shard
5. 加载 `qweight / scale / bias / extra tensors`
6. 执行必要的 post-load 处理
7. 做最小闭环验证，确认量化路径真的生效

infer 最关心的通常不是上游训练细节，而是：

- 这是 `Linear`、`MoEGMM` 还是 `KVCache`
- 需要读哪些张量
- 是否有额外后处理
- 是否允许回退

---

## 8. 新模型量化适配时，优先这样思考

### 场景 A：先确认能不能量化

先跑：

- `BF16` 基线
- 直转量化基线

如果直转已经够好，就没必要急着上 PTQ。

### 场景 B：精度不够

再判断：

- 是不是只有某个 `target` 掉点
- 是不是只需要对少数层做 PTQ
- 是不是结构类算法带来了额外运行时要求

### 场景 C：要交给 infer

优先先把下面几件事说清楚：

- target 对应 infer 的什么运行对象
- 量化张量有哪些
- 是否有结构变换或额外参数
- post-load 要做什么
- 哪些模块允许非融合或浮点回退

---

## 9. 本地方案和主流方案的差别

常见主流路线大致有三类：

- 统一 checkpoint 协议，runtime 解析配置后加载
- runtime 自己定义并消费预量化格式
- 先做专用 conversion，再交给专用 engine 跑

本地方案更像第四类：

- `amct_llm` 负责离线量化研发和导出
- infer 负责按契约接入并高性能运行

它的优点是：

- 对新模型和新 target 更友好
- 实验链路和部署链路边界清晰
- 更容易做 blockwise 和 unit-wise 试错

它的挑战是：

- 导出契约必须稳定
- 文档必须写清楚
- infer 侧需要逐步沉淀稳定加载能力

---

## 10. 新人第一周最值得做到的三件事

1. 跟着主链路把 `eval -> extract -> ptq -> deploy` 走通一遍。
2. 选一个现有模型 adapter，理解 `build_quant_block()` 和 `iter_ptq_units()` 的职责。
3. 打开一份 `deploy_quantization.md` 模板，从 infer 的视角重读一遍，看自己能不能只靠交付物说清怎么加载。

如果这三件事都做完了，你对 `amct_llm` 的整体理解通常就已经够支撑后续开发了。


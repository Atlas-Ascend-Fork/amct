# Classic `quantize` 逐层入口设计（Week 5）

本文定义 `quantize(..., config={"pipeline": "layerwise"})` 的 Week 5 入口骨架。
本阶段只交付分派、校验、配置翻译、模型解析和 Adapter 构造，不执行 Week 6 的
激活 Extract、Week 7 的 Solver/PTQ/manifest-last，也不宣称已完成 Week 8 的真实模型闭环。

## 1. 设计依据与目标

实习计划对 Week 5 的冻结要求是：

1. 为 Classic `quantize` 增加 `full`/`layerwise` 显式分派；
2. `full` 保持现有整模型行为；
3. `layerwise` 拒绝已实例化的 `torch.nn.Module`，接收本地模型目录；
4. 将 Classic config 显式翻译成 Adapter 所需的内部参数；
5. 根据本地 `config.json` 选择唯一的已注册 Adapter；
6. P0 只支持单 target、weight-only 直转或静态 activation MinMax；当前骨架临时只选 FFN；
7. 使用 mock Adapter 单测，不依赖真实模型、联网或 NPU。

现有 CLI 逐层能力仍是独立的上层入口。Classic layerwise 可以复用 Adapter、Provider、
Solver 和 PTQ I/O 等底层原语，但不得调用或复制 `LlmPtqWorkflow.run()`。

## 2. 公共接口合同

公共签名保持为：

```python
quantize(model, config=None, quant_result_dir=None)
```

| pipeline | model | Week 5 行为 |
|---|---|---|
| 未传 | `nn.Module` | 推断为 `full`，执行原有 Classic 整模型流程 |
| 未传 | 字符串 | 推断为 `layerwise`，进入逐层准备流程 |
| `config["pipeline"]="full"` | `nn.Module` | 执行原有 Classic 整模型流程 |
| `config["pipeline"]="full"` | 字符串 | 最终设计要求整网加载，本周明确报未实现 |
| `layerwise` | `nn.Module` | 在 Adapter、权重或设备操作前拒绝 |
| `layerwise` | 本地目录字符串 | 校验并生成内部准备计划 |
| 其他值（含空字符串和大小写变体） | 任意 | 明确拒绝，不静默回退 |

`pipeline` 只来自 config，不增加第二个同名公共参数。未指定时严格按老师总体设计由
`model` 类型推断。总体设计还要求 `full` 最终接收字符串并在内部加载整网，但 Week 5
同时要求 full 旧行为不变，且当前仓库没有冻结整网字符串加载方式；因此本周显式拒绝
该组合，避免把未设计完成的加载行为混入入口 PR。

原有整模型函数体移动到 `_quantize_full`，执行顺序、默认 config、模块替换、
Week 3/4 artifact 写入和返回值均保持不变。

## 3. Week 5 调用链

```text
quantize
  -> _resolve_pipeline
  -> full
       -> _quantize_full（原 Classic 实现）
  -> layerwise
       -> prepare_layerwise_quantization
            -> load_model_config（只读一次 config.json）
            -> _resolve_layerwise_quant_result_dir
            -> build_args_from_config
            -> register_llm_models(model_type=...)（按需注册）
            -> 通过注册表 metadata 反向解析 Adapter key
            -> 构造唯一 Adapter
            -> LayerwiseQuantizationPlan
       -> 显式 NotImplementedError
```

最后的 `NotImplementedError` 是有意的阶段边界：Week 5 没有 Extract 和 PTQ，不能返回
一个看似成功但没有量化 artifact 的结果。Week 6/7 接入真实编排后，该位置才会被
替换为执行逻辑；内部 `prepare_layerwise_quantization` 可由单测直接验证准备结果。

## 4. 模块职责

### `amct_pytorch/classic/quantize.py`

- 解析唯一 pipeline；
- 校验 pipeline 与 model 输入类型组合；
- 将 `full` 交给原 Classic 实现；
- 将 `layerwise` 交给准备模块；
- 不包含 Adapter、Extract、Provider 或 Solver 实现。

### `amct_pytorch/classic/layerwise.py`

- 校验并加载本地 `config.json`；
- 解析 `config.json:model_type`；
- 校验 Week 5 P0 配置边界；
- 把 Classic config 翻译成类型明确的 `LayerwiseArgs`；
- 选择并构造一个注册 Adapter；
- 返回只供后续阶段消费的 `LayerwiseQuantizationPlan`。

### `amct_pytorch/common/models/llm`

- Adapter 注册项增加 HuggingFace `model_type` metadata；
- `register_llm_models(model_type=...)` 只导入目标 Adapter；
- 原有无参数调用仍执行全量注册，CLI 调用方式不变。

按 `model_type` 延迟导入可以避免解析一个已支持模型时，被其他无关模型的可选
Transformers 依赖阻塞。未知 `model_type` 不触发任意模块导入，最终由模型解析器给出
包含可用注册项的错误。

## 5. 本地模型解析合同

`load_model_config` 只接受非空字符串形式的本地目录，目录中必须存在 UTF-8 编码的
`config.json`，顶层必须是 JSON object，并包含非空字符串 `model_type`。

解析器拒绝：

- 不存在的路径；
- 普通文件或缺少 `config.json` 的目录；
- 无效 UTF-8/JSON；
- 重复 JSON key；
- 非 object 顶层；
- 缺失、空值或非字符串 `model_type`。

模型名不根据目录名、类名或模糊字符串猜测。解析器只匹配
`RegistryItem.metadata["model_type"]`，并要求结果恰好为一个：

- 零个匹配：列出模型配置路径和当前可用 Adapter；
- 多个匹配：列出所有冲突 key，拒绝任选一个；
- 一个匹配：用注册表 key 取得 Adapter class。

## 6. Classic config 翻译合同

Classic config 是公共用户配置，不等于 CLI `argparse.Namespace`。翻译只读取以下字段；
未知根字段、未知 `quant_cfg` 字段和 MinMax 多余参数全部拒绝，防止拼写错误被静默忽略。

| Classic 字段 | 内部字段 | 规则 |
|---|---|---|
| `quant_cfg.weights.type` | `w_bits` | `int8 -> 8`，`int4 -> 4` |
| `quant_cfg.inputs` 缺失/禁用 | `a_bits` | `16`，表示 weight-only |
| 静态 `inputs.type=int8` | `a_bits` | `8` |
| `algorithm` | `algos` | weight-only 可为空；静态 activation 必须是无额外参数的 `minmax` |
| `batch_num` | `batch_num` | 正整数；三个样本字段都未提供时默认 128 |
| `batch_size` | `cali_bsz` | 正整数，默认 1 |
| `num_calibration_samples` | `nsamples` | 必须等于 `batch_num * batch_size` |
| `max_seq_length` | `seq_len` | 正整数，默认 4096 |
| `device` | `device` | 非空字符串，默认 `npu:0` |
| `dataset` | `dataset` | 可选本地路径；Week 6 才读取 |
| `work_dir` | `work_dir`/`output_dir` | 可选；默认 `<quant_result_dir>.work` |
| `skip_layers` | `skip_layers` | list/tuple/set，元素由 Classic 字段校验 |
| `trust_remote_code` | 同名字段 | 严格 bool，默认 `False` |

样本字段可以只提供一部分，但三个规范化值必须满足：

```text
nsamples = batch_num * cali_bsz
```

三个字段都未提供时得到 `(128, 1, 128)`；只提供 `batch_size=2` 时使用默认 128 个样本，
得到 `(batch_num, cali_bsz, nsamples)=(64, 2, 128)`；
提供 `batch_size=2, num_calibration_samples=10` 时得到 `(5, 2, 10)`；三个字段
同时存在但关系不成立时直接拒绝。

## 7. P0 target 与算法边界

Week 5 不开放公共 `quant_target`。当前准备骨架临时生成一个 FFN target：

- 配置含正整数专家数量字段时选择 `moe`；
- 其他已注册模型选择 `mlp`。

老师只冻结了 P0 为单 target，没有明确指定它必须是 FFN；详细设计又要求从
`quant_cfg` 推导 target。因此“FFN-only”仍是待 mentor 确认的临时边界，不能据此推断
最终 API 会忽略 attention。

当前 P0 接受：

- weight-only 直转或 MinMax：W8A16 或 W4A16；
- 静态 activation MinMax：W8A8 或 W4A8；
- 单个 `mlp` 或 `moe` target。

当前明确拒绝：

- AWQ、GPTQ、SmoothQuant 或算法组合；
- 动态 activation；
- KV cache；
- fuzzy layer 配置；
- 多 target 和通用 `block_spec`；
- 未注册模型和远程模型标识。

## 8. 路径与产物边界

layerwise 的 `quant_result_dir` 未提供时解析为当前目录下的 `./quant_result`；显式路径和
默认路径都只在 Week 5 规范化，不要求预先存在，也不创建目录。若路径已存在，则必须是
可读、可写且可搜索的目录。目录创建、不可变 artifact 冲突检查和 manifest-last 发布由
Week 7 的 writer 接线负责。`full` 继续沿用 Week 3/4 的调用者预创建目录合同。

`work_dir`、`data_dir=<work_dir>/ptq_data/<target>` 和 `output_dir` 只写入内部 args，
且 `work_dir` 与 `quant_result_dir` 必须是互不包含的独立目录树。Week 5 不创建这些目录；
目录创建、校准数据 schema、激活缓存和读回属于 Week 6。

## 9. 测试策略与完成标准

`tests/unit_test/classic/test_layerwise.py` 使用 toy `nn.Module`、临时目录、fake Registry
和 fake Adapter，验证：

1. 未传 `pipeline` 时按 model 类型推断，显式 `full` 走原实现；
2. 类型组合、非字符串和未知 pipeline 被早期拒绝；
3. 本地 config 的路径、JSON、重复 key 和 `model_type` 校验；
4. 注册表零/一/多匹配行为；
5. weight-only、静态 activation、位宽、样本数、路径和 MoE target 映射；
6. P0 外算法、动态 activation、KV cache、fuzzy/未知字段被拒绝；
7. 只构造解析出的一个 mock Adapter；
8. 按 `model_type` 注册时只导入对应 Adapter；
9. 公共入口不会在没有 Extract/PTQ 时静默宣称完成。

Week 5 的客观 DoD 是：聚焦单测、原 Classic 回归、Week 3/4 artifact 回归、LLM
workflow 回归、Ruff、格式和 `git diff --check` 全部通过；提交中不包含模型、缓存、
benchmark、截图或 Week 6-8 实现。

## 10. Week 6 接口交接

Week 6 只能在 `LayerwiseQuantizationPlan` 已成功生成后继续，并负责：

1. 定义并校验显式本地 calibration data schema；
2. 根据 weight-only/静态 activation 实现 `_needs_extract`；
3. 创建工作目录和激活 I/O 目录；
4. 复用 Adapter 的 embedding/block 原语完成一层 Extract 与读回；
5. 不调用 CLI workflow 的 `run()`。

真实模型/NPU 端到端不属于 Week 5 验收。Adapter metadata 与 mock 合同已测试，真实
checkpoint、tokenizer、Transformers 版本和 NPU 资源必须在 Week 8 的小模型闭环中提供
独立运行证据。

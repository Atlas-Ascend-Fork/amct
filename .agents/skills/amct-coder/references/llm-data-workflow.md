# 子能力：llm-data-workflow

## 触发

修改 LLM new-path 的 `eval`、`extract_ptq_data`、PTQ workflow、PPL、Wikitext、Pileval 校准样本、离线输入、GT、hook、dataloader、逐层 forward、`attention_mask`/`position_embeddings` 保存加载时使用。

“校准”本身不足以决定使用本手册：classic eager/graph_based 调用链内的校准使用 `classic-graph`；LLM 算法内部统计使用 `llm-algorithm`；通用 observe/min-max/scale 语义使用 `dtype-quant-module`。

## 源码锚点（按需选择）

以下路径不是全文必读清单。先按 eval、extract、PTQ 或 data provider 的当前动作选择直接链路，再按保存端/加载端契约扩展。

- `amct_pytorch/cli/llm/args.py`
- `amct_pytorch/workflows/llm_eval.py`
- `amct_pytorch/workflows/llm_extract_ptq_data.py`
- `amct_pytorch/workflows/llm_ptq.py`
- `amct_pytorch/common/datasets/preproc.py`
- `amct_pytorch/common/datasets/ptq_io.py`
- `amct_pytorch/common/datasets/ptq_provider.py`
- `amct_pytorch/common/evaluate/eval_ppl.py`
- `amct_pytorch/common/models/llm/common/base.py`
- `tests/unit_test/workflows/`
- `tests/unit_test/common/datasets/`
- `tests/unit_test/common/evaluate/`

## 上下游依赖

- 上游：数据流依赖 adapter 的 embedding/block forward、hook 名、`PtqUnit.kind` 和需要保存的 kwargs。
- 平级：workflow 只编排 eval、extract、PTQ，不写模型结构、dtype 或算法分支。
- 下游：`ptq-solver-params` 消费 `load_unit_inputs()`、`materialize_gt()` 和 PTQ kwargs；`direct-quant-eval` / `algorithm-validation` 依赖 eval 口径一致。
- 若新增 CLI 参数或用户可见数据格式，要同时读 `public-api-tests`。

## 执行步骤

1. 确认任务属于 eval、extract、PTQ 编排还是 data provider；不要把模型特例写入 workflow。
2. 从 adapter 的 `PtqUnit.kind` 反推需要的输入文件名、kwargs 文件和 GT 生成方式，并区分源码契约、磁盘产物与实际运行证据。
3. 修改 extract 时，同时检查保存端 `save_ptq_inps()` / `save_ptq_kwargs()` 和加载端 provider。
4. 修改 PTQ workflow 时，确认 unit 枚举、输入加载、GT materialize、solver 调用、参数保存的顺序不变。
5. 修改 eval 时，确认 BF16 与 quant 使用相同 `seq_len`、mask/position 传递和 bit policy 口径。
6. 对缺文件、空输入、NaN/恒 0 loss 等失败信号保留显式错误或日志，不 silent skip。
7. 补 workflow、data I/O、provider、PPL 或 CLI 的最小测试。

## 证据等级

数据流诊断必须把以下证据分开表述：

1. **源码契约对齐**：保存端构造的文件名、`PtqUnit.kind` 和加载端期望在当前源码中一致。
2. **产物存在**：目标 `data_dir` 中确实存在对应 layer、kind 和 kwargs 文件，且来源代码版本与本次分析一致。
3. **命令口径一致**：extract 与 PTQ 的 `data_dir`、`quant_target`、block 范围、`seq_len`、`nsamples` 等相关参数一致。
4. **集成闭环**：provider 实际加载输入/kwargs，GT materialize 和 solver 至少一个 batch 成功。

源码中的保存/加载前缀相同，只能排除当前代码里的直接静态命名错位，不能证明已有磁盘目录正确，也不能表述为 extract/PTQ 已跑通。反过来，文件存在也不能证明 kwargs shape、版本或 forward 消费语义正确。

只读诊断应先给出已达到的证据等级，再列下一等级所需的最小检查。没有真实产物或运行结果时，使用“静态一致”“候选根因”“建议验证”，不使用“已经正确”“完整闭环”。

## 离线格式演进门禁

扩展 kwargs 或 unit 输入格式时，先冻结以下契约，再确定容器、文件布局或实现接口：

| 契约 | 必须回答的问题 |
| --- | --- |
| 字段与所有权 | 精确字段集合是什么；由 adapter 捕获/生成，还是通用 I/O 只负责序列化 |
| 基数与作用域 | 每个 sample、batch、layer、unit 还是全局一份；不同 block 范围如何对应 |
| shape 与切批 | tensor/tuple/dict 的 shape、dtype；per-sample 字段如何与 hidden states 同步切 batch |
| 必填与默认 | 哪些 unit 必需、哪些可选；缺失时能否从稳定事实推导默认值 |
| 容器与版本 | 延续现有文件、结构化容器还是目录布局；是否需要版本标记和迁移 |
| 旧目录行为 | 旧 tensor-only 数据是无损读取、显式降级、拒绝并要求重新 extract，还是其它已定义策略 |

任一答案会改变布局或回放语义而尚未冻结时，只给条件分支。不得先锁定“任意 kwargs dict”、版本文件、按层/按 unit 文件名、目录 schema 或 CLI flag。per-sample 状态必须进入与输入相同的 batch 切分链，不能把全量 kwargs 原样传给每个 batch。

“兼容旧 tensor-only 目录”不等于为新必填字段静默填空。允许的兼容结果必须明确是继续读取、带可见降级读取，还是报错并提示重新 extract。unit 主输入缺失默认 fail-fast；若产品确需跳过，这会改变输出模型语义，必须先冻结显式 opt-in、结果缺失标记、eval/deploy 回载行为和用户可见接口，不能在规划中自行发明参数名。

## 数据流

extract:

1. `get_pileval()` 生成 calibration samples。
2. adapter `do_embedding_forward(..., hook_name=...)` 捕获 layer0 输入上下文。
3. `do_block_forward(..., hook_name=...)` 逐层保存 hook 输出。
4. `save_ptq_inps()` 写 `block_<layer>_<target>_in.pkl`。
5. `save_ptq_kwargs()` 写 `position_ids.pkl`、`position_embeddings.pkl`、`attention_mask.pkl`。

PTQ:

1. workflow 枚举 layer。
2. adapter `build_quant_block()`。
3. adapter `iter_ptq_units()`。
4. data provider `load_unit_inputs()`。
5. `materialize_gt()` 用关闭量化的 unit 生成 GT。
6. solver 训练并保存 unit 参数。

eval:

1. `get_wiki_inputs(seq_len=4096)`。
2. blockwise eval 走 embedding -> 每层 block -> head -> `wikitext2_ppl()`。
3. `eval_mode=bf16` 要求 bit policy 没有小于 16 的量化项。
4. `eval_mode=quant` 会重建 quant block，并按 bit policy 决定是否真正开启 quantizer。

## 规则

- PPL 默认口径是 Wikitext `seq_len=4096`。
- `extract_ptq_data` 和 PTQ 当前只支持单个 `quant_target`。
- 多目标 eval/deploy 允许组合，但参数目录必须按 target 对齐。
- attention unit 需要额外 kwargs 时，不要只用 tensor-only provider；要扩展 `ptq_io` / provider 的结构。
- `load_ptq_inps()` 返回 `None` 表示文件不存在，不能 silent skip 掩盖。
- 当前源码命名一致不等于真实离线目录或 extract/PTQ 集成已验证。

## BLOCKER

- BF16 PPL 离谱却不检查 mask/position 传递链。
- unit `kind` 找不到对应离线输入文件。
- 只有静态保存/加载链证据，却声称真实产物或 PTQ 输入链已经正确。
- kwargs 格式契约未冻结，却先确定通用 dict、版本标记、文件布局或 CLI 名称。
- per-sample kwargs 没有与输入同步切 batch，而是整份回放到每个 batch。
- `materialize_gt()` 未关闭量化或 observe 状态不对。
- eval 和 PTQ 使用不同 `seq_len` 却直接比较结论。
- workflow 中加入模型专属 hook 名或结构分支。

## 自验

- workflow 逻辑：`pytest tests/unit_test/workflows/test_llm_eval.py tests/unit_test/workflows/test_llm_extract_ptq_data.py tests/unit_test/workflows/test_llm_ptq.py`。
- 数据 I/O：`pytest tests/unit_test/common/datasets/test_ptq_io.py tests/unit_test/common/datasets/test_ptq_provider.py`。
- PPL：`pytest tests/unit_test/common/evaluate/test_eval_ppl.py`。
- 真实模型路径受环境限制时，最终回复明确未跑原因。

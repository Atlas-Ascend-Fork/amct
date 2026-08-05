# 子能力：llm-data-workflow

## 触发

用户提到 `eval`、`extract_ptq_data`、PTQ workflow、PPL、Wikitext、Pileval、离线输入、GT、hook、dataloader、逐层 forward、`attention_mask`/`position_embeddings` 保存加载时使用。

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
2. 从 adapter 的 `PtqUnit.kind` 反推需要的输入文件名、kwargs 文件和 GT 生成方式。
3. 修改 extract 时，同时检查保存端 `save_ptq_inps()` / `save_ptq_kwargs()` 和加载端 provider。
4. 修改 PTQ workflow 时，确认 unit 枚举、输入加载、GT materialize、solver 调用、参数保存的顺序不变。
5. 修改 eval 时，确认 BF16 与 quant 使用相同 `seq_len`、mask/position 传递和 bit policy 口径。
6. 对缺文件、空输入、NaN/恒 0 loss 等失败信号保留显式错误或日志，不 silent skip。
7. 补 workflow、data I/O、provider、PPL 或 CLI 的最小测试。

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

## BLOCKER

- BF16 PPL 离谱却不检查 mask/position 传递链。
- unit `kind` 找不到对应离线输入文件。
- `materialize_gt()` 未关闭量化或 observe 状态不对。
- eval 和 PTQ 使用不同 `seq_len` 却直接比较结论。
- workflow 中加入模型专属 hook 名或结构分支。

## 自验

- workflow 逻辑：`pytest tests/unit_test/workflows/test_llm_eval.py tests/unit_test/workflows/test_llm_extract_ptq_data.py tests/unit_test/workflows/test_llm_ptq.py`。
- 数据 I/O：`pytest tests/unit_test/common/datasets/test_ptq_io.py tests/unit_test/common/datasets/test_ptq_provider.py`。
- PPL：`pytest tests/unit_test/common/evaluate/test_eval_ppl.py`。
- 真实模型路径受环境限制时，最终回复明确未跑原因。

# LiveCodeBench 专项流程（hidden-tests pass@1）

代码生成类任务不走 lm-eval：本 skill 只负责在 Ascend 侧生成代码并写出官方 `custom_evaluator` 输入，**评分由官方 `lcb_runner` 完成**，避免自造指标。

## 前置

- 装好官方 runner：从官方仓装（`git clone https://github.com/LiveCodeBench/LiveCodeBench.git` 后在仓内 `pip install -e .`），PyPI 上没有这个包；`python <skill>/scripts/check_env.py --livecodebench_score` 可校验。
- 预下载数据：`python <skill>/scripts/prefetch_data.py --task livecodebench_release_v6`（累计集，一次预取；体量看 `prefetch_data.py` 打印的实际命中文件）。链路冒烟用 `--limit N` 少量生成即可（部分输出过不了官方评分的条数校验，冒烟只验生成、不评分）。
- 其余装配前提（KV cache 判断、数据本地读取、PYTHONPATH 前置、独立工作目录）同 [constraints.md](constraints.md)。

## 1. 生成官方 custom 输出

```bash
python <skill>/scripts/livecodebench_eval.py --model <model_path> --model_name qwen3 \
  --param_dir <ptq_params/mlp> --dataset_task livecodebench_release_v6 \
  --release_version release_v6 --max_gen_toks 2048 \
  --output_dir <out> \
  --output_json <out>/livecodebench_quant_custom_outputs.json \
  --raw_output_jsonl <out>/livecodebench_quant_raw.jsonl
```

- `--model_name` / `--output_dir` / `--output_json` 都是**必填**：`--model_name` 填错会把 MoE 按 dense 装配、专家参数一个都加载不上；`--output_json` 必填是为了产物落在用户选的运行目录、不落调用时的 cwd（SKILL.md §4.3）；`--output_dir` 沿用上游产 ptq 参数的同名目录参数，评测路径不往它写文件。
- `--device_map` 省略时按 `config.json` 推断：超单卡预算自动走 `auto`，尺寸判不了直接报错要求显式传；卡数与 `ASCEND_RT_VISIBLE_DEVICES` 由 `plan_device.py` 打出（每卡显存小于预算时的放大办法见 [env-and-portability.md](env-and-portability.md) §6「定卡数」）。
- `--quant_target` 省略时按 `--param_dir` 推导：目录里有 `layer_*_self_attn.pt` / `layer_*_linear_attn.pt` 就默认连 attention 一起评，此时必须加 `--allow_attn_quant`，且先按 [constraints.md](constraints.md) §1 与用户确认代价。
- 评直转时把 `--param_dir` 换成 `--direct` + 显式 `--quant_target`（无参数可读、全层量化，SKILL.md §4.2 变体表），其余逐项不变。
- 加 `--bf16` 跑基线；bf16 与量化各跑一次生成 + 一次官方评分，最后对比两个 pass@1。
- 模型目录带自定义建模代码时才加 `--trust_remote_code`（默认关，会执行该目录下的 Python，见 [constraints.md](constraints.md) 约束 5）。
- 脚本在**加载模型之前**解析数据集并打一行 `[data] <dataset_task>: N local file(s) from $BENCH_ASSETS`：缓存没预取、或 `all_of` 累积分片缺片都当场失败，不会等到生成跑到一半才报文件找不到。
- 脚本读取 `livecodebench/code_generation_lite` 的 `question_content`、`starter_code` 等字段拼 prompt，生成 Python 代码，写出官方格式：`[{"question_id": "...", "code_list": ["..."]}]`；`--raw_output_jsonl` 额外留一份原始生成便于排查。
- **不要加 `--limit` 产出正式分数**：官方 `custom_evaluator` 要求 custom 输出条数与所选 release 的 benchmark 条数一致。`--limit` 只用于生成链路冒烟。

## 2. 官方评分

```bash
python -m lcb_runner.runner.custom_evaluator --scenario codegeneration \
  --release_version release_v6 \
  --custom_output_file <out>/livecodebench_quant_custom_outputs.json
```

bf16 的 custom 输出同样跑一遍，得到基线 pass@1。

`custom_evaluator` 会把分数写进 `<custom_output_file 去掉 .json>_codegeneration_output_eval.json`（内容是 `[pass_at_k, results, final_metadata]`，`pass_at_k` 里就是 `pass@1` / `pass@5` …）。**pass@1 只存在于这个文件**：生成脚本的日志只打 `[lcb-gen]`（样本数 + 输出路径），不含分数，拿它去 `compare_results.py` 会直接报错。

生成脚本另在 `--output_json` 旁边写一份 `<custom_output_file 去掉 .json>_run_meta.json`，内容是本次运行的参数（`release_version` / `max_gen_toks` / 样本量 / 量化侧各项）。官方 `*_eval.json` 只有分数，`compare_results.py` 按上面那条文件名约定找到这份 sidecar，两侧才核得上可比参数。它必须与 `*_eval.json` 同目录、同前缀：搬动或改名其中一个，核对就失效，输出里的 `comparability_gap` 会点名是哪一侧。

### 从 `*_eval.json` 里取 `pass@1` 的步骤

1. 确认拿到的是 **evaluator 输出**，文件名以 `_codegeneration_output_eval.json` 结尾；生成脚本写出的 `*_custom_outputs.json`（内容是 `[{"question_id": ..., "code_list": [...]}]`）里没有分数。
2. 两侧（bf16 / quant）都用同一个 `--release_version` 各跑一次 `custom_evaluator`，得到两个 `_eval.json`。
3. 用 `compare_results.py` 对比，`--metric pass@1`。
4. 想直接看数字：`pass@1` 在 list 第一个元素里，与 `pass@5` 平级。

### 2.1 对比两侧 pass@1

```bash
python <skill>/scripts/compare_results.py \
  --bf16 <out>/livecodebench_bf16_custom_outputs_codegeneration_output_eval.json \
  --quant <out>/livecodebench_quant_custom_outputs_codegeneration_output_eval.json \
  --metric pass@1
```

样本量与可比参数从同目录的两份 `*_run_meta.json` 自动带出，不用手输 `--sample_len`。输出里 `compared_from` 是 `run metadata (both sides)` 才算核对过；是 `none` 就照 `comparability_gap` 说的做。

## 3. 与榜单严格对齐时要核对

prompt 模板、`release_version`、样本全集、`max_gen_toks`、是否 thinking 模式——任一项与目标口径不一致，pass@1 就不能直接和榜单数字横比。

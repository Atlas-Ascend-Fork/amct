# LiveCodeBench 专项流程（hidden-tests pass@1）

代码生成类任务不走 lm-eval：本 skill 只负责在 Ascend 侧生成代码并写出官方 `custom_evaluator` 输入，**评分由官方 `lcb_runner` 完成**，避免自造指标。

## 前置

- 装好官方 runner：`pip install --user livecodebench`（或按官方仓 README 安装）；`python scripts/check_env.py --livecodebench_score` 可校验。
- 预下载数据：`python scripts/prefetch_data.py --task livecodebench_release_v6`（累计集，约 4GB+）。只想跑单 release 冒烟用 `--task livecodebench`（仅 `test6.jsonl`）。
- 其余装配前提（KV cache 判断、离线加载、PYTHONPATH 前置、独立工作目录）同 [constraints.md](constraints.md)。

## 1. 生成官方 custom 输出

```bash
python scripts/livecodebench_eval.py --model <model_path> --model_name qwen3 \
  --param_dir <ptq_params/mlp> --dataset_task livecodebench_release_v6 \
  --release_version release_v6 --max_gen_toks 2048 \
  --output_json <out>/livecodebench_quant_custom_outputs.json \
  --raw_output_jsonl <out>/livecodebench_quant_raw.jsonl
```

- 加 `--bf16` 跑基线；bf16 与量化各跑一次生成 + 一次官方评分，最后对比两个 pass@1。
- 模型目录带自定义建模代码时才加 `--trust_remote_code`（默认关，会执行该目录下的 Python，见 [constraints.md](constraints.md) 约束 12）。
- 多卡（35B MoE）加 `--device_map auto`，并设 `ASCEND_RT_VISIBLE_DEVICES=0,1`。
- 脚本在**加载模型之前**解析数据集并打一行 `[data] <dataset_task>: N local file(s) from $BENCH_ASSETS`：
  缓存没预取、或 `all_of` 累积分片缺片（`config.UnresolvedSplit` 延迟到这里抛出）都会当场失败，
  不会等到模型装配完、生成跑了半小时才在 `question_content` 上报 `FileNotFoundError`。
  没看到这行 `[data]` 就说明用的不是当前版本的脚本。
- 脚本读取 `livecodebench/code_generation_lite` 的 `question_content`、`starter_code` 等字段拼 prompt，生成 Python 代码，写出官方格式：`[{"question_id": "...", "code_list": ["..."]}]`；`--raw_output_jsonl` 额外留一份原始生成便于排查。
- **不要加 `--limit` 产出正式分数**：官方 `custom_evaluator` 要求 custom 输出条数与所选 release 的 benchmark 条数一致。`--limit` 只用于生成链路冒烟。
- 含 attention 量化的参数目录同样受约束 1 管辖，需显式 `--allow_attn_quant` 并先与用户确认。

## 2. 官方评分

```bash
python -m lcb_runner.runner.custom_evaluator --scenario codegeneration \
  --release_version release_v6 \
  --custom_output_file <out>/livecodebench_quant_custom_outputs.json
```

bf16 的 custom 输出同样跑一遍，得到基线 pass@1。

`custom_evaluator` 会把分数写进 `<custom_output_file 去掉 .json>_codegeneration_output_eval.json`（内容是 `[pass_at_k, results, final_metadata]`，`pass_at_k` 里就是 `pass@1` / `pass@5` …）。**pass@1 只存在于这个文件**：生成脚本的日志只打 `[lcb-gen]`（样本数 + 输出路径），不含分数，拿它去 `compare_results.py` 会直接报错，而不是给你一个假的 `delta 0.0`。

## 2.1 对比两侧 pass@1

```bash
python scripts/compare_results.py \
  --bf16 <out>/livecodebench_bf16_custom_outputs_codegeneration_output_eval.json \
  --quant <out>/livecodebench_quant_custom_outputs_codegeneration_output_eval.json \
  --metric pass@1 --sample_len <N>
```

`compare_results.py` 认识上面那个 list 形态的 `_eval.json`，会自动取出含 `pass@` 的字典；`sample_len` / `total` 这类非指标键被排除在候选之外，不会被当成精度指标挑中。

## 3. 与榜单严格对齐时要核对

prompt 模板、`release_version`、样本全集、`max_gen_toks`、是否 thinking 模式——任一项与目标口径不一致，pass@1 就不能直接和榜单数字横比。

## 报告要求

写明 `release_version`、样本条数、`max_gen_toks`、bf16 与量化各自的 pass@1 及差值；用了 `--limit` 的结果只能称小样本冒烟，不能称官方分数。

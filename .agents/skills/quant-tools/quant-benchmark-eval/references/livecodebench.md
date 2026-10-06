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
- 多卡（35B MoE）加 `--device_map auto`，并设 `ASCEND_RT_VISIBLE_DEVICES=0,1`。
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

## 3. 与榜单严格对齐时要核对

prompt 模板、`release_version`、样本全集、`max_gen_toks`、是否 thinking 模式——任一项与目标口径不一致，pass@1 就不能直接和榜单数字横比。

## 报告要求

写明 `release_version`、样本条数、`max_gen_toks`、bf16 与量化各自的 pass@1 及差值；用了 `--limit` 的结果只能称小样本冒烟，不能称官方分数。

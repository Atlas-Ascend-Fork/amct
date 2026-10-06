---
name: quant-benchmark-eval
description: 在 Ascend NPU 上用 lm-eval-harness / LiveCodeBench 官方 runner 评测 amct 伪量化模型的生成式下游精度，并与 bf16 基线对比。触发场景：需要 WikiText2 PPL 之外的下游准确率。不适用于：跑量化命令（→ $quant-run）、判达标（→ $direct-quant-eval / $algorithm-validation）、补 KV cache（→ $model-adapter）。
---

# 量化 Benchmark 评测（quant-benchmark-eval）

## 概述

在 Ascend NPU 上对 **amct 伪量化** 大模型跑生成式下游 benchmark：把量化后的模型**装配成一个完整、可 `generate` 的对象**，再交 lm-eval-harness（或 LiveCodeBench 官方 runner）打分，并与 bf16 基线做同口径对比。

**只评测，不量化**：`extract_ptq_data` 与 `ptq` 由量化流程完成，本 skill 只消费已有 ptq 参数。

它回答的问题是：

- 量化模型在下游任务（通用知识 / 数学 / 代码）上的准确率是多少
- 与 bf16 基线的差值是否落在统计噪声范围内
- WikiText2 PPL 之外的口径能否支撑交付结论

它不负责：

- 跑量化命令、产 ptq 参数（→ `$quant-run`）
- 判 PPL `delta` 是否达标（→ `$direct-quant-eval`）、判 PTQ 收益（→ `$algorithm-validation`）
- 补 / 改 attention wrapper 的 KV cache（→ `$model-adapter`）
- 导出部署权重与交付文档（→ `$deploy-export`）

---

## 先读哪些文件

按需读，不要一次全读：

| 什么时候读 | 读什么 |
|---|---|
| **任何一次评测前（必读）** | [references/constraints.md](references/constraints.md) —— 9 条踩坑约束，编号稳定 |
| 换机器 / 配 env / 装依赖 | [references/env-and-portability.md](references/env-and-portability.md) |
| 要跑代码生成任务 | [references/livecodebench.md](references/livecodebench.md) |

---

## 覆盖度状态

⚠️ 下表结论均基于特定硬件：**2× Ascend 910（每卡 64GB HBM，即 65536 MB）**。其中显存相关判断（能否单卡放下、是否 OOM、要几张卡）与速度数字（约 6 秒 / 题等）强依赖这套配置；换用显存更大 / 更小、算力不同的卡后结论会变——例如更大显存上约束 8 的 OOM 可能不再出现，单卡也可能放得下 35B。迁移到其他硬件时把这些当参考量级、重新实测确认，不要当普适结论。

| 场景 | 状态 |
|---|---|
| MMLU-Pro，8B W8A8 仅 mlp，单卡 | ✅ 已验证 |
| 35B MoE 双卡 bf16（跨卡 generate + HFLM + 打分） | ✅ 已验证 |
| 35B MoE 双卡 W8A8 装配 + 跨卡 generate 冒烟 | ✅ 已验证（需约束 7 的 hook 修复） |
| 35B MoE 双卡 W8A8 + 长 prompt 评测 | ⚠️ npu:0 OOM（见约束 8，属显存调优） |
| MATH500，lm-eval `hendrycks_math500` / 别名 `math500` | ✅ 已接入（耗时需按机器实测） |
| LiveCodeBench code-generation-lite，官方 custom_evaluator hidden-tests pass@1 | ⚠️ 已接入生成官方 custom 输出；评分需安装并调用 `lcb_runner` |
| ARC-Challenge / ARC-Easy / BoolQ / COPA / PIQA / SciQ | ⚠️ 已登记为低资源候选；需在目标机器预下载并跑小样本确认 |

---

## 任务选择与资源成本

**不要默认替用户决定 benchmark**：先按用户的资源、时间预算、能力覆盖需求给候选，让用户选。

| 成本档 | 任务 | 默认主指标 | 适用场景 | 注意 |
|---|---|---|---|---|
| 低 | `arc_challenge` | `acc_norm` | 首选轻量下游回归；科学 / 常识推理，比 ARC-Easy 更敏感 | 同时保留 `acc`；适合先 `--limit` 冒烟 |
| 低 | `arc_easy` | `acc_norm` | 更便宜的科学问答补充 | 难度低，大模型区分度有限 |
| 低 | `boolq` | `acc` | 阅读理解 yes/no 回归 | 二分类，随机基线高，小样本波动大 |
| 低 | `copa` | `acc` | 常识因果二选一 smoke | 样本少，只能作补充，不要单独下结论 |
| 低 | `piqa` | `acc_norm` | 物理常识 / 日常操作推理 | 比 COPA 更稳，适合低成本补充 |
| 低 | `sciq` | `acc_norm` | 科学知识多选补充 | 与 ARC 能力有重叠，难度较低 |
| 高 | `mmlu_pro` | `exact_match,none` | 覆盖面更广的通用知识 / 推理 | 题量大，建议先 `--limit` |
| 高 | `math500` | `exact_match,none` | 数学推理 | 长生成，需更大 `--max_gen_toks`，耗时高 |
| 高 | `livecodebench` / `livecodebench_release_v6` | `pass@1` | 代码生成 hidden-tests pass@1 | 最耗时之一，评分依赖 `lcb_runner` |

- 低资源任务的汇总指标固定为：`arc_challenge` / `arc_easy` / `piqa` / `sciq` 看 `acc_norm`，`boolq` / `copa` 看 `acc`。做 bf16 vs quant 对比时**必须同任务同指标**，不要把 `acc` 与 `acc_norm` 混用；其它指标可留在明细里。
- 资源有限时优先建议：`WikiText2 PPL + arc_challenge`；有余量再加 `piqa` / `boolq` / `sciq` / `arc_easy` / `copa`；要综合结论再升级 `mmlu_pro`；要数学或代码能力再跑 `math500` / `livecodebench`。
- `--limit` 的结果只能称**小样本回归 / 冒烟**，不能称官方分数。

---

## 工作流

> 下文 `scripts/` 均相对本 skill 目录（`.agents/skills/quant-tools/quant-benchmark-eval/`）；按约束 9 要在独立工作目录运行，实际调用时写完整路径。

### 1. 环境与数据准备（一次）

```bash
python scripts/self_check.py                                  # 轻量自检：不加载模型、不占 NPU
python scripts/check_env.py                                   # 依赖 + lm-eval 版本检查
python scripts/prefetch_data.py --task <task>                 # 预下载到 $BENCH_ASSETS/<subdir>/
```

`<task>` 取上表任务名（`livecodebench` 只下单 release 的 `test6.jsonl`，累计集用 `livecodebench_release_v6`）。env 变量、依赖版本、离线读取机制见 [references/env-and-portability.md](references/env-and-portability.md)。

### 2. 装配 + 评测

```bash
# 8B dense，单卡
python scripts/assemble_and_eval.py --model <model_path> --model_name qwen3 \
  --param_dir <ptq_params/mlp> --task mmlu_pro --limit 10 --max_gen_toks 512 \
  --output_dir <out_dir>

# 35B MoE，双卡（先设 ASCEND_RT_VISIBLE_DEVICES=0,1）
python scripts/assemble_and_eval.py --model <model_path> --model_name qwen3_5_moe \
  --param_dir <ptq_params/moe> --task mmlu_pro --device_map auto --limit 10

# 低资源任务：换 --task arc_challenge / arc_easy / boolq / copa / piqa / sciq，--max_gen_toks 64
# MATH500：--task math500（别名，映射到 lm-eval hendrycks_math500）--max_gen_toks 1024
#         也可直接传 hendrycks_math500 / minerva_math500 换 prompt 口径
```

- **装配流程**：`_load_float()`（支持 `device_map`）加载模型 → 逐层 `build_quant_block`（KV-safe 目标：dense 用 `mlp`，MoE 用 `moe`）+ `load_selected_layer_ptq_params` → 开启量化、置 `eval_mode=True` → 量化块塞回层列表**并放回该层原本所在的卡** → 先做一次 `generate` 冒烟。出现 cpu / disk offload 直接报错（offload 会破坏量化正确性）。
- **`--bf16` 跑基线**：跳过量化，用**同一套 harness** 跑原始模型。两者都要跑，最后报告差值。
- **`--device_map auto`** 走多卡，省略则单卡 npu:0；输入经 `input_device()` 路由到 embedding 所在卡。
- **attention 量化默认被拦截**：脚本禁 `--quant_target attn-linear` / `attn-cache`，确需评完整 attention 量化对象要显式加 `--allow_attn_quant`，且先按约束 1 与用户确认取舍。

### 3. LiveCodeBench（可选）

代码生成不走 lm-eval：`scripts/livecodebench_eval.py` 生成官方 `custom_evaluator` JSON，再由 `python -m lcb_runner.runner.custom_evaluator` 算 hidden-tests pass@1。完整命令与对齐清单见 [references/livecodebench.md](references/livecodebench.md)。

### 4. 报告结果

- 对比量化与 bf16 的**主指标**（按「任务选择与资源成本」表的默认主指标）。
- 汇总差值：`python scripts/compare_results.py --bf16 <bf16.log> --quant <quant.log> --metric exact_match,none --sample_len <N> --limited`。
- 必须写明**样本量**与（LiveCodeBench 的）`release_version`。
- 用了 `--limit` 就写小样本冒烟，不写官方分数；差异落在统计噪声内表述为「无可见损失」，不表述为「量化更好」。

---

## 边界规则

- 只评测已产出的 ptq 参数，不跑量化、不改量化方案、不下 PPL `delta` 达标结论。
- 不默认修改 amct 源码；wrapper 的 KV cache 缺失属适配问题，交 `$model-adapter`。
- 量化参数含 attn-linear 时**必须先与用户确认取舍**（约束 1），不静默降级为只量化 MLP。
- 不替用户选 benchmark；先给候选与成本，由用户决定。
- 运行产物只落用户指定的工作目录，`scripts/` 只存代码（约束 9）。
- 引用仓库内代码、脚本和文档路径时，一律使用仓库相对路径，例如 `amct_pytorch/common/models/llm/...`、`.agents/docs/...`；不要写绝对路径或个人机器路径。

---

## 文件

| 文件 | 用途 |
|---|---|
| `references/constraints.md` | 9 条关键约束（attention / KV cache、离线数据、PYTHONPATH、OOM、多卡 hook 等） |
| `references/env-and-portability.md` | 依赖边界、可配置 env、环境模板、新机器部署步骤 |
| `references/livecodebench.md` | LiveCodeBench 生成 + 官方评分 + 榜单对齐清单 |
| `scripts/config.py` | env 解析（`AMCT_HOME` / `BENCH_ASSETS` / `HF_ENDPOINT`）+ 加载 `datasets.json`；提供 `local_dataset_map()`（旧名 `local_parquet_map()` 为兼容别名） |
| `scripts/utils.py` | 数据读取（parquet / jsonl）、依赖检查、LiveCodeBench 字段校验、attention 量化保护、结果日志解析 |
| `scripts/datasets.json` | benchmark 注册表（repo_id、subdir、split → parquet/jsonl/glob），prefetch 与离线 shim 的单一数据源 |
| `scripts/self_check.py` | 不加载模型、不跑 NPU 的轻量自检，验证脚本接线 |
| `scripts/check_env.py` | 检查运行依赖与 lm-eval 版本；`--livecodebench_score` 加查官方评分 runner |
| `scripts/prefetch_data.py` | `snapshot_download` 把数据集下到 `$BENCH_ASSETS` |
| `scripts/bench_offline.py` | monkeypatch `load_dataset` → fastparquet 读本地 parquet、pandas 读本地 jsonl |
| `scripts/assemble_and_eval.py` | 装配 fake-quant 模型（KV-safe 目标、多卡保留 hook）+ HFLM 评测 + `--bf16` 基线 |
| `scripts/livecodebench_eval.py` | 装配后生成代码，输出 LiveCodeBench 官方 `custom_evaluator` JSON |
| `scripts/compare_results.py` | 从 bf16 / quant 日志或 JSON 抽指标并输出差值 |

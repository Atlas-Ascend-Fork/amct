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
| **任何一次评测前（必读）** | [references/constraints.md](references/constraints.md) —— 13 条踩坑约束，编号稳定 |
| 换机器 / 配 env / 装依赖 | [references/env-and-portability.md](references/env-and-portability.md) |
| 要跑代码生成任务 | [references/livecodebench.md](references/livecodebench.md) |

---

## 覆盖度状态

下表只记录**功能是否打通**（任务能否接入、脚本链路是否跑通）：这与硬件无关，换卡不必重新验证。表内唯一一条显存相关状态（35B MoE 长 prompt OOM）是特定配置下的现象，量级与调优手段见 [references/constraints.md](references/constraints.md) 约束 8；速度量级见约束 6。这两条依赖 2× Ascend 910（每卡 64GB HBM），换硬件后按当机实测，不要当普适结论。

| 场景 | 状态 |
|---|---|
| MMLU-Pro，8B W8A8 仅 mlp，单卡 | ✅ 已验证 |
| 35B MoE 双卡 bf16（跨卡 generate + HFLM + 打分） | ✅ 已验证 |
| 35B MoE 双卡 W8A8 装配 + 跨卡 generate 冒烟 | ✅ 已验证（需约束 7 的 hook 修复） |
| 35B MoE 双卡 W8A8 + 长 prompt 评测 | ⚠️ npu:0 OOM（见约束 8，属显存调优） |
| MATH500，lm-eval `hendrycks_math500` / 别名 `math500` | ✅ 已验证（离线读仓库实际的 `test.jsonl` 并出分；全量耗时按机器实测） |
| LiveCodeBench code-generation-lite，官方 custom_evaluator hidden-tests pass@1 | ⚠️ 已接入生成官方 custom 输出；评分需安装并调用 `lcb_runner` |
| ARC-Challenge / ARC-Easy / BoolQ / COPA / PIQA / SciQ | ✅ 已验证（bf16 与量化各 6 任务小样本出分，主指标键与下表一致） |

---

## 任务选择与资源成本

**不要默认替用户决定 benchmark**：先按用户的资源、时间预算、能力覆盖需求给候选，让用户选。

| 成本档 | 任务 | 默认主指标（lm-eval 结果键） | 适用场景 | 注意 |
|---|---|---|---|---|
| 低 | `arc_challenge` | `acc_norm,none` | 首选轻量下游回归；科学 / 常识推理，比 ARC-Easy 更敏感 | 同时保留 `acc,none`；适合先 `--limit` 冒烟 |
| 低 | `arc_easy` | `acc_norm,none` | 更便宜的科学问答补充 | 难度低，大模型区分度有限 |
| 低 | `boolq` | `acc,none` | 阅读理解 yes/no 回归 | 二分类，随机基线高，小样本波动大 |
| 低 | `copa` | `acc,none` | 常识因果二选一 smoke | 样本少，只能作补充，不要单独下结论 |
| 低 | `piqa` | `acc_norm,none` | 物理常识 / 日常操作推理 | 比 COPA 更稳，适合低成本补充 |
| 低 | `sciq` | `acc_norm,none` | 科学知识多选补充 | 与 ARC 能力有重叠，难度较低 |
| 高 | `mmlu_pro` | `exact_match,custom-extract` | 覆盖面更广的通用知识 / 推理 | 题量大，建议先 `--limit`；filter 名是 `custom-extract` 不是 `none` |
| 高 | `math500` | `exact_match,none` | 数学推理 | 长生成，需更大 `--max_gen_toks`，耗时高 |
| 高 | `livecodebench` / `livecodebench_release_v6` | `pass@1` | 代码生成 hidden-tests pass@1 | 最耗时之一，评分依赖 `lcb_runner` |

- lm-eval 的结果键格式固定为 **`<指标名>,<filter名>`**：task YAML 声明了 `filter_list` 就用其中每条的 `name`（mmlu_pro 是 `custom-extract`），没声明则默认 `none`。上表已按 lm-eval 0.4.12 实测键名填写，`scripts/self_check.py` 会拿已安装的 lm-eval task index 反查校验，升版本后跑一次自检即可发现漂移。
- `livecodebench` 的 `pass@1` 由官方 `lcb_runner` 产出，不经 lm-eval，故无 `,filter` 后缀。**生成日志里没有分数**：`livecodebench_eval.py` 只打 `[lcb-gen]`（样本数与输出路径），`pass@1` 要拿 `custom_evaluator` 写出的 `*_eval.json` 去 `compare_results.py` 对比（见 [references/livecodebench.md](references/livecodebench.md) §2）。
- 低资源任务的汇总指标固定为：`arc_challenge` / `arc_easy` / `piqa` / `sciq` 看 `acc_norm,none`，`boolq` / `copa` 看 `acc,none`。做 bf16 vs quant 对比时**必须同任务同指标**，不要把 `acc` 与 `acc_norm` 混用；其它指标可留在明细里。
- 资源有限时优先建议：`WikiText2 PPL + arc_challenge`；有余量再加 `piqa` / `boolq` / `sciq` / `arc_easy` / `copa`；要综合结论再升级 `mmlu_pro`；要数学或代码能力再跑 `math500` / `livecodebench`。
- `--limit` 的结果只能称**小样本回归 / 冒烟**，不能称官方分数。

---

## 工作流

> 下文 `scripts/` 均相对本 skill 目录（`.agents/skills/quant-tools/quant-benchmark-eval/`）；按约束 9 要在独立工作目录运行，实际调用时写完整路径。

### 1. 环境与数据准备（一次）

```bash
python scripts/self_check.py                                  # 轻量自检：不加载模型、不占 NPU
python scripts/self_check.py --remote                         # 加验 datasets.json 与各数据集仓真实文件列表是否一致（只读元数据，需联网；按 any_of/all_of 分别判定）
python scripts/check_env.py                                   # 依赖 + lm-eval 版本检查
python scripts/prefetch_data.py --task <task>                 # 预下载到 $BENCH_ASSETS/<subdir>/
```

`<task>` 取上表任务名（`livecodebench` 只下单 release 的 `test6.jsonl`，累计集用 `livecodebench_release_v6`）。env 变量、依赖版本、离线读取机制见 [references/env-and-portability.md](references/env-and-portability.md)。

`prefetch_data.py` 会逐个 split 打印实际命中的文件，**任一 split 没命中就以非 0 退出并给出排查方向**（网络被拦 / `datasets.json` 的 glob 与仓库布局不符）——不要带着 MISSING 往下跑，否则评测阶段只会看到一句误导性的 `run prefetch_data.py first`。

`datasets.json` 里一个 split 的多条路径有两种语义，**不能混**：

- `{"any_of": [...]}` —— 同一份数据的不同**候选布局**（MATH-500 既可能发 `test.jsonl` 也可能发 `data/test-*.parquet`）：按声明顺序取**第一个命中**的候选，其余忽略。缓存里同时存在两种布局（上游迁移 / 旧缓存残留）时也只读一份，样本不会翻倍。
- `{"all_of": [...]}`（或裸字符串 / 裸列表）—— 必须**全部合并**的分片（LiveCodeBench 累积集 `test.jsonl` … `test6.jsonl`）：缺任一分片即以 PARTIAL 非 0 退出，避免静默少跑样本、指标分母错。

这两种语义在 `prefetch_data.py`（本地文件）、`config.resolve_split_files()`（离线读取）与 `self_check.py --remote`（上游文件列表）三处**判定口径一致**：`any_of` 有一条候选命中即通过，`all_of` 要求每条分片都命中——**部分命中同样算不通过**。所以 `all_of` 缺分片在三处都硬失败：prefetch 以 PARTIAL 非 0 退出，离线读取抛 `FileNotFoundError` 点名缺哪几片（绝不拿子集凑数：LiveCodeBench 的分片是累积集，少一片就等于把 pass@1 的分母悄悄改小，产出的却是一张看起来正常的结果表），`--remote` 报 `datasets.json does not match upstream repos`（此时是注册表 glob 与上游布局真的对不上——上游改名 / 少发分片，不是声明形态没被识别）。

离线读取的这次失败是**延迟到真正读该 split 时**才抛的（`config.UnresolvedSplit`）：`local_dataset_map()` 会解析注册表里全部任务，若在建表时就抛，某个 benchmark 的缓存不完整会连带阻塞另一个健康 benchmark 的评测。存下错误、由读它的人抛，既保持 fail-loud，又只影响真正在用的数据。

**加一个 benchmark 只需要改 `datasets.json`**（`bench_offline.py` 的扩展约定），不用动 `self_check.py`：默认自检的离线回归对**没有手写桩**的 repo 会按其声明自动派生一份上游文件列表（`_derived_listing()`），因此新任务不会因为「桩文件列表缺这个 repo_id」而让默认自检变红。手写桩（`_FAKE_UPSTREAM`）只保留给当前 10 个任务对应的 7 个仓，作用是钉住真实布局（MATH-500 只发 parquet、LiveCodeBench 6 个累积分片）。派生桩与声明天然自洽，只能覆盖 `any_of`/`all_of` 匹配逻辑与「声明的 glob 是否被 `allow_patterns` 覆盖」（`prefetch` 按 `allow_patterns` 下载，被过滤掉的 glob 永远落不到本地）；**新任务的 glob 是否真的贴合上游布局，仍须跑一次 `--remote` 兜底**。自检会打印一行 `note: no hand-written upstream stub for <repo_id>` 提示这一点；桩里出现注册表已不再引用的 repo_id 会直接报错（防止对照项静默失效）。派生是**按 repo_id 聚合该仓下全部任务的声明**再取并集，不是只取字典序第一个任务：同一 repo_id 挂多个任务（`datasets.json` 里 livecodebench / livecodebench_release_v6 已经是这个形态，只是两者都有手写桩才绕过派生分支）时，只派生第一个会让第二个任务的分片不在列表里，默认自检就会把**桩缺口**报成「注册表误报」。自检第 5 步专门有一条「两个任务共用一个未建桩 repo_id」的对照，含摘掉其中一个分片必须点名到该任务的反向验证。

### 2. 装配 + 评测

```bash
# 8B dense，单卡
python scripts/assemble_and_eval.py --model <model_path> --model_name qwen3 \
  --param_dir <ptq_params/mlp> --task mmlu_pro --limit 10 --max_gen_toks 512 \
  --output_dir <out_dir> --algos lwc,lac

# 35B MoE，双卡（先设 ASCEND_RT_VISIBLE_DEVICES=0,1）
python scripts/assemble_and_eval.py --model <model_path> --model_name qwen3_5_moe \
  --param_dir <ptq_params/moe> --task mmlu_pro --device_map auto --limit 10 --algos lwc,lac

# 低资源任务：换 --task arc_challenge / arc_easy / boolq / copa / piqa / sciq，--max_gen_toks 64
# MATH500：--task math500（别名，映射到 lm-eval hendrycks_math500）--max_gen_toks 1024
#         也可直接传 hendrycks_math500 / minerva_math500 换 prompt 口径
```

- **`--task` 先做离线预检**：加载模型**之前**用 lm-eval task index 把任务（含别名 / group 展开）解析成 `(dataset_path, dataset_name)`，逐个对照 `datasets.json`。拼错或未注册的任务立刻报错并列出已有 repo，不会等到模型装好、每层量化块都塞回去之后才以 `FSTimeoutError` 炸掉（那时报错里根本不提 `--task`）。通过后打一行 `[task] <name> reads [...] from $BENCH_ASSETS`，便于事后核对评的是哪份数据。详见约束 2。
- **装配流程**：`load_float()`（支持 `device_map`）加载模型 → 逐层 `build_quant_block`（KV-safe 目标：dense 用 `mlp`，MoE 用 `moe`）+ `load_selected_layer_ptq_params` → 开启量化、置 `eval_mode=True` → 量化块塞回层列表**并放回该层原本所在的卡** → 先做一次 `generate` 冒烟。出现 cpu / disk offload 直接报错（offload 会破坏量化正确性）。
- **`--algos` 必须与 ptq 参数训练时的算法一致**（默认 `lwc,lac`；amct 已注册 `autoround / flatquant / gptq / lac / let / lwc`）。算法决定了量化块里构造出哪些子模块，不一致时加载参数会 `KeyError`；脚本在装配前先按 `ALGO_REGISTRY` 校验键名。详见约束 10。
- **ptq 参数必须真的加载上**：脚本逐层统计 `loaded/missing`，一个都没加载上直接报错（否则会静默回落直转，评的就不是用户的 PTQ 结果）；部分层缺参数打 `[warn]`，结论里要写明是混合对象。详见约束 11。
- **`--bf16` 跑基线**：跳过量化，用**同一套 harness** 跑原始模型。两者都要跑，最后报告差值。
- **`--device_map auto`** 走多卡，省略则单卡 npu:0；输入经 `input_device()` 路由到 embedding 所在卡——取不到 embedding 设备时退回 `model.device`，两者都拿不到就报错并保留原因，**不会兜底硬写 `npu:0`**（多卡下把 input_ids 送到没有权重的卡，只会在 forward 深处炸成跨设备错误）。
- **bit config 先验存在再装模型**：`--bit_config` 默认取 `$AMCT_HOME/amct_pytorch/configs/w8a8.yaml`，`require_bit_config()` 在加载任何权重之前确认该文件真在盘上。否则 AMCT_HOME 指错 / 指向不含该配置的 checkout 时，失败会发生在 amct 流水线内部、float 模型已经上卡之后，报错里也看不出是这个默认路径的问题。`--bf16` 不读 bit config，不受影响。
- **attention 量化默认被拦截**：脚本禁 `--quant_target attn-linear` / `attn-cache`，确需评完整 attention 量化对象要显式加 `--allow_attn_quant`，且先按约束 1 与用户确认取舍。
- **`--trust_remote_code` 默认关闭**（与 amct CLI 一致）：加载非原生支持架构的模型目录需要它时才显式加，加了就会执行该目录下的自定义建模代码——只对可信目录开，且必须先与用户说明。未开启时脚本会打 `[warn]` 提示可能加载失败，详见约束 12。

### 3. LiveCodeBench（可选）

代码生成不走 lm-eval：`scripts/livecodebench_eval.py` 生成官方 `custom_evaluator` JSON，再由 `python -m lcb_runner.runner.custom_evaluator` 算 hidden-tests pass@1。完整命令与对齐清单见 [references/livecodebench.md](references/livecodebench.md)。

### 4. 报告结果

- 对比量化与 bf16 的**主指标**（按「任务选择与资源成本」表的默认主指标）。
- 汇总差值：`python scripts/compare_results.py --bf16 <bf16.log> --quant <quant.log> --metric exact_match --sample_len <N> --limited`。`--metric` 传指标名（`exact_match` / `acc_norm` / `acc` / `pass@1`）或完整结果键（`exact_match,custom-extract`）都能匹配；省略则自动挑选，`_stderr` 键与子任务键会被排除，优先取组聚合键。lm-eval 任务喂运行日志；LiveCodeBench 喂 `custom_evaluator` 写出的 `*_eval.json`（`--metric pass@1`），**不要喂生成日志**——它没有分数，脚本会直接报错而不是给你一个假的 `delta 0.0`。
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
| `references/constraints.md` | 13 条关键约束（attention / KV cache、离线数据、PYTHONPATH、OOM、多卡 hook、`--algos` 与 ptq 参数匹配、`--trust_remote_code`、transformers dtype 参数名等） |
| `references/env-and-portability.md` | 依赖边界、可配置 env、环境模板、新机器部署步骤 |
| `references/livecodebench.md` | LiveCodeBench 生成 + 官方评分 + 榜单对齐清单 |
| `scripts/config.py` | env 解析（`AMCT_HOME` / `BENCH_ASSETS` / `HF_ENDPOINT`）+ 加载 `datasets.json`；提供 `local_dataset_map()`（键为 `(repo_id, dataset_name)`，即 lm-eval 传给 `load_dataset` 的那对值，**不是任务名**；两个任务共用同一对键却声明不同本地数据时直接抛错，不静默覆盖）与 `split_declaration()` / `resolve_split_files()`（`any_of` 取首个命中布局、`all_of` 合并全部分片，**缺片即抛 `FileNotFoundError` 点名缺哪几片**，不静默按子集算分）、`require_amct_home()` / `require_bit_config()`（默认位宽配置存在性预检） |
| `scripts/utils.py` | 数据读取（parquet / jsonl）、依赖检查、LiveCodeBench 字段校验、attention 量化保护、结果日志解析；`check_task_offline_ready()` 用 lm-eval task index 把 `--task` 解析成 `(dataset_path, dataset_name)` 并对照 `datasets.json`，做加载模型前的离线预检；「任务名 lm-eval 不认」（`TaskIndexMiss`，拼写 / 别名问题）与「任务存在但不声明 dataset」（`TaskHasNoDataset`，加注册表条目也没用）分开报错，不洗成同一句「检查拼写」 |
| `scripts/datasets.json` | benchmark 注册表（repo_id、subdir、split → parquet/jsonl/glob，或 `{"any_of": [...]}` / `{"all_of": [...]}`），prefetch 与离线 shim 的单一数据源 |
| `scripts/self_check.py` | 不加载模型、不跑 NPU 的轻量自检：脚本接线、注册表声明、`any_of`/`all_of` 布局语义（临时目录实测，防样本翻倍）、`--remote` 的 glob 匹配语义（注入假文件列表离线回归，防把正确注册表报成不匹配；未建桩的 repo 按声明自动派生列表，**加任务只改 `datasets.json` 不会让默认自检变红**）、离线 shim 是否真拦住 lm-eval 的 `load_dataset` 调用形态（注入临时 jsonl 实跑一次，防评测静默回落到网络 / HF cache）、已预下载任务的 split 是否真能解析、SKILL.md 指标列 vs lm-eval task index、文档点名的代码标识符是否还存在（`name()` / 裸名 / `mod.name` 三种写法都查；amct 侧名字对 `$AMCT_HOME` 源码核，第三方名走显式允许表且**表里失效的条目要报错**）、shim 键冲突（同 repo 不同 `dataset_name` 必须两键并存、同键不同数据必须报错、同键同声明可折叠）、注册表任务 vs lm-eval task index 的离线预检（含未知任务名 / 未注册数据集 / 任务不声明 dataset 三条反向对照，防预检被改成永真）、`all_of` 缺片必须拒绝且只影响该任务（`_check_partial_split_deferred`）、shim 只读被请求的 split、`input_device()` 不许猜设备、bit config 存在性、离线 shim 透传失败是否可诊断；`--remote` 再加验注册表 vs 数据集仓真实文件列表 |
| `scripts/check_env.py` | 检查运行依赖与 lm-eval 版本；`--livecodebench_score` 加查官方评分 runner |
| `scripts/prefetch_data.py` | `snapshot_download` 把数据集下到 `$BENCH_ASSETS`，并校验每个 split 是否真命中文件（`any_of` 未命中、`all_of` 缺分片即非 0 退出） |
| `scripts/bench_offline.py` | monkeypatch `load_dataset` → fastparquet 读本地 parquet、pandas 读本地 jsonl；**只校验并读取调用方要的那个 split**（请求 `test` 不会因为没预取的 `validation` 报错，也不多读没人评分的数据）；不在注册表里的请求透传给真实 loader，透传失败时把「没注册哪条 + 已注册哪些」链进报错（原始 loader 异常保留在 `__cause__`） |
| `scripts/assemble_and_eval.py` | 装配 fake-quant 模型（KV-safe 目标、`--algos` 校验、ptq 参数落地校验、多卡保留 hook）+ HFLM 评测 + `--bf16` 基线；评测前先跑 `--task` 离线预检（`--smoke` 只验装配链路、不读数据，故跳过） |
| `scripts/livecodebench_eval.py` | 装配后生成代码，输出 LiveCodeBench 官方 `custom_evaluator` JSON（同样支持 `--algos` / `--trust_remote_code`）；只打 `[lcb-gen]`，分数由官方 evaluator 产出 |
| `scripts/compare_results.py` | 从 bf16 / quant 日志或 JSON（含官方 lcb_runner `*_eval.json`）抽指标并输出差值；排除 `_stderr` 与 `sample_len` 等非指标键；日志没有 `[RESULT]` 行时给出可诊断报错 |

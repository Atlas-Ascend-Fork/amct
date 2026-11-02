---
name: quant-benchmark-eval
description: 在 Ascend NPU 上评测 amct 量化后模型的生成式下游准确率（mmlu_pro / math500 / livecodebench 等），并与 bf16 基线同口径对比出差值。触发场景：需要 WikiText2 PPL 之外的下游准确率、用户点名某个下游 benchmark、交付前要下游证据。不适用于：跑量化命令、产 ptq 参数、判达标 / 判 PTQ 收益、补 attention wrapper 的 KV cache（属模型适配）、导出部署权重与交付文档。
---

# quant-benchmark-eval

## 1. 概述

把量化后模型**装配**成可 `generate` 的对象，交给 lm-eval-harness（或 LiveCodeBench 官方 runner）打分，与 bf16 基线同口径对比出差值。

## 2. 评测口径

**评测对象**：量化侧是装配出的伪量化模型（float 权重 + 量化块，量化效果模拟在算子上），bf16 侧是同一个 float 模型——两侧之差即量化本身。量化块有两种装法：`--param_dir` 指向的 PTQ 训练产物灌参数，或 `--direct` 的直转（不读参数，权重 / 激活 scale 前向当场算）。

**评测方法**：同一模型在量化前后各执行一次评测，计算两侧准确率差值。差值成立的前提是两次评测在下述三点上完全一致：**评测任务**（选哪个 benchmark；LiveCodeBench 还含题库版本 `release_version`，见 [references/livecodebench.md](references/livecodebench.md) §1）、**样本量**（试跑子集还是全量）、**生成 token 上限**（代码生成与数学推理任务需要更高的上限）。任一点不一致，差值即无法归因于量化。评测侧就是这几点，量化侧那几项与其余设置项逐项取值见 §4.2「与量化侧共享的参数」表。prompt 模板无对应设置项，两侧共用同一模板；与外部榜单横比时，prompt 模板与生成 token 上限都可能是未对齐项（[references/livecodebench.md](references/livecodebench.md) §3）。

**评测结果**：每侧一个准确率，两侧相减得 delta。

## 3. 任务选型

| 成本档 | 任务 | 默认主指标 | 适用场景 | 注意 |
|---|---|---|---|---|
| 低 | `arc_challenge` | `acc_norm,none` | 首选轻量下游回归；科学 / 常识推理，比 ARC-Easy 更敏感 | 同时保留 `acc,none` |
| 低 | `arc_easy` | `acc_norm,none` | 更便宜的科学问答补充 | 难度低，大模型区分度有限 |
| 低 | `boolq` | `acc,none` | 阅读理解 yes/no 回归 | 二分类，随机基线高，小样本波动大 |
| 低 | `copa` | `acc,none` | 常识因果二选一 smoke | 样本少，只能作补充，不要单独下结论 |
| 低 | `piqa` | `acc_norm,none` | 物理常识 / 日常操作推理 | 比 COPA 更稳，适合低成本补充 |
| 低 | `sciq` | `acc_norm,none` | 科学知识多选补充 | 与 ARC 能力有重叠，难度较低 |
| 高 | `mmlu_pro` | `exact_match,custom-extract` | 覆盖面更广的通用知识 / 推理 | 题量大 |
| 高 | `math500` | `exact_match,none` | 数学推理 | 长生成任务 |
| 高 | `livecodebench_release_v6` | `pass@1` | 代码生成 hidden-tests pass@1 | 入口是 `livecodebench_eval.py`，分数由官方 `lcb_runner` 写进 `*_eval.json`（[references/livecodebench.md](references/livecodebench.md)） |

「低 / 高」按要不要生成分档：低档全是 `multiple_choice`（不生成），高档要生成——lm-eval 侧 `output_type` 是 `generate_until`，LiveCodeBench 走官方 runner。

资源有限时的优先级：先低档挑一个，有余量再加其余低档；要综合结论 / 数学 / 代码能力再升高档，按上表「适用场景」列挑。

## 4. 工作流

> 下文 `<skill>` 指本 skill 目录（`.agents/skills/quant-tools/quant-benchmark-eval/`）的绝对路径，命令里的 `<skill>/scripts/...` 照原样展开后再执行。

### 4.1 准备

第一次在一个环境上跑：依赖 / 环境变量 / 部署步骤按 [references/env-and-portability.md](references/env-and-portability.md) 走一遍。

```bash
python <skill>/scripts/check_env.py                     # 判依赖装齐、lm-eval 钉住版本、AMCT_HOME 能否解析；成功时打出 [env] 行（BENCH_ASSETS、transformers 版本与本次用的权重 dtype 参数名）
python <skill>/scripts/prefetch_data.py --task <task>   # 预下载到 $BENCH_ASSETS/<subdir>/
```

`<task>` 取 §3 表里的任务名。

报告要写明本次用的是哪一份 amct，按下面三条查实际生效的那份。

```bash
python -c "import amct_pytorch; print(amct_pytorch.__file__)"
python -c "import importlib.metadata as m; print(m.version('amct_pytorch'))"
git -C "$AMCT_HOME" rev-parse HEAD
```

装的是 wheel 用第二条的版本号；源码 checkout 用第三条的 commit。

**任一步骤红 / 打告警就停下**，症状与处理查 §5。

### 4.2 装配与评测

**装配**：脚本加载 float 模型，给 ptq 参数目录里有参数的那些层装上量化块并灌入参数，装配完才是一个可 `generate` 的对象；进度看 `[assemble]` 行。`--direct` 下没有参数目录：全部 decoder 层都装量化块、不灌参数、不构造算法模块，被评对象是直转（`[direct]` 行），层范围看 `[quant-scope]` 行。

**强制停点**——跑之前必须已与用户确认，下面的命令模板不得绕过：

- **停点 A · 评哪个 benchmark**：给候选 + 成本档（§3），由用户选。
- **停点 B · attention 量化的代价与目标**：判据是 `ls <ptq_params_dir> | grep -E '_(self|linear)_attn\.pt$'` 有没有输出。有输出时默认就评 attention，与用户确认两件事。一是**代价**（约束 1，慢多少以冒烟实测为准，可能要缩 `--limit` 或换轻量任务）。二是**哪个 attention 目标**：脚本按这个目录的路径分量认目标，认不出时取 `attn-linear` 并打一行 `[warn]`；该行出现就问清这批参数是哪个目标产的，是 `attn-cache` 则显式传 `--quant_target attn-cache`。`--direct` 下没有目录可判：目标由用户显式给（`--quant_target` 必填），点名了 attn 目标同样撞放行闸，按这两点确认后再加 `--allow_attn_quant`。

**命令模板**（dense 单卡 + `mmlu_pro` 全量，两侧参数逐字对应）：

```bash
# 量化侧：被评对象由 --param_dir 推导
python <skill>/scripts/assemble_and_eval.py \
  --model <model_path> --model_name qwen3 --device_map '' \
  --param_dir <ptq_params_dir> --algos lwc,lac --allow_attn_quant \
  --task mmlu_pro --max_gen_toks 512 \
  --output_dir <run_dir> --output_path <run_dir>/quant_mmlu_pro.json \
  2>&1 | tee <run_dir>/quant_mmlu_pro.log

# bf16 基线：除 --bf16、产物文件名（及随之无意义的 --param_dir / --algos / --allow_attn_quant）外逐字相同
python <skill>/scripts/assemble_and_eval.py \
  --model <model_path> --model_name qwen3 --device_map '' --bf16 \
  --task mmlu_pro --max_gen_toks 512 \
  --output_dir <run_dir> --output_path <run_dir>/bf16_mmlu_pro.json \
  2>&1 | tee <run_dir>/bf16_mmlu_pro.log
```

正式跑之前先冒烟，两级递进：

- `--smoke`：只验装配与能否 `generate`，不需要预取数据。
- `--limit 10`：验整条链路并出分数。`--limit 10` 是**冒烟值**，出正式分数时删掉。

**变体**：

| 方向 | 改什么 |
|---|---|
| 换模型形态 | MoE 多卡：`--model_name` 用 amct 的 MoE pipeline 键（`$quant-run` 的命令记录里有）、`--device_map auto`，卡号用 `plan_device.py` 打出的填（每卡显存小于预算时按 env-and-portability.md §6「定卡数」放大） |
| 换任务 | 低成本档**无需配 `--max_gen_toks`**；`mmlu_pro` 用 `512`；`math500` 配 `1024`（别名映射到 lm-eval 的 `hendrycks_math500`，也可直接传 `hendrycks_math500` / `minerva_math500` 换 prompt 口径） |
| 评直转 | 加 `--direct` + 显式 `--quant_target`，去掉 `--param_dir` / `--algos`；bf16 侧命令逐字不变 |
| 评 attention 量化 | **默认即评**（推导与放行闸见停点 B、约束 1）。要窄口径（只评其中一部分）才显式传 `--quant_target`（amct 平级目标：`mlp` / `moe` / `attn-linear` / `attn-cache`） |
| 非原生架构 | 加 `--trust_remote_code`（约束 5） |
| 代码生成 | 换 `scripts/livecodebench_eval.py`（完整命令与取分步骤见 [references/livecodebench.md](references/livecodebench.md)）；`--max_gen_toks` 该传多少以 [references/livecodebench.md](references/livecodebench.md) §1 的命令模板为准 |

**与量化侧共享的参数**（每项都要跟量化时一致；接手别人的评测前先核）：

| 参数 | 从哪来 |
|---|---|
| `--model` / `--model_name` | 用户的模型目录；`--model_name` 是 amct pipeline 键，跟量化时用的同一个（`$quant-run` 的命令记录里有） |
| `--param_dir` / `--quant_target` | 评 PTQ 结果：`--param_dir` 指 quant-run 的 PTQ 训练产物（`layer_<idx>_<unit>.pt`），`--quant_target` 缺省由它推导（unit 类别看文件名，attention 目标看路径分量，停点 B），量化哪些层也由它决定（约束 4）。评直转：无参数可读，传 `--direct` + 显式 `--quant_target`（amct 平级目标：`mlp` / `moe` / `attn-linear` / `attn-cache`），不传 `--param_dir`。显式传 `--quant_target` 时两侧必须一致 |
| `--algos` | PTQ 训练时用的算法键（`$quant-run` 的命令记录里有）；两侧必须一致（约束 3）。`--direct` 下不传（直转不构造算法模块） |
| `--bit_config` | 量化时那份位宽 yaml（`$quant-run` 的命令记录里有），抄不到就问用户；不传时用 amct 的默认位宽 yaml（默认值解析与缺文件处置见 env-and-portability.md §4） |
| `--quant_dtype` | 量化数据类型（`int` / `mxfp` / `hifp` / `fp`），抄 ptq 参数训练时那个值；缺省 `int` |
| `--is_per_tensor` | 激活裁剪用 per-tensor 统计；ptq 训练时带了这里就带 |
| `--seq_len` | 量化装配用的长度，取值抄 `$quant-run` 的命令记录；本脚本另有自己的默认值，bf16 侧不读取、两侧之间无需配对 |
| `granularity` | 无对应 flag，钉死 `block` |

**必知参数**（完整清单与默认值看 `--help`）：

| 参数 | 用途 |
|---|---|
| `--device_map` | `auto` 走多卡、`''` 单卡 npu:0；省略则按 `config.json` 估算权重体积推断。启动前先 `python <skill>/scripts/plan_device.py --model <model_path>` 问出来 |
| `--trust_remote_code` | 加载带自定义建模代码的模型目录，默认关（约束 5） |
| `--allow_attn_quant` | attention 量化的**放行闸**：加不加见停点 B（约束 1） |
| `--direct` | 评直转（无 ptq 参数）：全层装量化块、不构造算法模块、scale 前向当场算。必须配显式 `--quant_target`；与 `--param_dir` / `--algos` / `--bf16` 互斥 |

**硬失败前置**：`--task` 可读性、ptq 参数与 `--quant_target` 的对应、单一 attention 目标、offload / 设备、`--algos` 键名、`--param_dir` 缺失、`--direct` 的搭配、`--limit` 取值，脚本都在加载权重 / 装配之前拦住，不用自己预检；症状与处理见 §5。

### 4.3 产物与报告

产物都指到同一个运行目录（命令模板里的 `<run_dir>`），结果与 tee 出的运行日志同处。必填项：`--output_dir`（两个入口都必填，沿用上游产 ptq 参数的同名目录约定；评测路径不往它写文件，必填只为两个入口的参数面不分叉）、`assemble_and_eval.py` 的 `--output_path`、`livecodebench_eval.py` 的 `--output_json`。

**产物清单**：

| 产物 | 内容 |
|---|---|
| `--output_path` 的 JSON | lm-eval 结果字典 + 样本量 + 本次运行参数（task / limit / max_gen_toks / quant_target / algos / device_map …），`compare_results.py` 直接读它 |
| 运行日志 | 见下面的日志行含义 |
| LCB 的 `*_eval.json` | 官方 evaluator 写出的分数，`pass@1` 只在这里 |
| LCB 的 `*_run_meta.json` | `livecodebench_eval.py` 写在 `--output_json` 旁边的本次运行参数（`release_version` / `max_gen_toks` / 样本量 / 量化侧各项 / `quant_scope`）；官方 `*_eval.json` 只有分数，`compare_results.py` 按文件名约定找到这一份来核对两侧可比 |

| 日志行 | 含义 |
|---|---|
| `[run]` | 本次实际生效的参数（启动即打，用来核对两侧是否可比） |
| `[task]` | `--task` 解析成哪些 `(dataset_path, dataset_name)`、从 `$BENCH_ASSETS` 读 |
| `[data]` | LiveCodeBench 侧解析到的本地文件数 |
| `[device]` | 权重估算、推断出的 device_map 与该开的卡号（两个入口都打）；模型实际跨了哪些卡（仅装配入口打） |
| `[bf16]` | 基线侧加载原始模型（不装配量化块）；装配与 LiveCodeBench 入口都打，看到这行说明跑的是 `--bf16` 侧 |
| `[assemble]` | 逐层装配进度、algos / param_dir、量化目标 |
| `[direct]` | 本次是直转，分数描述直转而非 PTQ checkpoint（两个入口都打） |
| `[smoke]` | 装配后的一次短 `generate`，打出实际生效的 `use_cache=`（`--smoke` 到此即返回，不进 lm-eval、不出分数） |
| `[ptq-pre]` | 装配前 `--param_dir` 静态预检，两个入口都打：命中的层索引范围 / unit 种类，覆盖不满全模型时点名未覆盖的层索引（覆盖段里的洞、attention 目标认不出另有 `[warn]`） |
| `[quant-scope]` | 本次实际量化的层范围，两个入口都打：范围从哪来（PTQ 是 `--param_dir` 覆盖的层，直转是全部 decoder 层）、覆盖哪几层、共几层、覆盖段里的洞；覆盖不满全模型时点名其余层保持 bf16 |
| `[ptq]` | 装配后每个目标的 `loaded=N missing=0` 逐层统计，两个入口都打；**只在全部加载上时打**，有缺失打的是 `[warn]`，一个都没加载上直接失败、本行不出现 |
| `[eval]` | 交给 lm-eval 的 task / limit / max_gen_toks |
| `[generate]` | LiveCodeBench 侧生成开始：dataset_task / release_version / 待生成样本数 / max_gen_toks |
| `[sample]` | LiveCodeBench 侧逐条生成进度 `idx/总数` 与该条生成的字符数 |
| `[lcb-gen]` | LiveCodeBench 侧写出的 custom 输出：样本条数 + `--output_json` / `--raw_output_jsonl` / `*_run_meta.json` 路径（**不含分数**） |
| `[next]` | LiveCodeBench 侧接着要跑的官方评分命令，以及它会写出的 `*_eval.json` 文件名 |
| `[out]` | `--output_path` 写出的文件与样本量 |
| `[warn]` | 异常口径：`.pt` 文件名不匹配任何 unit 类、覆盖段里有洞、层索引超出模型层数、attention 目标由路径认不出、已装配的层里有量化单元没吃到参数、窄口径没评 attn、pipeline 层数大于 checkpoint 层数、attn 量化下 `use_cache=False`、`--trust_remote_code` 开 / 关、LiveCodeBench 侧 `--limit` 只产部分输出 |
| `[RESULT]` | 准确率，键名 `<指标名>,<filter名>` |

**对比**：

```bash
python <skill>/scripts/compare_results.py \
  --bf16 <run_dir>/bf16_mmlu_pro.json --quant <run_dir>/quant_mmlu_pro.json
```

两侧都带运行参数时，样本量与 `--limited` 自动带出、可比参数自动核对（评测侧清单见 §2，量化侧见 §4.2 表），不用手输 `--sample_len` / `--limited`（喂日志也支持，但这两个参数仍需手输）。输出里 `compared_from` 是 `run metadata (both sides)` 才算核对过，否则照 `comparability_gap` 说的做。`--metric` 传指标名（`exact_match` / `acc_norm` / `acc` / `pass@1`）或完整结果键（`exact_match,custom-extract`）都可，省略则自动挑选。

喂什么文件：lm-eval 任务喂运行日志或结果 JSON；LiveCodeBench 喂 `custom_evaluator` 写出的 `*_eval.json`（生成日志没有分数）。

**报告必填**：

- 样本量（lm-eval 侧取 `[out]` 行或 `--output_path` JSON；LiveCodeBench 侧取 `[lcb-gen]` 行）
- LiveCodeBench 的 `release_version`
- 是否用了 `--limit`
- 数据来源：本次 `prefetch_data.py` 新下载 / 复用盘上已有缓存（约束 2）
- 用的是哪一份 amct：§4.1 打出的 `amct_pytorch.__file__` 路径 + 版本或 commit
- 被评对象是直转（`[direct]` 行，或对比输出的 `direct`）→ 写明**不是 PTQ checkpoint**
- 量化层范围不是全层（`[quant-scope]` 行，或对比输出的 `quant_scope`）→ 写明**被评对象是部分量化 checkpoint**，并写出量化的层范围（约束 4）
- `[warn]` 点名量化单元没吃到参数 → 写明被评对象是**混合对象**（约束 4）
- `[warn]` 没评 attn → 写明**被评对象是 MLP-only、分数不描述完整量化 checkpoint**
- 引用仓库内代码 / 脚本 / 文档写仓库相对路径

**表述纪律**：

- 只报实测 delta，不表述为「量化更好」
- **默认只有准确率、没有误差棒**：`compare_results.py` 只产准确率和 `delta_quant_minus_bf16`，lm-eval 不出 stderr——没显式传 `--bootstrap_iters` 产误差棒，就不得写「差异不显著」「无可见损失」
- **`--limit` 的准确率只能自己跟自己比**：`--limit N` 是 N 道题的准确率，不是官方分数，也不能跟别人全量跑出来的数比，只写小样本冒烟

两侧跑完、对比出 delta 后，把每侧的 task / limit / max_gen_toks、样本量、`--output_path` JSON 路径与 delta 摘要写进 `<run_dir>/` 下的一份运行记录。

## 5. 失败模式

「约束 N」指 [references/constraints.md](references/constraints.md) 的第 N 条（该文开头有触发条件索引，一条约束一行）。

| 症状 | 根因 | 处理 | 约束 / 章节 |
|---|---|---|---|
| hub 请求失败的报错（超时 / 不可达 / 连接被关皆可），特征是报错里不提 `--task` | 请求的 dataset 不在注册表里，shim 把它透传给了真实 loader（`--task` 拼错 / 未注册这两种已被入口的前置检查先接住，那两种报错点名 `--task`） | 照报文点名的 repo / path 补一条注册表 entry，改完跑 `python <skill>/scripts/self_check.py --remote` 核上游布局，再 prefetch | 约束 2、[references/datasets-registry.md](references/datasets-registry.md) |
| `prefetch` 打 MISSING / PARTIAL | 下载没齐（端点 / 代理不通、`allow_patterns` 太窄），或注册表声明的 glob 跟上游布局对不上（`all_of` 缺分片同理） | 下载失败那类，连通性归调用方（端点 / 代理 / 镜像自己指），盘上已有的文件照用；glob 不命中才改 entry 的 glob / `allow_patterns`，改完同样跑一次 `--remote`。不要带着 MISSING / PARTIAL 往下跑 | [references/datasets-registry.md](references/datasets-registry.md) |
| `none of which is on disk` / `has no local file` | 注册表声明的路径不在盘上（同上一行的两类成因） | 同上 | [references/datasets-registry.md](references/datasets-registry.md) |
| `RuntimeError: layer N: the ptq params under ... name submodules this quant block does not have`（内嵌 `KeyError: Submodule 'input_transform' is not found`） | `--algos` 与 ptq 参数训练时用的算法不一致；报文同时点名 attention 目标认错这一成因 | 按训练时的算法传 `--algos` 后重跑 | 约束 3 |
| `--param_dir required unless --bf16 or --direct` | 量化侧既没传 `--param_dir` 也没加 `--direct` | 评 PTQ 结果就传本次 ptq 参数目录；评直转就加 `--direct` + 显式 `--quant_target`；只评基线就加 `--bf16` | §4.2 |
| `--direct requires an explicit --quant_target ...` | `--direct` 没有目录可推导目标 | 传 `--quant_target`（amct 平级目标：`mlp` / `moe` / `attn-linear` / `attn-cache`） | §4.2 |
| `--direct and --bf16 are mutually exclusive` / `--direct reads no ptq params` / `--direct builds no algorithm module` | `--direct` 与 `--bf16` / `--param_dir` / `--algos` 描述的不是同一个被评对象 | 按报文点名去掉其中一个 flag：评 PTQ 结果就去掉 `--direct`，评直转就去掉另一个 | §4.2 |
| `RuntimeError: --param_dir ... holds no layer_<idx>_<unit>.pt file for the requested quant_target ...` | 目录与本次 `--quant_target` 不对应（例如目录是 attn 那轮产的、却按 `mlp` 评） | 按目录实际内容传 `--quant_target`，或换成本次 target 的目录 | 约束 4 |
| `RuntimeError: --quant_target ... names both attention targets` | 一次运行只评一个 attention 目标：两个目标落盘同名，单 `--param_dir` 只能喂一个 | 按 target 各跑一次，两个分数分开报 | 约束 1 |
| `RuntimeError: no ptq params were loaded from --param_dir ...`（报文内嵌 `loaded=0`，**此时不打 `[ptq]` 行**） | ptq 参数一个都没加载上 | 核对 `--param_dir` / 文件名 / `--algos` 后重跑 | 约束 4 |
| `[warn] N quant unit(s) found no ptq params ... mixed PTQ/direct` | 已装配的层里有量化单元没吃到参数，被评对象是混合体 | 数字照样出、不必重跑；要评纯 PTQ 就核对 `--param_dir` 覆盖的 unit 后重跑 | 约束 4 |
| 报错要求加 `--allow_attn_quant` | `--param_dir` 里有 attn 参数，推导出的目标撞了放行闸 | 回 §4.2 停点 B 与用户确认代价后再加 | 约束 1、§4.2 |
| 内嵌 `KeyError: Submodule 'q_cache_quantizer' ...` 的 `name submodules this quant block does not have`（内嵌 `input_transform` 的那条是 `--algos` 不对，见上） | attention 目标认错：目录里是 `attn-cache` 产的参数、按 `attn-linear` 装配（显式传 `--quant_target` 时没有 `[warn]` 兜底） | 按参数实际来源传 `--quant_target`（`attn-cache` / `attn-linear`），判据见 §4.2 停点 B | 约束 1、§4.2 |
| 短输入冒烟能过、长 prompt OOM | MoE 专家张量 + embedding 都压在 npu:0 | 用 `max_memory` 把部分层挪离 npu:0，或缩短 `--seq_len` | env-and-portability.md §2 |
| exit 137 / 143 + dmesg 有 cgroup OOM | page cache 吃掉容器内存配额 | 缩 `--limit` 重跑（lm-eval 入口还可缩 `--batch_size`）；要清 page cache 先问用户（需 root、影响整机所有租户） | env-and-portability.md §2 |
| `requires you to execute the modeling file` | 模型目录带自定义建模代码，未开 `--trust_remote_code` | 先向用户说明这会执行该目录下的 Python，确认后再加 | 约束 5 |
| 大模型在 aclop 编译阶段崩溃 | `PYTHONPATH` 被整个覆盖，丢了 CANN 塞进去的 `python/site-packages` 与 `opp/built-in/op_impl/ai_core/tbe` | 改成前置：`export PYTHONPATH=$AMCT_HOME:$PYTHONPATH` | env-and-portability.md §5 |
| `cannot infer --device_map` | `config.json` 读不出层数 / hidden size，尺寸不可知 | 显式传 `--device_map ''`（单卡）或 `auto`（多卡） | env-and-portability.md §6「定卡数」、§4.2 |

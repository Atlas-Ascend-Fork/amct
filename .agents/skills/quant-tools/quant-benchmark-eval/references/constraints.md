# 关键约束（踩坑换来的 · 运行前必读）

每次评测前都要过一遍这份清单。编号稳定，脚本注释与本文互相引用。

## 1. attention 量化与 KV cache：按 wrapper 现状判断，不沿用历史结论

`attn-linear` 量化本身不应破坏 KV cache——它只量化 q/k/v/o 投影，不改变 HF `past_key_values` 机制。但部分 amct attention wrapper 只面向 blockwise / PPL 最小路径实现，会忽略 `past_key_values` 并返回 `(attn_output, None)`；这种 wrapper 一旦量化 attention，`generate` 只能 `use_cache=False`，每步全上下文重算，速度退化到分钟级 / 题。

**运行前必须按当前源码重新判断**：目标模型的 attention wrapper 是否已支持 `past_key_values` / `cache_position` / `attention_mask`。

- 已接通 → 允许 `attn-linear` 与 `use_cache=True` 同时用于生成式评测。
- 未接通 → 默认只量化 MLP，attn 保持 bf16，KV cache 正常工作。

补 / 改 KV cache 属于适配工作，本 skill 不默认改源码；只有用户明确要求，或用户确认本次 benchmark 必须先补 wrapper 时才进入源码修改。实现原则、最小正确路径与等价性验证清单见 `$model-adapter` 的「Attention 适配原则」。

⚠️ **用户给的量化参数里含 attn-linear 时，必须先说明取舍并征得同意，不得静默跳过**——静默跳过等于把用户量化过的 attn 降级回 bf16，被评测对象就不再是完整 W8A8。三条路由用户选，不要替他决定：

| 选项 | 含义 | 代价 |
|---|---|---|
| (a) | 只量化 MLP，attn 保持 bf16 | 评测可行，但**不是完整 W8A8** |
| (b) | 连 attn 一起量化，wrapper 未接 KV cache | 是完整 W8A8，但会极慢 |
| (c) | 连 attn 一起量化，wrapper 已接 KV cache | 可评完整 attention 量化对象，但必须先 `generate` 冒烟 + 小样本限量确认速度 / 显存 |

走 (b) / (c) 必须显式加 `--allow_attn_quant`；脚本默认拦截 `attn-linear` / `attn-cache`。

## 2. 数据必须离线加载，且用 fastparquet 引擎

直接用 HF `datasets` 会有两个问题：远程 glob 在本机镜像上超时（`FSTimeoutError`）；pyarrow 读 MMLU-Pro 的 parquet 会崩溃（`Index not in dictionary bounds`）。

解决办法：先用 `snapshot_download` 把数据集仓下载到本地，再 monkeypatch `datasets.load_dataset`，改用 **fastparquet** 读本地 parquet、pandas 读本地 jsonl（见 `scripts/bench_offline.py`）。运行前用 `scripts/prefetch_data.py` 预下载一次即可，运行期保持 `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1`。

只注册表里有的任务才真离线：`(repo_id, dataset_name)` 不在 `datasets.json` 的请求会被 shim 原样透传给真实 loader，也就是回到上面那个 `FSTimeoutError`，而且是在**模型加载完、每层量化块都装好之后**才炸，报错里一个字都不提 `--task`。所以 `assemble_and_eval.py` 在加载模型**之前**先跑一次预检 `utils.check_task_offline_ready()`：用 lm-eval 已安装的 task index 把 `--task`（含别名，group 会展开子任务）解析成 `(dataset_path, dataset_name)` 集合，逐个对照注册表。

- 任务名拼错 / 未注册（`arc-challenge`、`gsm8k`）→ 立刻报错，并列出注册表里已有的 repo，告诉你要么改拼写要么补 `datasets.json` + `prefetch_data.py`；
- 对不上注册表但确实要评的任务 → 按报错提示补一条注册表项（glob 要覆盖上游真实布局，补完跑一次 `self_check.py --remote` 兜底）再预下载；
- 解析时 `dataset_name` 允许回落到 `(path, None)`，这是 shim 自己的兜底语义，MATH-500 就靠它（task 声明 `dataset_name: default`，注册表按仓维度不写 name），不需要为它多注册一条。

`self_check.py` 会把这两条反向对照跑一遍（未知任务名、未注册数据集都必须报错，且报错要能区分「索引里没有」和「注册表里没有」），避免预检被改成永真。

shim 的键是 **`(repo_id, dataset_name)`，不是任务名**——它取的是 lm-eval 传给 `datasets.load_dataset(path=..., name=...)` 的那对值。多个任务共用一个仓是允许的（`arc_challenge` / `arc_easy` 共用 `allenai/ai2_arc`、`livecodebench` / `livecodebench_release_v6` 共用 `livecodebench/code_generation_lite`），前提是 `dataset_name` 把它们区分开。两个任务**共用同一对键却声明不同本地数据**是注册表自相矛盾：`local_dataset_map()` 会直接抛 `ValueError`（点名两个任务与该键），而不是让后写的静默覆盖前一个——否则 shim 会把 A 的文件喂给 B，评的不是注册表声明的那份数据。声明完全相同的重复项属无害，仍允许。

## 3. HFLM 直接接受内存中的模型对象

脚本已处理，无需自定义 LM 子类。运行时 `pretrained not of type str` 的 warning 属正常，可忽略。

## 4. PYTHONPATH 必须把 amct 前置，不能整个覆盖

```bash
export PYTHONPATH=$AMCT_HOME:$PYTHONPATH   # 前置（prepend）
```

若直接覆盖（`PYTHONPATH=$AMCT_HOME`），会丢掉 CANN 的 tbe / te，导致大模型在 aclop 编译阶段崩溃。<!-- codespell:ignore tbe,te -->

## 5. page cache 引起的 OOM

进程以 exit 137 / 143 退出、且 dmesg 显示 cgroup OOM 时，启动前先清一次缓存：

```bash
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

## 6. 速度与规模参考

短答案的 MMLU-Pro 约 6 秒 / 题（仅量化 MLP、KV cache 开启），全量 12032 题约需 20 小时。建议用 `--limit N` 限制每个学科的题数，例如 10 题 / 学科共 140 题、约 50 分钟。

长 CoT 任务（MATH500 / LiveCodeBench）明显更慢，且需要更大的 `--max_gen_toks`：MATH500 常用 1024+，LiveCodeBench 常用 2048+。

## 7. 多卡替换层时必须保留 accelerate hook

脚本已自动处理（取出旧 hook、`.to(dev)` 后重新挂回）。手动改装配逻辑时要注意，否则跨卡 `generate` 会报 `tensors on npu:0 and npu:1`。原理见 `scripts/assemble_and_eval.py` 注释。

## 8. 量化 MoE 跑长 prompt 会在 npu:0 上 OOM

量化专家 forward 有额外中间张量，而 npu:0 上又集中了前半层和 embedding，最先被填满：短输入冒烟能过，长 prompt 就 OOM。这是真实容量不足（非显存碎片），`max_split_size_mb` 无效；应改用 `device_map` 的 `max_memory` 把部分层挪离 npu:0，或缩短序列长度。

## 9. 在独立工作目录里运行，不要直接在 `scripts/` 下跑

amct 会往当前目录写 `amct_log/` 和 `fusion_result.json`。在一个专用运行目录（如 `<workdir>/<task>_run/`）里用完整路径调用脚本，保持 skill 的 `scripts/` 只存代码、不留运行产物。

## 10. `--algos` 必须与 ptq 参数训练时用的算法一致

amct 按 `args.algos` 在每个量化块里**构造对应的算法子模块**：`lwc` / `lac` 落在 `weight_quantizer.algorithms.<key>` 与 `input_quant.algorithms.<key>`，`let` / `flatquant` 属 structure 目标，落在 `input_transform` / `hidden_transform`。加载 ptq 参数是拿参数文件里的 key 去找同名子模块，所以两侧不一致直接 `KeyError`：

- 参数由 `let` 训出、脚本却按默认 `lwc,lac` 构造 → `KeyError: "Submodule 'input_transform' is not found in module 'QuantQwen3MLP'."`
- 把 `lwc` 的 key 改名冒充 `let` → `KeyError: "Submodule 'gate_proj.weight_quantizer.algorithms.let' is not found ..."`

因此 `assemble_and_eval.py` / `livecodebench_eval.py` 都暴露 `--algos`（默认 `lwc,lac`），并在装配前用 amct 的 `ALGO_REGISTRY` 校验键名，写错立刻报错并列出已注册算法（`autoround / flatquant / gptq / lac / let / lwc`）。评别人的 ptq 结果前，先确认它训练时用了哪些算法（`$quant-run` 的命令里有）。

## 11. `--param_dir` 必须真的加载上，否则评的是直转不是 PTQ

amct 的 `load_selected_layer_ptq_params(strict=False)` 对缺失文件只打 debug 日志并**回落直转**：目录写错、文件名不对（应为 `layer_<idx>_<unit>.pt`）、算法不匹配时，评测照样出数，但被评对象已经不是用户的 PTQ 结果。脚本因此逐层统计 `loaded/missing`：全部为 0 直接 `RuntimeError`；部分缺失打 `[warn]`（例如只有前 2 层有参数、其余 26 层走直转 → 混合对象，结论里必须写明）。

## 12. `--trust_remote_code` 默认关闭，只在可信模型目录上显式打开

`trust_remote_code=True` 会让 HuggingFace **执行模型目录里的自定义建模代码**（`modeling_*.py`），等价于运行任意 Python。`--model` 是用户给的目录，所以本 skill 不把它写死为 True，而是与 amct 自己的 CLI 保持一致：

- `amct_pytorch/cli/llm/args.py` 的 `--trust_remote_code` 默认 `False`；
- `amct_pytorch/common/models/llm/common/base.py` 在未开启时打 WARNING。

因此 `assemble_and_eval.py` / `livecodebench_eval.py` 都暴露 `--trust_remote_code`（默认关），并把这个值一路透传到 `AutoTokenizer` / `AutoModelForCausalLM` 与 amct 的 pipeline args（amct 内部据此建 config / tokenizer），两个方向都打 `[warn]`：

- 未开启 → 提示部分模型需要它才能加载自定义建模代码，报 “requires you to execute the modeling file” 时再加；
- 已开启 → 提示将执行该目录下的自定义代码，**只对可信目录使用**。

agent 代为执行时的规矩：默认不加；确实需要（非原生支持架构）时先向用户说明「这会执行 `<model_dir>` 里的自定义建模代码」，得到确认再加，并在结论里记一笔。绝不为了让报错消失而默默打开。

## 13. 权重 dtype 参数名随 transformers 版本变化，不能写死 `dtype=`

`from_pretrained` 的权重 dtype 参数在 **transformers 4.56 才从 `torch_dtype` 改名为 `dtype`**。本仓不同模型目录钉的版本并不一致（`examples/algorithms/flatquant/requirements.txt` 4.51.0、`amct_pytorch/experimental/quantization/DeepSeekV3.2/requirements.txt` 4.52.4、`MiniMax-M2.7` 4.57.6），所以写死新名字在 4.52 环境上**不是**报“unexpected keyword”，而是该值落进 `**kwargs` → 被写进 config → 延后炸成 `TypeError: Object of type dtype is not JSON serializable`，很难归因。

因此 `load_float()` 统一走 `utils.dtype_kwargs(torch.bfloat16)`：按已安装版本选 `dtype`（≥4.56）或 `torch_dtype`（更旧，4.57+ 只是弃用告警、未删除）。`check_env.py` 会打印当前 transformers 版本与解析出的参数名，环境不符时先看这一行，不要为了消错去升级/降级用户的 transformers。

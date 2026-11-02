# 关键约束

按需读：撞到下表哪一行的触发条件，才读对应那一条。

## 触发条件索引

每条约束一行触发条件：

| 撞到什么 | 读哪条 |
|---|---|
| `--param_dir` 里有 `layer_*_self_attn.pt` / `layer_*_linear_attn.pt`，或 `--direct` 下 `--quant_target` 点名了 attn 目标（SKILL.md §4.2 停点 B） | #1 |
| 要评的任务不在注册表里，或 prefetch 打 MISSING / PARTIAL | #2 |
| `--algos` 与 ptq 参数训练时用的算法不一致 | #3 |
| `[quant-scope]` 报的量化层范围不是全层，或 `[warn]` 点名量化单元没吃到参数 | #4 |
| 模型目录带自定义建模代码（非原生支持架构） | #5 |

## 1. attention 量化与 KV cache：按 wrapper 现状判断，不沿用历史结论

`attn-linear` 量化本身不应破坏 KV cache——它只量化 q/k/v/o 投影，不改变 HF `past_key_values` 机制。但部分 amct attention wrapper 只面向 blockwise / PPL 最小路径实现，会忽略 `past_key_values` 并返回 `(attn_output, None)`；这种 wrapper 一旦量化 attention，`generate` 只能 `use_cache=False`，每步都重算整段上下文，逐 token 的代价随已生成长度上涨。

**运行前必须按当前源码重新判断**：目标模型的 attention wrapper 是否已支持 `past_key_values` / `cache_position` / `attention_mask`。

- 已接通 → 允许 `attn-linear` 与 `use_cache=True` 同时用于生成式评测。
- 未接通 → attn 量化**仍是被评对象的一部分**：脚本关掉 `use_cache`，每个 token 都重算整段上下文，慢多少只能冒烟实测（机理见本条首段）。代价确认与 `--limit` / 换轻量任务走 SKILL.md §4.2 的强制停点 B。「只量化 MLP」是用户**显式选择**的窄口径。

补 / 改 KV cache 属于适配工作，本 skill 不默认改源码；只有用户明确要求，或用户确认本次 benchmark 必须先补 wrapper 时才进入源码修改。实现原则、最小正确路径与等价性验证清单见 `$model-adapter` 的「Attention 适配原则」。

⚠️ **用户给的量化参数里含 attn-linear 时，必须先说明取舍并征得同意，不得静默跳过**——静默跳过等于把用户量化过的 attn 降级回 bf16，被评测对象就不再是完整量化 checkpoint。路由选项由用户选，不要替他决定：

| 选项 | 含义 | 代价 |
|---|---|---|
| (a) | 只量化 MLP，attn 保持 bf16（**显式窄口径**：自己传 `--quant_target mlp` / `moe`） | 评测可行、KV cache 正常工作，但**分数不描述完整量化 checkpoint**，报告必须写明 |
| (b) | 连 attn 一起量化，wrapper 未接 KV cache | 是完整量化对象，代价见本条首段 |
| (c) | 连 attn 一起量化，wrapper 已接 KV cache | 可评完整 attention 量化对象，但必须先 `--smoke` 冒烟（SKILL.md §4.2）+ 小样本限量确认速度 / 显存 |

走 (b) / (c) 必须显式加 `--allow_attn_quant`；脚本默认拦截 `attn-linear` / `attn-cache`。

`--allow_attn_quant` 是 attention 量化的**放行闸**：`--quant_target` 缺省时脚本按 `--param_dir` 推导，目录里有 `layer_*_self_attn.pt` / `layer_*_linear_attn.pt` 就推导出含 attention 的目标（即路由 (b) / (c)），没有才回落 (a)。判据命令与「有输出 / 无输出」的读法见 SKILL.md §4.2 停点 B。`--direct` 下没有目录可推导，`--quant_target` 必填：它点名 attn 目标时同样撞闸，按上表同一取舍确认后再加 `--allow_attn_quant`。

**两个 attention 目标落盘文件名相同**：amct 只按层类型取 unit 名（`self_attn` / `linear_attn`），与目标是 `attn-linear` 还是 `attn-cache` 无关，所以两边都是 `layer_<idx>_self_attn.pt`。文件内容不同——cache 那侧存的是 `q_cache_quantizer` / `k_cache_quantizer` 的算法参数，linear 那侧存的是激活 / 权重量化器的算法参数——但名字分不出。目标只记在目录上：产 ptq 参数时的目录 leaf 就是 target 字面值（`<output_dir>/ptq_params/<model_name>/<quant_target>`）。脚本按 `--param_dir` 的路径分量精确认目标（`attn-cache-exp3` 这类不命中；一条路径里同时出现两个 attention leaf 算歧义），认不出时取 `attn-linear` 并打一行 `[warn]` 点名歧义。

**一次运行只评一个 attention 目标**：amct 一轮 PTQ 只训一个 target（`llm_ptq.py` 要求 `quant_target` 长度为 1），产物目录 leaf 就是 target 字面值，所以 attn-linear 与 attn-cache 的参数天生分在两个目录里；本 skill 只收一个 `--param_dir`，也就只能喂其中一个。`--quant_target` 同时含两个 attention 目标时直接报错（`ensure_single_attention_target`），按 target 各跑一次、两个分数分开报。

装错目标不静默：另一个目标的量化器要么没建、要么建成 `nn.Identity()`，加载参数两种都抛 `KeyError`；装配循环把它包成本 skill 的 `RuntimeError`，报文同时点名「attention 目标认错」与「`--algos` 与训练时不一致」两种成因（症状见 SKILL.md §5），`[warn]` 在装配前给同一判断。显式传 `--quant_target` 时不推导、也就不打这行 warn，传错直接落到那条 `RuntimeError`。

显式传了窄 `--quant_target` 而目录里有 attn 参数时，脚本打一行 `[warn]` 点名「本次没评 attn、被评对象是 MLP-only」，不拦（要不要接受窄口径是人的取舍），但这行 warn 必须写进报告（SKILL.md §4.3）。

## 2. 数据来源：下载在线、读取本地、任务必须在注册表里

下载**默认在线**：走哪个端点（官方 / 镜像 / 代理）由调用方的环境变量决定，本 skill 既不设默认也不诊断连通性。下载失败但注册表声明的 split 在盘上已齐全时，prefetch 打一行 warn 后照样通过——真正的门是它后面那段逐 split 核验，MISSING / PARTIAL 一律非 0 退出。撞到的报错对照 SKILL.md §5。

读取钉死本地文件 + `fastparquet`：本地读取 shim 无条件接管 `datasets.load_dataset`，不把请求交回它。**钉死读取路径的理由是两侧口径可比**：bf16 与量化侧读的是同一批盘上文件、同一引擎、同一次解析，差值里不掺「两次读取本身不同」这一项；换引擎或换成运行时再去 hub 解析，就把这一项放回了 delta。**读取引擎不能换回 pyarrow，读取路径不能换回运行时网络解析**：要补一份没注册的数据，用 `huggingface_hub.snapshot_download` 拉到 `$BENCH_ASSETS` 再按注册表读。

只有注册表里有的任务才走 shim：`--task` 拼错或未注册时，入口在加载模型**之前**报错并列出已注册的 `(repo_id, dataset_name)`。确实要评一个没注册的任务，报错本身指了路——补一条注册项再 prefetch；字段语义、`any_of` / `all_of`、shim 键与撞键行为见 [datasets-registry.md](datasets-registry.md)。

## 3. `--algos` 必须与 ptq 参数训练时用的算法一致

`--algos` 决定量化块里构造哪些算法子模块；加载 ptq 参数是拿参数文件里的 key 找同名子模块，两侧必须一致。`assemble_and_eval.py` / `livecodebench_eval.py` 装配前用 amct 的 `ALGO_REGISTRY` 校验键名。评别人的 ptq 结果前先确认它训练用了哪些算法（`$quant-run` 的命令里有）。直转不构造算法模块：`--direct` 下不传 `--algos`，传了入口直接报错。撞到的 `KeyError` 对照 SKILL.md §5。

## 4. 量化范围由 `--param_dir` 决定：目录覆盖到哪层，就只量化到哪层

装配按 `--param_dir` 里实际存在的 `layer_<idx>_<unit>.pt` 决定量化哪些层：覆盖到的层才 `build_quant_block` 并灌参数，其余层保留原 bf16 模块。amct 训练侧按 `start_block_idx` / `end_block_idx` 逐层落盘，所以一轮提前停（超时、OOM，或本来就分段跑）留下的目录只覆盖部分层——评测侧照它评，不替用户补齐，也不把没训到的层悄悄换成直转。`--direct` 无目录可读，范围是 pipeline 的全部 `num_layers` 层（amct 自己的直转评测也是全层），`[quant-scope]` 行报出这一点。

覆盖范围是被评对象身份的一部分：只量化了前面一部分层，分数描述的就是「部分量化 checkpoint」，与全层量化的分数不可直接比。`[quant-scope]` 行报出本次覆盖的层范围、层数与覆盖段里的洞，`--output_path` 的 JSON 里记作 `quant_scope`；不是全层时报告必须写明量化层范围（SKILL.md §4.3）。

加载统计仍按 unit 粒度：目录里没有本次 `--quant_target` 对应的文件、或一个 unit 都没加载上，直接 `RuntimeError`（**不打 `[ptq]` 行**，报文里内嵌 `loaded=0`）；已装配的层里有 unit 没吃到参数（例如 MoE 只训了部分 expert）打 `[warn]`，那些 unit 回落直转 → 被评对象是混合体；全部加载上才打 `[ptq]` 行，所以该行的 missing 恒为 0。**混合体数字照样出，这条 warn 必须写进报告**（SKILL.md §4.3）。

两个入口的预检与日志行相同，`quant_scope` 也都落盘：lm-eval 侧在 `--output_path` 的 JSON 里，LiveCodeBench 侧在 `--output_json` 旁边的 `*_run_meta.json` 里（官方 `*_eval.json` 只有分数），报告不用从日志抄。

## 5. `--trust_remote_code` 默认关闭，只在可信模型目录上显式打开

`trust_remote_code=True` 会让 HuggingFace **执行模型目录里的自定义建模代码**（`modeling_*.py`），等价于运行任意 Python。`--model` 是用户给的目录，所以默认关。两个入口按开启与否分别打 `[warn]`：未开启时提示部分模型需要它才能加载自定义建模代码（撞到该报错时再加，症状对照 SKILL.md §5）；已开启时提示将执行该目录下的自定义代码。开不开由用户定（SKILL.md §5）。

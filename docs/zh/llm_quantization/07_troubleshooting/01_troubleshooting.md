# 问题排查

> 本文按现象说明 AMCT LLM 量化流程中的常见问题、检查方法和解决步骤。

## 1. 排查方法

开始排查前先记录以下信息，避免排查过程丢失现场：

- AMCT 版本与安装方式（源码 / wheel）；
- Python、PyTorch、torch_npu 和 CANN 版本（`python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"`，`npu-smi info` 看 CANN/驱动）；
- 模型名称和路径（`model_name` 注册名 + 模型目录）；
- 使用的完整命令；
- `quant_target`、`granularity`、`quant_dtype` 和 bit 配置（或部署 JSON）；
- 完整错误日志（各命令日志在 `{output_dir}/logs/`，文件名见[输出产物参考](../06_reference/05_artifacts.md)第 1 节）。

排查通则：先看报错栈的**最后一帧**属于 AMCT 校验层（参数/配置错误，信息通常自解释）还是模型/设备层（环境问题）；配置类错误按 [CLI 参考](../06_reference/02_cli.md)第 4 节的约束矩阵核对参数组合，产物类问题按[输出产物参考](../06_reference/05_artifacts.md)第 4 节的检查方法核对。

## 2. 安装和环境

### `amct_pytorch` 无法导入

- 现象：`ModuleNotFoundError: No module named 'amct_pytorch'`。
- 检查：`pip list | grep -i amct`；源码运行时确认 `PYTHONPATH` 指向 AMCT 仓库根。
- 原因：未安装 wheel，或源码方式运行但 `PYTHONPATH` 未设置。
- 解决：安装发布包，或 `export PYTHONPATH=/path/to/amct`。**注意**：如果通过 `conda activate` 切换环境，activate 可能覆盖 `PYTHONPATH`——在 activate 之后再 export。

### NPU 不可用 / torch_npu 后端加载失败

- 现象：`RuntimeError: Failed to load the backend extension: torch_npu. You can disable extension auto-loading with TORCH_DEVICE_BACKEND_AUTOLOAD=0.`
- 检查：`npu-smi info`；`python -c "import torch_npu"`。
- 原因：运行环境缺 CANN 环境变量（`LD_LIBRARY_PATH`、`ASCEND_HOME` 等）。常见于非交互 shell（裸 `ssh <host> <cmd>`、CI、cron）——它们不加载完整的用户 shell 配置。
- 解决：运行前 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`（路径按 CANN 安装位置调整）。

### 数据集下载失败（离线环境）

- 现象：`[Errno 101] Network is unreachable` 或 HF 超时后打印 `Using the latest cached version of the dataset since ... couldn't be found on the Hugging Face Hub`。
- 原因：eval 用 wikitext-2、extract 用 pileval，默认从 HuggingFace 拉取；离线机器无法访问。
- 解决：datasets 库会自动回退本地缓存——在有网机器上先跑一次让数据进缓存（`~/.cache/huggingface/datasets/`）；也可设置 `HF_ENDPOINT=https://hf-mirror.com` 走第三方镜像，但使用前应确认镜像来源可信，并校验数据集完整性。缓存命中时离线环境可直接运行。

## 3. 模型加载

### 模型路径错误或文件缺失

- 现象：`OSError` / `NoSuchFile`，或提示找不到 `config.json` / `model.safetensors.index.json` / tokenizer。
- 检查：模型目录需包含 `config.json`、tokenizer 文件和完整 safetensors 权重（分片 + index）。
- 解决：核对 `--model` 路径；分片模型确认 index 里的每个分片文件都在。

### `model_name` 与模型结构不匹配

- 现象：加载或结构解析阶段报 KeyError / AttributeError（找不到适配器预期的模块属性）。
- 原因：`--model_name` 注册名与实际模型家族不一致（如 Qwen3.5 的 dense 模型误用 `qwen3_6`，或 MoE 模型用了 dense 注册名）。
- 解决：对照[CLI 参考](../06_reference/02_cli.md)第 2 节的注册表选名，或从代码中的模型注册表（`MODEL_REGISTRY`）确认；同系列 dense 与 MoE 是不同注册名（如 `qwen3_5` vs `qwen3_5_moe`）。

### `trust_remote_code` 配置问题

- 现象：日志打 warning `xxx is running without --trust_remote_code. Some models may require --trust_remote_code to load custom model code.`
- 原因：未开启远程代码执行。标准架构模型（已进 transformers 的）不需要；自定义架构模型需要。
- 解决：标准模型忽略该 warning（可正常运行）；自定义架构模型仅在确认模型来源可信后加 `--trust_remote_code`。

### 权重 shape 或 dtype 不一致

- 现象：加载权重时报 shape mismatch / dtype 不符。
- 原因：模型版本与适配器预期不符，或权重被其他工具改写过。
- 解决：用官方发布的原始 BF16 checkpoint；`deploy` 明确要求源是未量化、未旋转的 BF16 权重。

## 4. CLI 和配置

### 参数缺失或取值不合法

- 现象：argparse 直接报 `the following arguments are required: --model_name` 或 `invalid choice`。
- 解决：优先查[CLI 参考](../06_reference/02_cli.md)（第 2、3 节给出每个参数的必填项与取值范围），或直接看各命令 `--help`。常见易错点：
  - `--model_name` 必填，取值只能是注册名（见 CLI 参考第 2 节）；
  - `--quant_dtype` 只接受 `int` / `mxfp` / `hifp` / `fp`；
  - `--deploy_platform` 只接受 `A3` / `ascend950`。

### `granularity` 不受支持

- 现象：`ValueError: Deployment requires --granularity tensor`（deploy）；或 eval 报 granularity 仅支持 block/model。
- 原因：各命令的粒度支持范围不同（deploy 仅 tensor、extract/ptq 仅 block、eval 为 block/model）。
- 解决：对照[CLI 参考](../06_reference/02_cli.md)第 4 节的约束矩阵。

### `quant_target` 不合法

- 现象：`extract_ptq_data` / `ptq` 校验报 quant_target 必须恰好一个；argparse 层报 invalid choice。
- 解决：这两个命令每次运行只指定一个目标（`--quant_target mlp`），多目标分多次运行。

### `eval_mode` 与 bit 配置不匹配

- 现象：`ValueError: eval_mode=bf16 requires a bit_config with no <16-bit entries.`（附 BitPolicy 摘要）
- 原因：bf16 基线评估要求全 16bit 配置。
- 解决：bf16 评估用空配置（`bf16.yaml`）或不传 `--bit_config`；量化评估用 `--eval_mode quant`。

### `quant_dtype` 和配置不匹配

- 现象：`Unsupported deployment quantization scheme: ({dtype}, {w}, {a}); supported schemes are int/W8A8, mxfp/W8A8, mxfp/W4A8, hifp/W8A8`；或 `Deployment platform A3 does not support quant_dtype ...`。
- 原因：部署 JSON 规则的 dtype/位宽组合超 schema，或平台能力不匹配（A3 仅 int）。
- 解决：对照[配置参考](../06_reference/04_config.md)第 3 节的方案与平台能力表。

### `bit_config` 字段未生效

- 现象：`eval --eval_mode quant` 的 PPL 与 bf16 几乎相同，日志 `BitPolicy: {}` 为空。
- 原因：没传 `--bit_config`（CLI 设置位宽的唯一途径，无 `--w_bits` 之类的独立参数）；或 bit_config 里与 `quant_target` 相关的分组没有低于 16bit 的条目（重建了量化模块但量化器保持关闭）。
- 解决：核对 `--bit_config` 路径与内容；确认目标分组确实写了低于 16bit 的 `w_bits`/`a_bits` 成对条目。

### PTQ 前后配置不一致

- 现象：`ptq` 找不到校准数据或评估结果异常。
- 原因：`extract_ptq_data` 与 `ptq` 的模型、`quant_target`、`seq_len`、`data_dir` 不一致；或 eval 加载参数时 `--algos` / `--bit_config` / `--quant_dtype` 与训练时不同。
- 解决：三段流程（提取 → 训练 → 评估）保持同一组模型/目标/长度/dtype/算法配置。

## 5. PTQ 数据提取

### 输出目录为空

- 现象：`extract_ptq_data` 跑完 `data_dir` 没有文件，或 `ptq` 报 `Run extract_ptq_data first.`
- 检查：`ls <data_dir>` 应有 `block_{idx}_{target}_in.pkl`（每层一份）和 1-3 个 kwargs pkl。
- 解决：确认 `--data_dir` 在两次运行间一致；重新提取。目标中缀映射：attn-linear / attn-cache 统一为 `attn`，mlp 为 `mlp`，moe 为 `moe`。

### 校准数据不可读

- 现象：`ptq` 加载 pkl 报错。
- 原因：数据由不同模型/seq_len 生成，与当前运行不匹配；或文件损坏（中途被杀）。
- 解决：用一致的模型和 `seq_len` 重新提取；确认磁盘空间充足。

### `seq_len` 或 `nsamples` 不合适

- 现象：提取/训练过慢，或校准统计不稳定。
- 解决：`--nsamples` 控制校准样本数（提取与训练需一致）；`--seq_len` 影响显存和速度（0.8B 级模型 2048 可用，默认 4096）。

### 缺少输入、kwargs 或中间结果

- 现象：`ptq` 报找不到 `block_x_attn_in.pkl` 或 kwargs 文件。
- 原因：目标不一致（提取的是 mlp 数据，训练时给了 attn-linear）；或模型 forward 不产出对应 kwargs（如 attention_mask 为 None 的模型不会有 `attention_mask.pkl`——这是正常现象，kwargs 文件按模型而定）。
- 解决：核对目标一致性；缺失 kwargs 文件先确认是否为该模型的正常行为。

## 6. PTQ 优化

### 找不到 PTQ 数据

见第 5 节「输出目录为空」。

### 缺少 block 或 unit 参数

- 现象：`eval --eval_mode quant` 或 blockwise 程序化部署时提示某层缺参数。
- 原因：`ptq` 的 block 区间没覆盖全部层（`--end_block_idx` 是**排他上界**，24 层模型全量应传 24；超出层数会自动钳制并告警）。
- 解决：确认区间语义后补跑缺失区间；断点续跑会跳过已有参数的 unit（日志 `Skip PTQ unit ... params already exist`）。

### 算法组合不支持

- 现象：报未知算法，或量化器上找不到算法子模块。
- 解决：`--algos` 只接受注册算法（`lwc` / `lac` / `autoround` / `gptq` / `let` / `flatquant` / `learnable_had`）；评估加载 PTQ 参数时必须传与训练**完全相同**的 `--algos`（见第 7 节）。

### 训练无法收敛

- 现象：日志 `Layer {n} epoch {e}/{E} avg_loss=...` 不下降或爆炸。
- 检查：`--base_lr`（默认 1e-5）是否适合当前算法；`--epochs` / `--nsamples` 是否过小导致统计不稳。
- 解决：降低学习率或增加样本/轮数；MoE 模型注意单 expert 很小、损失量级本来就低（1e-6 量级），不要用绝对值跨模型比较。

### PTQ 过程异常中断

- 现象：进程被杀（OOM、手动中断）。
- 解决：直接重跑同一条命令——已保存的 unit 参数会被跳过，从中断处继续；想重训某 unit 就删对应 `layer_*.pt`。

### 输出参数目录或文件命名不符合预期

- 现象：参数出现在 `{output_dir}/ptq_params/{model_name}/{quant_target}/` 而非预期位置。
- 原因：未显式传参数目录（`--moe_mlp_param_dir` 等）时自动创建默认目录并打 warning。
- 解决：显式传参固定位置；命名规则见[输出产物参考](../06_reference/05_artifacts.md)第 2 节。

## 7. 量化评估

### 量化模型无法重建 / PTQ 参数无法加载

- 现象：`KeyError: "Submodule 'gate_proj.weight_quantizer.algorithms.lwc' is not found in module 'QuantQwen35MLP'."`（栈顶在 `ptq_params.py` 的 load_module）。
- 原因：`eval --eval_mode quant` 加载 PTQ 参数时 `--algos` 与 `ptq` 训练时不一致——算法以子模块形式挂在量化器上，参数文件按训练时的结构保存，算法对不上就找不到子模块。
- 解决：eval 命令补上与训练一致的 `--algos`（bit_config / quant_dtype 也应一致）。

### PPL 明显下降

- 现象：量化 PPL 相比 bf16 基线差距过大（如超过 2 倍）或出现 inf/nan。按第 9 节顺序排查。
- 原因之一：量化格式选择不当——W4 位宽下部分模型劣化明显；换 W8A8 / W4A8 等更高位宽组合重新评估。

### BF16 与量化结果无法比较

- 现象：两次 eval 的 PPL 不可比。
- 原因：`seq_len` 不同（wikitext 切块数和内容都变）、数据集版本不同、`granularity` 不同。
- 解决：两次评估保持相同 `seq_len` / `seed` / 数据；记录每次的完整参数。参考口径：bf16 与量化评估的汇总行都含 Model / Quant target / Quant dtype / PPL / BitPolicy 摘要，逐项对齐后再比较。

### 评估日志缺失或断点无法恢复

- 现象：找不到日志文件；`--dump` 续跑失败。
- 解决：日志在 `{output_dir}/logs/eval_{model_name}_{eval_mode}.log`，按 output_dir 找；`--dump` 仅 `granularity=block` 有效，断点文件在 `{output_dir}/temp/eval/` 且只保留最新一个（PPL 成功输出后清理，这是预期行为）。

## 8. 部署导出

### 缺少 PTQ 参数

- 现象：误以为 `deploy` 需要 PTQ 参数。
- 说明：tensorwise `deploy` 从原始 BF16 权重直转，**不消费** PTQ 参数——传 `--algos` 或参数目录会报 `Tensor deployment does not consume --xxx`。PTQ 参数由 `eval --eval_mode quant` 或 blockwise 程序化部署使用。

### blockwise 和 tensorwise 配置混用

- 现象：tensorwise 传了 `quant_target`（warning `Tensor JSON selection replaces quant_target`）或传了 PTQ 参数目录（报错）。
- 解决：tensorwise 只用 `quant_layers_config` JSON + `deploy_platform`；`quant_target` / `bit_config` 由 JSON 规则替代，`quant_dtype` 由 JSON 决定（显式传时与规则一致）。

### `quant_layers_config` 路径不匹配

- 现象：`quant_layers patterns matched no modules: [...]`。
- 原因：JSON 的 pattern 前缀与模型实际模块名不符（不同家族模块前缀不同，如 `model.layers.*` vs `model.language_model.layers.*`）。
- 检查：读模型 `model.safetensors.index.json` 的 `weight_map` 键，取目标模块名去掉 `.weight` 后对照 pattern。
- 解决：按实际模块名改 pattern。ignore pattern 匹配 0 个只 warning 不报错（跨模型复用 JSON 的正常现象）。

### 权重分片或索引文件不完整

- 现象：deploy 结束抛 `Deployment artifact validation failed; see {validation_path}`。
- 检查：打开 `deployment_validation.json` 看 `error_counts` 与 `errors` 明细；内置校验覆盖索引/分片/描述三方一致性、dtype 与 shape 契约、scale 值合法性等。
- 解决：按 errors 列表逐条处理（多为源模型文件不完整或输出目录复用导致）；修复后清空输出目录重跑。

### 部署格式和平台不匹配

- 现象：`Deployment platform A3 does not support quant_dtype`（通用校验），或 DeepSeek V4.1 未指定 `--deploy_platform` 导致平台校验失败。
- 解决：A3 用 int（W8A8）；mxfp 方案（W8A8_MXFP8 / W4A8_MXFP）和 hifp 方案（W8A8_HIFP8）需要 `--deploy_platform ascend950`。

### 导出结果无法被下游框架加载

- 说明：`deployment_validation.json` 的 `status=passed` 只代表 AMCT 导出自检通过，不等于下游框架加载成功。下游加载问题需在实际推理框架中排查。
- 排查时对照[输出产物参考](../06_reference/05_artifacts.md)第 5 节——部署产物的支持范围：int → A3（默认）或 ascend950，mxfp / hifp → ascend950；下游推理框架为 vllm（vllm-ascend）；确认所用框架版本支持 `quant_model_description.json` 中的 `model_quant_type`，以及 `config.json` 中的量化声明（如有）。

### MTP 层处理

- 现象：`Deployment MTP quantization is not supported: ...; use ignore_layers`，或担心 MTP 层被误量化。
- 说明：MTP 检测基于模型 config 的 `num_hidden_layers` / `num_nextn_predict_layers` 推断；存在 config 启发式未识别出 MTP 结构的情况——在 `ignore_layers` 里显式加 `mtp.layers.*` 是可靠兜底。

## 9. 精度异常

量化 PPL 异常时按以下顺序检查（从前往后，命中即处理再重评）：

```text
模型和数据
  → 模型文件完整、model_name 正确、数据集缓存正常
BF16 基线
  → 先确认 bf16 PPL 正常（异常说明模型/环境问题，与量化无关）
bit_config
  → 位宽组合、分组覆盖是否符合预期（eval 日志的 BitPolicy 摘要可直接核对）
quant_target
  → 评估目标与训练目标一致
量化 dtype
  → quant_dtype 与 bit 配置匹配
校准样本
  → nsamples / seq_len 足够且提取与训练一致
PTQ 参数
  → 参数覆盖全部目标层；加载无报错
部署格式
  → 部署侧才掉精度时核对量化方案与平台
```

需要区分四类问题：模型加载错误（第 3 节）、量化配置错误（第 4 节）、PTQ 训练问题（第 6 节）、部署转换问题（第 8 节）。评估侧：未训练层直接量化可能带来数十个百分点级的 PPL 退化，具体幅度因模型、位宽和量化目标而异；全层 PTQ 后差距应显著收窄。差距是否「异常」需要和自己模型的基线对比，而不是绝对阈值。

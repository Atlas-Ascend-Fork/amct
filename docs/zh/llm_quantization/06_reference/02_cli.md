# CLI 参考

> 本文集中说明 AMCT LLM 的命令行入口、参数和执行约束。
>
> 参数说明以当前源码为准。

## 1. 命令总览

AMCT LLM 的工作流通过 4 个 CLI 命令完成评估、校准数据提取、PTQ 优化和部署导出：

| 命令 | 作用 | 主要输入 | 主要输出 |
|---|---|---|---|
| `eval` | 评估 BF16 或量化模型的 PPL | 模型、bit 配置、可选 PTQ 参数 | 评估日志（含 PPL） |
| `extract_ptq_data` | 提取 PTQ 校准所需的中间数据 | 模型、`quant_target`、`data_dir` | 校准数据目录（pkl 文件） |
| `ptq` | 训练并保存 PTQ 量化参数 | 模型、校准数据、bit 配置、算法 | PTQ 参数目录（`layer_*.pt`） |
| `deploy` | 导出部署所需的量化权重和配置 | 模型、`quant_layers_config` | 部署权重 + 索引 + 描述 + 验证报告 |

### 调用方式

推荐使用 `python -m` 模块入口（AMCT 安装后即可用）：

```bash
python -m amct_pytorch.eval          --help
python -m amct_pytorch.extract_ptq_data --help
python -m amct_pytorch.ptq           --help
python -m amct_pytorch.deploy        --help
```

### 命令执行顺序

```text
模型准备
  ↓
eval（BF16 基线）
  ↓
extract_ptq_data（校准数据）
  ↓
ptq（量化参数）
  ↓
eval（量化模型，加载 PTQ 参数对比基线）
  ↓
deploy（部署导出与验证）
```

`deploy` 的 tensorwise 路径直接从原始 BF16 权重转换（DeepSeek V4/V4.1 的官方 FP8 源例外），不依赖 PTQ 参数；其余步骤按上图顺序衔接。各命令对模型、量化目标、粒度和数据配置必须保持一致（见第 4 节）。

## 2. 通用参数

所有命令共享以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--model` | str | `deepseek-ai/DeepSeek-V4-Flash` | 模型路径或 HuggingFace 仓库名。本地目录需包含 `config.json`、tokenizer 和 safetensors 权重 |
| `--model_name` | str | **必填** | 模型注册名，决定使用哪个模型适配器（见下表） |
| `--trust_remote_code` | flag | False | 允许 HuggingFace 仓库执行自定义模型代码；开启后会执行模型仓库中的任意 Python 代码，仅对可信来源使用。纯标准架构模型不需要 |
| `--device` | str | `npu:0` | 运行设备，默认 `npu:0`；可按需指定（如 deploy 权重转换可用 `cpu`）。需要指定物理卡时，通过环境变量 `ASCEND_RT_VISIBLE_DEVICES` 设置可见设备 |
| `--granularity` | str | `block` | 执行粒度。各命令支持范围不同，见第 4 节约束矩阵 |
| `--seed` | int | 42 | HuggingFace 与 PyTorch 的随机种子 |
| `--seq_len` | int | 4096 | 序列长度。eval 的 wikitext 切块与 extract 的校准样本都按此长度生成 |
| `--data_dir` | str | 空 | 校准数据目录（`extract_ptq_data` 的输出、`ptq` 的输入）。这两个命令必填 |
| `--nsamples` | int | 128 | 校准样本数（用于 `extract_ptq_data` 和 `ptq`；校准数据为 pileval 数据集，自动下载、支持本地缓存）。注意：eval 的 PPL 评估使用 wikitext 切块后的全部样本，不受此参数控制 |
| `--output_dir` | str | `./outputs` | 输出根目录。日志统一写 `{output_dir}/logs/` |

### 支持的 `model_name`（模型注册表）

`--model_name` 必须是已注册的适配器名：

| 家族 | 注册名 |
|---|---|
| Qwen | `qwen3`、`qwen3_moe`、`qwen3_5`、`qwen3_5_moe`、`qwen3_6`、`qwen3_6_moe`、`qwen3_next` |
| DeepSeek | `deepseek_v3_2`、`deepseek_v4`、`deepseek_v4_1` |
| GLM | `glm5`、`glm5_2`、`glm5_3` |
| 其他 | `hy_v3`、`longcat_lite`、`longcat_next`、`kimi_k3` |

`deepseek_v3_2`、`deepseek_v4` 需要 `--trust_remote_code`。`deepseek_v4_1`、`glm5_3`、`kimi_k3` 仅支持 tensorwise 部署，不支持 PPL 评估与 PTQ；其余注册名支持 PPL/PTQ。

`model_name` 与模型结构必须匹配（例如 Qwen3.5 系列对应 `qwen3_5`，其 MoE 版本对应 `qwen3_5_moe`）；不匹配时模型加载或结构解析会报错。注册表只表示适配器存在；PPL/PTQ 与 tensorwise deploy 的支持范围不同，tensorwise 支持模型见第 3 节末尾，其他适配说明见[模型适配文档](../04_model_adaptation/01_overview.md)。

## 3. 量化参数

### 量化目标与数据类型

| 参数 | 类型 | 默认值 | 取值 | 说明 |
|---|---|---|---|---|
| `--quant_target` | list | 空 | `mlp` / `moe` / `attn-linear` / `attn-cache` | 量化目标。`extract_ptq_data` 和 `ptq` 每次必须且只能指定一个；`deploy`（tensorwise）不使用此参数 |
| `--quant_dtype` | str | 空 | `int` / `mxfp` / `hifp` / `fp` | 量化数据类型，选择量化/反量化实现。`int` 为整数量化（激活逐 token 动态、权重逐 channel）；`mxfp` 为 MX 块浮点（块大小 32）；`hifp` 为华为 HiFloat 浮点格式，PTQ 伪量化支持 HIFP8 / HIFP4（按位宽选择），tensorwise 部署当前仅支持 HIFP8/W8A8；`fp` 为块浮点（行块 128、列块 `--block_size_col`，块内共享 scale） |
| `--bit_config` | str | 空 | YAML 路径 | 按角色描述位宽的配置文件（全局 `w_bits`/`a_bits` + 分组覆盖）。`eval` / `ptq` 设置位宽的**唯一 CLI 途径**；不传时全 16bit（相当于不量化）。`extract_ptq_data` 不消费位宽；`deploy` 的位宽由 `quant_layers_config` 决定。样例见 `amct_pytorch/configs/`，详见[配置参考](04_config.md) |
| `--block_size_col` | int | 128 | `1` / `128` | 块浮点（`quant_dtype=fp`）的列块大小，其余 dtype 忽略 |
| `--scale_dtype` | str | `fp32` | `fp32` / `e8m0` | 块浮点（`quant_dtype=fp`）的 scale 数据类型，其余 dtype 忽略 |
| `--is_per_tensor` | flag | False | — | 激活裁剪使用 per-tensor 统计 |

`quant_dtype` 与 `w_bits`/`a_bits` 是正交的两个维度：dtype 决定量化实现类，bit 配置决定该实现内部的位宽。16bit 的层会直接旁路（bypass）量化。

### 量化算法

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--algos` | list | 空 | 启用的量化算法，按注册目标路由到权重/激活/结构。注册算法：`lwc`（权重裁剪）、`lac`（激活裁剪）、`autoround`（权重）、`gptq`（权重）、`let`（结构，lwc+let 等效 OmniQuant）、`flatquant`（结构）、`learnable_had`（结构，可学习 Hadamard 变换） |
| `--k_size` | int | 128 | 结构类算法（`learnable_had` / `flatquant`）的变换矩阵大小；要求对应维度可被其整除 |

**重要**：用 `ptq` 训练出的参数，在 `eval --eval_mode quant` 加载时必须传**完全相同的** `--algos`——算法以子模块形式挂在量化器上，算法不一致会导致参数加载报 `KeyError: Submodule ... is not found`。

### PTQ 训练参数（`ptq` 命令）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--cali_bsz` | int | 4 | PTQ 训练使用的 batch size |
| `--base_lr` | float | 1e-5 | PTQ 训练使用的学习率 |
| `--optimizer` | str | `adamw` | PTQ 训练使用的优化器类型，当前支持 `adamw` / `adam` / `sgd` / `signsgd` |
| `--weight_decay` | float | 0.0 | 权重衰减（L2 正则）系数，适用于 `adamw` / `adam` / `sgd` / `signsgd`；`0.0` 表示不启用 |
| `--momentum` | float | 0.9 | `sgd` 类型优化器的动量参数（`signsgd` 同样使用） |
| `--lr_scheduler` | str | `cosine` | 学习率调度器 |
| `--lr_step_size` | int | 1 | step 调度器的步长 |
| `--lr_gamma` | float | 0.1 | step 调度器的衰减系数 |
| `--epochs` | int | 15 | PTQ训练轮数 |
| `--start_block_idx` | int | 0 | 起始 block 序号（**包含**） |
| `--end_block_idx` | int | 61 | 结束 block 序号（**不包含**，排他上界；超出模型层数时自动钳制到层数并告警） |

block 区间语义示例：36 层模型全量训练为 `--start_block_idx 0 --end_block_idx 36`，实际处理 block 0 到 35。

### PTQ 参数目录

PTQ 参数按量化目标分目录存放，三组参数分别对应三个目标：

| 参数 | 对应目标 | 自动创建的默认目录 |
|---|---|---|
| `--attn_linear_param_dir` | `attn-linear` | `{output_dir}/ptq_params/{model_name}/attn-linear/` |
| `--attn_cache_param_dir` | `attn-cache` | `{output_dir}/ptq_params/{model_name}/attn-cache/` |
| `--moe_mlp_param_dir` | `mlp`、`moe` | `{output_dir}/ptq_params/{model_name}/{quant_target}/` |

`ptq` 未显式指定参数目录时自动创建默认目录并打 warning。`eval --eval_mode quant` 通过相同参数加载训练好的 PTQ 参数；未提供参数目录的层回退直接量化。已有参数文件的 unit 会被跳过（断点续跑，日志出现 `Skip PTQ unit ... params already exist`）。

### eval 专属参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--eval_mode` | str | `bf16` | `bf16`：直接加载原始 BF16 block 评估；`quant`：重建量化模块并按位宽开启量化器，可加载 PTQ 参数 |
| `--dump` | flag | False | 保存/恢复 blockwise 断点（仅 `granularity=block`）。断点文件在 `{output_dir}/temp/eval/`，只保留最新一个，PPL 成功输出后清理 |

### deploy 专属参数

| 参数 | 类型 | 默认值 | 取值 | 说明 |
|---|---|---|---|---|
| `--deploy_platform` | str | None（解析为 `A3`） | `A3` / `ascend950` | 目标部署硬件。A3 仅支持 `int`；ascend950 支持 `int`、`mxfp` 和 `hifp`（hifp 仅 W8A8）。DeepSeek V4.1 必须显式指定 |
| `--quant_layers_config` | str | None | JSON 路径 | tensorwise 部署的层选择与量化规则（module glob + 逐规则 dtype/位宽），**deploy 必填**。JSON 规则替代 `quant_target` 和 `bit_config`；dtype 由 JSON 决定，显式传 `--quant_dtype` 时应与规则一致。样例见 `amct_pytorch/configs/deploy/`，详见[配置参考](04_config.md) |

### tensorwise 支持的模型

`deploy --granularity tensor` 当前支持以下注册名：

| 家族 | 支持 tensorwise 的注册名 |
|---|---|
| Qwen | `qwen3`、`qwen3_5`、`qwen3_5_moe`、`qwen3_6`、`qwen3_6_moe` |
| DeepSeek | `deepseek_v4`、`deepseek_v4_1` |
| GLM | `glm5_2`、`glm5_3` |
| 其他 | `hy_v3`、`kimi_k3` |

其余注册名（`qwen3_moe`、`qwen3_next`、`deepseek_v3_2`、`glm5`、`longcat_lite`、`longcat_next`）当前不支持 CLI tensorwise deploy；其中部分模型仍可通过 Python workflow 的 blockwise 路径部署。

## 4. 命令约束

### granularity 支持矩阵

| 命令 | `block` | `model` | `tensor` |
|---|---|---|---|
| `eval` | ✅（逐 block 评估，支持 `--dump`） | ✅（整模型加载评估 PPL） | ❌ |
| `extract_ptq_data` | ✅（唯一支持） | ❌ | ❌ |
| `ptq` | ✅（唯一支持） | ❌ | ❌ |
| `deploy` | ❌ | ❌ | ✅（唯一支持，CLI 强制校验） |

- `deploy` 不支持 `block` 粒度，只支持 `tensor`；传 `block`（或不传，使用默认值）会报 `ValueError: Deployment requires --granularity tensor`。
- `eval` 的 `--dump` 仅在 `granularity=block` 下有效。

### `quant_target` 的必填与单目标限制

- `extract_ptq_data` 和 `ptq`：必须指定 `quant_target`，且**每次只能一个**目标（多个目标需要分多次运行，各自的校准数据和参数目录可以共存）。
- `eval`：可传多个 `quant_target`（评估时按位宽开关相关量化器）；`eval_mode=bf16` 不需要。
- `deploy`（tensorwise）：不使用 `quant_target`，层选择由 `quant_layers_config` JSON 决定；显式传 `quant_target` 会打 warning `Tensor JSON selection replaces quant_target`。

### `eval_mode` 与 bit 配置的关系

- `eval_mode=bf16` 要求 bit 配置不含任何低于 16bit 的条目，否则报错：

  ```text
  eval_mode=bf16 requires a bit_config with no <16-bit entries.
  ```

- `eval_mode=quant` 且 bit 配置中与 `quant_target` 相关的条目存在低于 16bit 的位宽：重建量化 block 并开启量化器；无低于 16bit 的条目：重建量化模块但保持量化关闭。
- `eval_mode=quant` 加载 PTQ 参数时，`--algos`、`--bit_config`、`--quant_dtype` 必须与 `ptq` 训练时一致。

### 参数依赖与互斥

| 规则 | 表现 |
|---|---|
| `data_dir` 对 `extract_ptq_data` / `ptq` 必填 | 缺失时命令在参数校验阶段报错 |
| `ptq` 直接读取 `extract_ptq_data` 生成的校准数据 | `quant_target`、`seq_len`、模型必须与提取时一致 |
| tensorwise `deploy` 拒绝 PTQ 相关参数 | 传 `--algos` / `--attn_linear_param_dir` / `--attn_cache_param_dir` / `--moe_mlp_param_dir` 报错 `Tensor deployment does not consume --xxx` |
| tensorwise `deploy` 的 `hifp` 仅支持 `ascend950` | `hifp` 仅支持 W8A8；A3 上使用会报平台不支持。当前由 DeepSeek V4/V4.1 适配器支持 |
| `bit_config` 的 `w_bits`/`a_bits` 必须成对 | 只写一个会在加载时报 `Incomplete bit entry` |
| deploy 输出目录必须为空或不存在 | 与源目录重叠或非空时报错 |

### 预留不生效参数

以下参数在已发布脚本和文档中出现过，当前版本**不生效**——传入的值会被忽略并重置为默认值，同时打 warning。保留它们是为了兼容既有脚本：

| 参数 | 文档默认值 | 行为 |
|---|---|---|
| `--min_lr` | 0.0 | 传值被忽略，打 `[RESERVED]` warning |
| `--wikitext_final_out` | 空 | 同上 |

## 5. 示例和行为限制

各命令的具体示例统一参照[快速上手](../01_getting_started/02_quick_start.md)（以 Qwen3-4B 为例）。

### 行为限制

- CLI 的 `deploy` 仅支持 `granularity=tensor`，不支持 `block`。
- 预留参数（`--min_lr` / `--wikitext_final_out`）保持兼容性存在但不生效。

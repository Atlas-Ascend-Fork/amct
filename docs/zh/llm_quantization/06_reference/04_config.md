# 配置参考

> 本文说明 AMCT LLM 量化和部署过程中使用的配置文件。
>
> 配置格式与校验行为以当前源码为准；文中样例均摘自 `amct_pytorch/configs/`。

## 1. 配置类型

| 配置 | 格式 | 使用命令 | 作用 |
|---|---|---|---|
| bit_config | YAML | `eval` / `ptq` | 按角色描述量化位宽（`--bit_config`），决定哪些层量化、量化到几位；`extract_ptq_data` 不消费位宽，`deploy` 由部署 JSON 决定 |
| quant_layers_config | JSON | `deploy`（tensorwise） | 部署层选择与量化规则（`--quant_layers_config`），决定哪些模块转成什么格式的部署权重 |
| 部署平台选择 | CLI 参数 | `deploy` | `--deploy_platform`（A3 / ascend950），限定部署产物支持的量化数据类型 |

样例文件在仓内 `amct_pytorch/configs/` 目录：

| 路径 | 内容 |
|---|---|
| `amct_pytorch/configs/bf16.yaml` | 空配置（全 16bit，BF16 基线评估用） |
| `amct_pytorch/configs/w4a4.yaml`、`w4a8.yaml`、`w8a8.yaml` | 常用均匀位宽配置 |
| `amct_pytorch/configs/example_w4a8.yaml` | 带 schema 注释的完整示例（分组覆盖演示） |
| `amct_pytorch/configs/deploy/` | 按模型族分目录的 tensorwise 部署规则；qwen 下为 qwen3 / qwen3_5 / qwen3_6（dense 与 MoE）的 int / mxfp8 W8A8 规则，deepseek 另有 int / hifp 规则，glm / hyv3 / kimi 也有各自规则 |

## 2. bit_config

### YAML 基本结构

bit_config 是一个 YAML mapping，顶层 `w_bits` / `a_bits` 设置全局默认，四个分组按角色覆盖（均带 Apache license 头的仓内文件，以下展示省略头部）：

```yaml
# 全局默认：所有 linear 的权重量化位数 / 激活量化位数
w_bits: 4
a_bits: 8

# 分组覆盖（可选）
attn-linear:      # 注意力线性层
  o_proj: { w_bits: 8, a_bits: 16 }
mlp:              # 稠密 MLP
  down_proj: { w_bits: 4, a_bits: 16 }
moe:              # MoE 专家
  routed:
    down_proj: { w_bits: 4, a_bits: 16 }
  shared:
    w_bits: 8
    a_bits: 8
attn-cache:       # KV Cache（键为 q/k/p/v，单值位宽）
  k: 8
  v: 8
```

### 全局配置

- 顶层 `w_bits` / `a_bits` 是全局默认，**必须成对出现**（只写一个报错）；都不写（如 bf16.yaml 空文件）等效全 16bit，即不量化。
- 位宽取值只允许 `4` / `8` / `16`（整数）。`16` 表示该处旁路（bypass）量化。

### 分组与键

| 分组 | 键形态 | 说明 |
|---|---|---|
| `attn-linear` | `layer_name` + `{w_bits, a_bits}`，或组级默认 | 注意力里的线性层。`layer_name` 匹配量化模块的 `name=`（真实层名，如 `o_proj`、`in_proj_qkv`） |
| `mlp` | 同上 | 稠密 MLP 的 gate/up/down projection 等 |
| `moe` | 支持一层子分组（`routed` / `shared`）再嵌套层名条目 | MoE 的路由专家与共享专家；支持点分子组路径（如 `moe.routed`） |
| `attn-cache` | `q` / `k` / `p` / `v` 单值位宽 | 不做 w/a 成对检查；缺省键为 16。任一键小于 16 即表示 KV Cache 量化 |

模型中不存在的 `layer_name` 会被**静默忽略**——可以放心地在同一份 bit_config 里写多个模型的层名。

### 分组覆盖规则

解析顺序从最具体到最不具体：`layer_name` 条目 → 组级默认 → 顶层全局。每一级取**第一个同时含 `w_bits` 和 `a_bits` 的完整条目**；没有完整条目时回落全局。注意：线性分组内出现不完整条目会在加载时报 `Incomplete bit entry`，不会被静默跳过。即：

```text
attn-linear.o_proj（`layer_name` 条目） → attn-linear（组默认） → 顶层 w_bits/a_bits
moe.routed.down_proj → moe.routed → moe → 顶层
```

### 权重和激活字段约束

- 线性分组内任何节点出现 `w_bits` / `a_bits` 之一时**必须两者都有**（含嵌套节点），否则加载时报 `Incomplete bit entry`。
- 所有标量值必须为 int 且 ∈ (4, 8, 16)。
- `eval` / `ptq` 在 CLI 上设置位宽的**唯一途径**是 `--bit_config`；没有独立的 `--w_bits` / `--a_bits` 命令行参数。`extract_ptq_data` 不消费位宽；`deploy` 的位宽由部署 JSON 决定。
- bit_config 与 `--quant_dtype` 是正交维度：dtype 决定量化实现类（int / mxfp / hifp / fp），位宽决定该实现内部的 bits。

## 3. 部署配置

### quant_layers_config（JSON）

`deploy --granularity tensor` 必须提供 JSON 规则文件，顶层只允许两个键：

```json
{
  "quant_layers": [
    { "pattern": "model.language_model.layers.*.mlp.gate_proj",
      "quant_dtype": "int", "w_bits": 8, "a_bits": 8 }
  ],
  "ignore_layers": [
    "model.visual.*",
    "model.language_model.embed_tokens",
    "lm_head",
    "mtp.layers.*"
  ]
}
```

**结构化规则字段**（四个字段缺一不可、多一个也不行）：

| 字段 | 约束 |
|---|---|
| `pattern` | 模块名 glob（fnmatch 语义，大小写敏感），**不带 `.weight` 后缀**；匹配 0 个模块时报错 |
| `quant_dtype` | `int` / `mxfp` / `hifp`；`hifp` 仅支持 W8A8，且需要 `--deploy_platform ascend950`（当前由 DeepSeek V4/V4.1 适配器支持） |
| `w_bits` / `a_bits` | 仅 4 或 8；schema 允许 W8A8 / W4A8 / W4A4，当前可执行的部署方案为 int/W8A8、mxfp/W8A8、mxfp/W4A8、hifp/W8A8 |

**匹配语义**：

- `ignore_layers` 优先于 `quant_layers`；应用 ignore 后无选中模块时报错。
- 一个未被 `ignore_layers` 排除的模块被多个 quant pattern 匹配 → 报错；每个选中模块必须恰好命中一条规则。
- ignore pattern 匹配 0 个模块 → 仅 warning，不报错（对跨模型复用 JSON 很重要）。
- MTP 层（按模型 config 的 `num_hidden_layers` / `num_nextn_predict_layers` 推断）不允许被量化；config 启发式未识别出 MTP 结构时，用 `ignore_layers` 显式排除是可靠兜底。
- 整个模型只允许**一种量化方案**（所有规则的 dtype+位宽组合必须一致）。
- JSON 规则替代 `quant_target` / `bit_config`，并决定层的 `quant_dtype`：显式传 `quant_target` 会打 warning `Tensor JSON selection replaces quant_target; quant_target is not used`；显式传 `--quant_dtype` 时应与 JSON 规则一致；传 `--algos` 或 PTQ 参数目录会报错 `Tensor deployment does not consume --xxx`。

**平台与 dtype 能力**：

| `--deploy_platform` | 支持的 quant_dtype | 说明 |
|---|---|---|
| `A3`（默认） | `int` | int 仅支持 W8A8 |
| `ascend950` | `int`、`mxfp`、`hifp` | mxfp 支持 W8A8 / W4A8；hifp 仅 W8A8 |

平台不支持的 dtype 会报错，跨平台时用 `--deploy_platform` 指定正确平台。DeepSeek V4.1 的 tensor 部署必须显式指定平台。

### blockwise 与 tensorwise 的配置差异

| 维度 | tensorwise（CLI deploy） | blockwise（仅 Python workflow 类可达） |
|---|---|---|
| 层选择 | `quant_layers_config` JSON | `quant_target` + `bit_config` |
| 权重来源 | 原始 BF16 checkpoint 直转（不依赖 PTQ 参数） | 量化模块 + PTQ 参数导出 |
| 量化格式 | `int`（W8A8_DYNAMIC：I8 + BF16 scale + 零 offset）/ `mxfp`（W8A8_MXFP8 / W4A8_MXFP：块大小 32） | 由 `quant_dtype` + bit_config 决定的伪量化模块导出 |
| 附加参数 | 拒绝 `--algos` / PTQ 参数目录 | 可使用 PTQ 参数目录（缺失时回退直接量化） |

## 4. 配置检查

配置错误会在两个阶段出现：文件加载/校验阶段（参数校验、JSON schema），以及模型匹配阶段（pattern 与实际模块名比对）。

### bit_config 加载错误

| 场景 | 错误信息（原文） |
|---|---|
| 顶层只写 `w_bits` 或 `a_bits` | `Incomplete bit entry at top level: must set both 'w_bits' and 'a_bits', got {...}.` |
| 组内条目不完整（含嵌套） | `Incomplete bit entry at {path!r}: must set both 'w_bits' and 'a_bits', got {...}.` |
| 值非 int | `{path} must be int, but got {type}.` |
| 值不在 (4, 8, 16) | `{path} must be one of (4, 8, 16), but got {value}.` |
| YAML 顶层不是 mapping | `bit_config yaml at {path} must be a mapping at top level` |

### 配置与命令的匹配检查

| 场景 | 表现 | 检查方法 |
|---|---|---|
| `eval_mode=bf16` 但 bit_config 含低于 16bit 的条目 | `ValueError: eval_mode=bf16 requires a bit_config with no <16-bit entries.` | bf16 评估用 `bf16.yaml`（空配置）或位宽全 16 的配置 |
| bit_config 与 quant_target 不相关（如 bit_config 只写了 mlp 分组但 target 是 attn-linear） | 评估会重建量化模块但相关量化器关闭（不报错但结果为不量化的 PPL） | 检查 bit_config 里目标分组是否真的写了低于 16bit 的条目 |
| deploy JSON pattern 与模型结构不匹配 | `quant_layers patterns matched no modules: [...]` | 用模型权重索引（`model.safetensors.index.json` 的键）核对模块名前缀 |
| 未被 ignore 的模块被多个 pattern 命中 | `Modules matched multiple quant_layers patterns: {...}` | 保证 pattern 互斥 |
| 平台不支持所选 dtype | `Deployment platform A3 does not support quant_dtype`（通用校验当前不带具体 dtype）；部分适配器会报 `... on A3 does not support quant_dtype values: [...]` | 对照平台能力表 |
| 位宽组合超 schema | `{path}: Quantization configuration compatibility issue: bit combination W{w}A{a} is outside the AMCT selection schema; expected W8A8, W4A8, or W4A4` | 只用三种支持组合 |
| 未知 JSON 顶层键 | 加载即报错，只允许 `quant_layers` / `ignore_layers` | — |

### 常见配置错误的日志表现

| 日志特征 | 原因 | 处理 |
|---|---|---|
| `Tensor JSON selection replaces quant_target; quant_target is not used`（WARNING） | deploy tensorwise 时显式传了 `--quant_target` | 属预期提醒，层选择以 JSON 为准 |
| `ignore_layers patterns matched no modules: ['lm_head']`（WARNING） | ignore pattern 在该模型上无匹配（如无独立 lm_head） | 可忽略；不影响导出 |
| `Skip PTQ unit ... params already exist`（INFO） | ptq 断点续跑，已有参数的 unit 跳过 | 预期行为；如需重训删对应 `layer_*.pt` |
| `No 'moe_mlp_param_dir' specified for quant_target 'mlp'; auto-create ...`（WARNING） | ptq 未显式指定参数目录 | 自动创建默认目录，可显式指定以固定位置 |

## 5. 配置示例

以下样例摘自 `amct_pytorch/configs/`（省略 license 头）。

### bf16.yaml（BF16 基线评估）

```yaml
# Bf16 baseline. Empty config -> everything stays at 16 bit.
```

- 适用：`eval --eval_mode bf16`；任何命令的全 16bit 参照。

### w4a4.yaml（均匀 W4A4）

```yaml
# Uniform W4A4 across all linears.
w_bits: 4
a_bits: 4
```

- 适用：`ptq` / `eval --eval_mode quant` 的 W4A4 方案。

### w4a8.yaml（均匀 W4A8 + MoE 分组微调）

```yaml
# Uniform W4A8 across all linears.
w_bits: 4
a_bits: 8

moe:
  routed:
    w_bits: 4
    a_bits: 8
  shared:
    w_bits: 8
    a_bits: 8
```

- 适用：`ptq` / `eval --eval_mode quant` 的 W4A8 方案（共享专家保 W8A8）。

### w8a8.yaml（均匀 W8A8）

```yaml
# Uniform W8A8 across all linears. Anything not covered (e.g. KV cache) stays
# at 16 by default; override below if needed.
w_bits: 8
a_bits: 8
```

- 适用：W8A8 方案。

### example_w4a8.yaml（分组覆盖完整演示，带 schema 注释）

```yaml
w_bits: 4
a_bits: 8

attn-linear:
  o_proj: { w_bits: 8, a_bits: 16 }        # 输出投影提高位宽
  comp_wgate:       { w_bits: 8,  a_bits: 8 }
  idx_weights_proj: { w_bits: 16, a_bits: 16 }  # 小模块保 fp16

mlp:
  down_proj: { w_bits: 4, a_bits: 16 }

moe:
  routed:
    down_proj: { w_bits: 4, a_bits: 16 }
  shared:
    w_bits: 8
    a_bits: 8

attn-cache:
  k: 8
  v: 8
  # q / p not listed -> stay at 16
```

- 适用：需要按角色差异化位宽的场景；不存在的层名（如非 MLA 模型的 `comp_wgate`）会被静默忽略。

### 部署规则 qwen3_5_w8a8_int.json（tensorwise deploy）

```json
{
  "quant_layers": [
    {"pattern": "model.language_model.layers.*.mlp.gate_proj", "quant_dtype": "int", "w_bits": 8, "a_bits": 8},
    {"pattern": "model.language_model.layers.*.mlp.up_proj", "quant_dtype": "int", "w_bits": 8, "a_bits": 8},
    {"pattern": "model.language_model.layers.*.mlp.down_proj", "quant_dtype": "int", "w_bits": 8, "a_bits": 8},
    {"pattern": "model.language_model.layers.*.self_attn.q_proj", "quant_dtype": "int", "w_bits": 8, "a_bits": 8},
    {"pattern": "model.language_model.layers.*.self_attn.k_proj", "quant_dtype": "int", "w_bits": 8, "a_bits": 8},
    {"pattern": "model.language_model.layers.*.self_attn.v_proj", "quant_dtype": "int", "w_bits": 8, "a_bits": 8},
    {"pattern": "model.language_model.layers.*.self_attn.o_proj", "quant_dtype": "int", "w_bits": 8, "a_bits": 8},
    {"pattern": "model.language_model.layers.*.linear_attn.in_proj_qkv", "quant_dtype": "int", "w_bits": 8, "a_bits": 8},
    {"pattern": "model.language_model.layers.*.linear_attn.in_proj_z", "quant_dtype": "int", "w_bits": 8, "a_bits": 8},
    {"pattern": "model.language_model.layers.*.linear_attn.in_proj_b", "quant_dtype": "int", "w_bits": 8, "a_bits": 8},
    {"pattern": "model.language_model.layers.*.linear_attn.in_proj_a", "quant_dtype": "int", "w_bits": 8, "a_bits": 8},
    {"pattern": "model.language_model.layers.*.linear_attn.out_proj", "quant_dtype": "int", "w_bits": 8, "a_bits": 8}
  ],
  "ignore_layers": [
    "model.visual.*",
    "model.language_model.embed_tokens",
    "model.language_model.norm",
    "lm_head",
    "mtp.layers.*"
  ]
}
```

- 适用：`deploy --granularity tensor --quant_layers_config <本文件>`，qwen3_5 系列模型，A3（int）；W8A8 动态量化。
- 同系列变体：`qwen3_5_w8a8_mxfp8.json`（同目录，mxfp，适用 ascend950）、`qwen3_5_moe/`（MoE 版，mlp pattern 换为 `mlp.experts.*.{gate,up,down}_proj`）、`qwen3_6/`、`qwen3_6_moe/`（Qwen3.6 系列）。
- 适配新模型时先用 `model.safetensors.index.json` 的键核对 pattern 前缀（见第 4 节检查方法）。

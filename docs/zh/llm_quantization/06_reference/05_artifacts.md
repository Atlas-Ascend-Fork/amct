# 输出产物参考

> 本文说明 AMCT LLM 各阶段生成的文件、目录结构和检查方法。
>
> 目录树以 Qwen3.5-0.8B（24 层）与 Qwen3.6-35B-A3B（40 层、256 expert/层）为例，模型相关数值（层数、文件数）以目标模型为准。

## 1. 产物总览

| 阶段 | 产物 | 生成命令 | 后续读取方 |
|---|---|---|---|
| 评估 | 评估日志（含 PPL） | `eval` | 用户读取 PPL 对比 |
| 评估（断点） | `{output_dir}/temp/eval/*.pt` | `eval --dump` | `eval --dump` 续跑（仅自身） |
| PTQ 数据 | 校准数据目录（pkl 文件） | `extract_ptq_data` | `ptq` |
| PTQ 参数 | `{output_dir}/ptq_params/**/layer_*.pt` | `ptq` | `eval --eval_mode quant`（加载参数对比） |
| 部署 | 权重分片 + 索引 + 描述 + 验证报告 | `deploy` | 下游推理框架 |
| 运行日志 | `{output_dir}/logs/*.log` | 全部命令 | 排障 |

各命令的日志文件名（`{output_dir}/logs/` 下）：

| 命令 | 日志文件 |
|---|---|
| `eval` | `eval_{model_name}_{eval_mode}.log` |
| `extract_ptq_data` | `extract_ptq_data.log` |
| `ptq` | `ptq_{start_block_idx}_{end_block_idx}.log` |
| `deploy` | `deploy.log` |

PPL 数值只出现在 eval 日志中，汇总行格式：

```text
Model: {model_name}; Quant target: {targets}; Quant dtype: {dtype}; PPL: {ppl}; BitPolicy: {bit_config 摘要}
```

## 2. PTQ 产物

### 校准数据目录（`--data_dir`）

`extract_ptq_data` 把中间数据平铺写入 `--data_dir`（24 层模型 + mlp 目标为 26 个文件）：

```text
ptq-data/
├── block_0_mlp_in.pkl          # 每个 block 一份输入激活（torch.save 的张量，
├── block_1_mlp_in.pkl          #   按样本维 concat 成单张量）
├── …（每层一份，共 num_layers 个）
├── block_23_mlp_in.pkl
├── position_ids.pkl            # embedding 前向的输入 kwargs
└── position_embeddings.pkl
```

命名规则：`block_{layer_idx}_{target}_in.pkl`。target 由量化目标映射——`attn-linear` / `attn-cache` 统一记为 **`attn`**（如 `block_0_attn_in.pkl`）；`mlp` 记 `mlp`；`moe` 记 `moe`。

kwargs 文件最多三个：`position_ids.pkl`、`position_embeddings.pkl`、`attention_mask.pkl`——**是否生成取决于模型 forward 是否用到**（Qwen3.5、Qwen3.6 系列只生成前两个，attention_mask 为 None 时不落盘）。

多个 quant_target 的数据共存于同一目录（文件名中缀区分），每个 target 需要单独运行一次 `extract_ptq_data`。

`ptq` 通过 `data_dir` 读回这些文件；缺失时报 `Run extract_ptq_data first.`。**data_dir 里的数据与模型、`quant_target`、`seq_len` 必须严格对应**。

### PTQ 参数（`ptq_params`）

`ptq` 把每个 PTQ unit 的可学习参数存为 `layer_{layer_idx}_{unit_name}.pt`（`unit_name` 中 `.` 替换为 `_`）。目录结构：

```text
{output_dir}/
└── ptq_params/
    └── {model_name}/
        └── {quant_target}/
            ├── layer_0_mlp.pt        # mlp 目标：unit 名为 mlp
            ├── layer_1_mlp.pt
            └── …
```

各目标的典型 unit 名：

| quant_target | unit 名 | 参数文件示例 |
|---|---|---|
| `mlp` | `mlp` | `layer_0_mlp.pt` |
| `attn-linear` / `attn-cache` | `self_attn`（部分模型为 `linear_attn`） | `layer_0_self_attn.pt` |
| `moe` | `expert_{idx}`（每个 expert 一个 unit） | `layer_0_expert_0.pt` … `layer_0_expert_255.pt`（Qwen3.6-35B-A3B 每 block 256 个） |

未显式指定参数目录时自动创建 `{output_dir}/ptq_params/{model_name}/{quant_target}/` 并打 warning。已有参数文件的 unit 重跑时跳过（断点续跑），日志 `Skip PTQ unit '{unit}' in layer {idx}: params already exist`。

**参数文件内部结构**（以 `layer_0_mlp.pt`、算法 `lwc lac` 为例）是「量化器子模块 → 参数」的嵌套 dict：

```text
gate_proj.weight_quantizer.algorithms.lwc:
    clip_factor_max: Tensor [3584, 1]   # lwc 的逐 channel 裁剪因子
    clip_factor_min: Tensor [3584, 1]
up_proj / down_proj:                    # 同构，维度随各 projection 变化
input_quant.algorithms.lac:
    clip_factor_min / clip_factor_max: Tensor [1]
    maxval / minval: Tensor 标量        # lac 的激活统计
```

注意两点：

- 内容因算法而异：GPTQ 存 `{"H": hessian, "nsamples": int}`；AutoClip 存 `clip_factor_min/max` + `maxval/minval`；无 `export_ptq_params` 的模块收集全部可训练参数。
- **weight scale / offset 不在 PTQ 参数文件里**——PTQ 参数是算法的可学习参数与统计量；部署侧的 scale 由量化器在导出时现场计算。

读取方：`eval --eval_mode quant` 和 blockwise 程序化部署（`--attn_linear_param_dir` / `--attn_cache_param_dir` / `--moe_mlp_param_dir` 指向对应 `{model_name}/{quant_target}/` 目录）。未提供参数目录的层回退直接量化。

## 3. 部署产物

### tensorwise（CLI `deploy --granularity tensor`）

输出目录（必须为空或不存在、不得与源目录重叠）：

```text
deploy-out/
├── model.safetensors-00001-of-00001.safetensors   # 转换后的权重分片，沿用源分片文件名
├── model.safetensors.index.json                    # 权重索引（total_size 重算后刷新）
├── config.json                                     # qwen3_5 等通用路径为源文件原样复制；
│                                                   #   DeepSeek V4/V4.1 会移除源 quantization_config
├── quant_model_description.json                    # 逐张量量化类型标注
├── deployment_validation.json                      # 导出自检报告
├── logs/deploy.log
└── （tokenizer*.json、merges.txt、vocab.json、chat_template.jinja、
     README.md、LICENSE 等源目录支持文件原样复制）
```

各文件的量化产物约定：

| 量化方案 | 张量形态 |
|---|---|
| int W8A8（`W8A8_DYNAMIC`） | `{module}.weight`（INT8，2 维）+ `{module}.weight_scale`（BF16，形状 `[out, 1]`，有限且 > 0）+ `{module}.weight_offset`（全零同形） |
| mxfp W8A8（`W8A8_MXFP8`） | `.weight`（F8_E4M3）+ `.weight_scale`（UINT8，`[out, in/32]`，块大小 32） |
| mxfp W4A8（`W4A8_MXFP`） | `.weight`（UINT8 打包，每字节 2 个 E2M1）+ `.weight_scale`（UINT8） |
| hifp W8A8（`W8A8_HIFP8`） | `.weight`（UINT8，HiFloat8 编码）+ `.weight_scale`（FLOAT32，形状 `[out, 1]`） |

未选中的权重保持 FLOAT 原样存放。

`quant_model_description.json` 的结构：全局字段（`version` / `model_quant_type` / `metadata` / `group_size`；int 方案另有 `is_rot_used`，mxfp / hifp 方案为 `optional`）+ 逐张量条目（权重名 → `W8A8_DYNAMIC` / `W8A8_MXFP8` / `W4A8_MXFP` / `W8A8_HIFP8` / `FLOAT`）。

`config.json`：qwen3_5 / qwen3_6 等 Qwen 路径不改写（源文件复制）；DeepSeek V4 / V4.1、GLM-5.3、Kimi-K3 的 tensorwise 路径会移除源文件中的 `quantization_config`（GLM/Kimi 同时移除 `text_config.quantization_config`，如有），不写入新的量化元数据。blockwise 路径（程序化）会写 compressed-tensors 风格的 `quantization_config`（`config_groups` / `format` / `quant_method` / `ignore` 等）。

### blockwise（仅 Python workflow 类可达，CLI 不可达）

CLI 当前强制 tensorwise；通过 `LlmDeployWorkflow` 程序化调用时产物为：

```text
deploy-block/
├── layer_000.safetensors          # 每个 block 一份量化权重分片
├── layer_001.safetensors           #   （文件名层数部分宽度 = max(3, 层数位数)，
├── …                              #    62 层模型 → layer_000 … layer_061）
├── rest_00000.safetensors         # 未量化权重合并分片，单分片上限 8GB
├── model.safetensors.index.json    # 索引刷新
├── config.json                    # 含 quantization_config（compressed-tensors 风格）
└── （支持文件复制）
```

blockwise 不生成 `quant_model_description.json` / `deployment_validation.json`。

## 4. 完整性检查

### 自带检查：deployment_validation.json

`deploy`（tensorwise）结束时自动执行产物校验并写 `deployment_validation.json`，校验失败会抛异常终止命令。报告示例（W8A8 int，186 个选中模块）：

```json
{
  "status": "passed",
  "selected_modules": 186,
  "tensor_count": 860,
  "dynamic_tensor_count": 558,
  "float_tensor_count": 302,
  "total_tensor_bytes": 1250867648,
  "error_counts": {"missing": 0, "extra": 0, "duplicate": 0, "mapping": 0,
                    "dtype": 0, "shape": 0, "values": 0, "description": 0,
                    "metadata": 0, "files": 0},
  "errors": [],
  "contract": {"quant_dtype": "int", "weight_bits": 8, "activation_bits": 8,
               "model_quant_type": "W8A8_DYNAMIC", "group_size": 0},
  "artifacts": {"config": "config.json",
                "description": "quant_model_description.json",
                "index": "model.safetensors.index.json",
                "shards": ["model.safetensors-00001-of-00001.safetensors"]}
}
```

内置校验覆盖：三方集合一致性（索引 `weight_map` ↔ 分片实际张量 ↔ description 标注）、逐分片 header 扫描（不整载权重）、辅助张量值检查（int 的 `weight_scale` 有限且 > 0、`weight_offset` 全零）、shape 契约（`[out,1]` 或 `[out, in/group]`、group 整除性）、MTP 层必须 FLOAT、`metadata.total_size` 与实际字节和一致、禁止出现旋转/静态量化参数（`rot.weight`、`.input_scale` 等）。

### 手工最小检查

不依赖 AMCT 时的最小验证清单：

1. **索引与分片一致**：`model.safetensors.index.json` 的 `weight_map` 引用的每个分片文件存在；分片内张量集合与 `weight_map` 键一致（用 safetensors header 读取即可，不必加载权重数据）。
2. **量化声明匹配**：`quant_model_description.json` 中标注 `W8A8_DYNAMIC` 等量化类型的张量，其 dtype 必须符合方案契约（如 INT8/BF16 scale）；标注 `FLOAT` 的必须为浮点 dtype。
3. **配置声明匹配**：`config.json` 的 `quantization_config.model_quant_type`（有该字段时）与 description 一致。
4. **最小加载验证**：用 safetensors 库逐分片 `safe_open` 读取 header，确认张量名、dtype、shape 无异常；进一步可用下游框架的加载入口试载。

### PTQ 侧检查

- `ptq` 后确认目标 block 区间内每个 `layer_{idx}_{unit}.pt` 都已生成（`ls ptq_params/{model_name}/{quant_target}/`）；断点续跑时靠「已存在即跳过」语义补齐。
- `eval --eval_mode quant` 能加载参数目录且 PPL 与 bf16 基线同数量级，即参数可用（加载报 `KeyError: Submodule ...` 通常是 `--algos` 与训练时不一致）。

## 5. 产物记录和兼容性

### PTQ 产物（校准数据 + PTQ 参数）

| 记录项 | 内容 |
|---|---|
| AMCT Git 提交 | 获取并记录当前使用 commit 的 hash |
| 模型支持 | 支持 PPL/PTQ 的注册名；完整列表见[CLI 参考](02_cli.md)第 2 节 |
| 量化配置 | bit_config 位宽组合（w/a ∈ 4/8/16，如 W8A8 / W4A8 / W4A4）× `quant_dtype`（int / mxfp / hifp / fp）× 量化算法（`--algos`，取值见[CLI 参考](02_cli.md)第 3 节） |
| 部署格式 | 不直接部署——CLI tensorwise `deploy` 不消费 PTQ 参数；blockwise 程序化部署会加载可用的 PTQ 参数后导出 |
| 目标硬件 | Ascend NPU（校准与训练在 NPU 上执行，参数本身无平台绑定） |
| 下游推理框架 | 不直接对接——经 `deploy` 转换后由推理框架加载 |

### 部署产物

| 记录项 | 内容 |
|---|---|
| AMCT Git 提交 | 获取并记录当前使用 commit 的 hash |
| 模型支持 | 支持 tensorwise deploy 的注册名，见[CLI 参考](02_cli.md)第 3 节“tensorwise 支持的模型” |
| 量化配置 | quant_layers_config JSON：`quant_dtype` 取 int / mxfp / hifp；当前可执行方案为 int/W8A8、mxfp/W8A8、mxfp/W4A8、hifp/W8A8 |
| 部署格式 | `W8A8_DYNAMIC` / `W8A8_MXFP8` / `W4A8_MXFP` / `W8A8_HIFP8`；tensorwise 五件套（见第 3 节） |
| 目标硬件 | 由量化配置和平台能力决定：int → A3（默认）或 ascend950；mxfp / hifp → ascend950 |
| 下游推理框架 | vllm（vllm-ascend） |

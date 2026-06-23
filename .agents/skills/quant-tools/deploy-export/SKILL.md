---
name: deploy-export
description: 在 amct 中导出 deploy-ready 的量化模型目录。适用于用户已经明确模型、quant_target、bit、dtype，并希望把直转或 PTQ 结果导出成 HuggingFace safetensors 权重目录；同时由 skill 按模板补齐对 infer 仓及通用下游消费者可直接使用的量化说明文档。
---

# Deploy 导出

这个 skill 只用于导出 deploy-ready 的量化模型目录。

调用这个 skill，默认表示：
- 当前量化方案已经接受
- 当前任务已经进入导出阶段
- 导出结果默认面向 infer 仓及通用下游消费者

这个 skill 不负责：
- 新模型适配
- 推荐量化方案
- 跑 PTQ
- 评测量化精度

精度是否可行，应由 `$direct-quant-eval` 或 `$quant-workflow` 先完成。

## 用户调用格式

- 如果用户明确点名调用这个 skill，先返回这个 skill 对应的固定输入模板，再开始执行。
- 不替用户改成别的 skill；只有当用户没有指定 skill、只是描述目标时，才建议是否回到 `$quant-workflow`。
- 固定输入模板见：[../references/skill-input-template.md](../references/skill-input-template.md)

## 先看哪些文件

按下面顺序阅读：

1. `amct_pytorch/workflows/llm_deploy.py`
2. `amct_pytorch/common/models/llm/common/deploy_export.py`
3. 如需确认 PTQ 参数按 target 的加载方式，再看 `amct_pytorch/common/models/llm/common/base.py`
4. 生成说明文档时，参考：
   - `references/deploy_quantization_template.md`
   - 文档需要能被下游推理仓或其他消费者直接理解和接入

## 命令模板

导出命令以 `examples/deploy.sh` 为唯一权威模板，不要凭空拼命令。按当前方案替换：

- `--model`：原模型目录（分片或单文件 `model.safetensors`）
- `--model_name`：**必填**，已注册适配器名（见 `.agents/docs/repo-map.md`）
- `--granularity block`：**必填**（同 eval，默认 model 不量化）
- `--quant_target` / `--quant_dtype`：与评测阶段一致（`--quant_dtype` 必填 `int`/`mxfp`）
- `--bit_config`：与评测同一份 yaml
- `--output_dir`：导出目录

> `deploy.py` 只产出权重 + `config.json` + `model.safetensors.index.json`；`deploy_quantization.md` 由本 skill 另行生成。

## 固定流程

1. 先确认用户目标是“只做 deploy 导出”。
   - 如果用户还没拿到量化方案，回到 `$quant-workflow`
   - 如果用户只想评测当前量化方案，回到 `$direct-quant-eval`
2. 先判定导出模式。
   - 用户期望模式：直转 / PTQ / 未明确
   - 实际导出模式：直转 / PTQ / 混合
   - 如果两者不一致，必须在执行前说明，并在结论中保留这个差异
3. 检查前置条件。
   - 模型已经适配
   - 目标 `quant_target` 的量化通路已经能正常构图
   - 当前请求使用 `granularity=block`
   - 当前 `quant_dtype` 已实现 `export_deploy()`
4. 明确导出配置。
   - `model`
   - `model_name`
   - `quant_target`
   - `quant_dtype`
   - bit 配置
   - `output_dir`
5. 如果用户没给 `output_dir`，优先使用稳定默认值：
   - `outputs/deploy/<model_name>/<quant_dtype>/<target-tag>`
6. 如果目标目录已经存在，先判断能否复用旧结果。
   - 能回答当前问题则先复用
   - 需要重跑则不要静默覆盖
7. 运行 `amct_pytorch/cli/llm/deploy.py` 完成权重导出。
8. 导出后至少检查：
   - 非权重支持文件已复制到输出目录
   - `model.safetensors.index.json` 已刷新
   - index 中引用的 shard 文件都存在
   - `config.json` 已刷新
9. 在 deploy 代码导出完成后，由 skill 继续生成：
   - `deploy_quantization.md`

## 产物自检（probe，静态、不碰 infer runtime）

导出后做静态产物核验，**贴输出、不空打勾**；infer 运行期消费（前向/算子）由 infer 仓负责，本处不跑推理：

- **object 级**：抽查本次 `--quant_target` 对应模块在 `config.json` 的 `quantization_config` 中确为量化（`format`/`num_bits` 非空、未误入 `ignore`），与方案一致。
- **权重级对账**：safetensors 中量化层的 `*.weight_scale`（int 路径 + `*.weight` int 化 / mxfp + `*.weight_packed`）键数与目标量化层数匹配；`model.safetensors.index.json` 0 missing、shard 齐全。
- **PTQ 参数**：带 `--algos` 时核对导出已消费 ptq 参数（对应层 scale 非占位/非全 0），且 deploy 的 `--algos` 与 ptq 一致（否则 load_module KeyError，详见 `$quant-run`）。
- **范围**：自检止于产物结构与 config 契约；不验证 infer 前向 / 算子生效（交 infer 仓）。

## 关键边界

这一点必须明确：

- deploy 代码只负责导出权重目录。
- `amct_pytorch/workflows/llm_deploy.py` 不生成 `deploy_quantization.md`（早期内嵌的自动 stub 已移除）。
- `deploy_quantization.md` 由当前 skill 按模板生成；裸跑 `amct_pytorch/cli/llm/deploy.py` 不会产出该文档。
- 生成后的文档放在导出目录根下，与本次权重目录一一对应。

换句话说：

1. `llm_deploy.py` 负责“导出什么”
2. `deploy-export` skill 负责“如何把这次导出说明清楚”

不要把文档生成逻辑混进 deploy 核心代码。

## 主交付文档

`deploy_quantization.md` 是本次导出的主说明文档。

它同时服务两类使用者：
- 下游消费者：读取量化说明，用于完成量化接入
- agent：复用导出结论，不再重复整理同一份信息

这份文档重点回答：
- 量化系数是什么
- 算法原理是什么
- 这些系数怎么使用
- 下游如何据此实现推理
- `quantization_config` 是否可以被下游直接复用
- 量化目标如何映射到下游运行对象（如 `Linear`、`MoEGMM`、KVCache）
- 哪些模块允许回退，哪些场景必须停止接入

注意：
- 不要贴当前代码仓源码
- 不要写成当前代码仓的实现注释
- 要写成对通用下游消费者可直接使用的部署规格说明
- 不能只停留在“总结”层，必须写到下游可以据此实现量化推理为止
- 必须显式给出“最小交付物是否齐备”和“下游接入边界”

## 文档生成要求

`deploy_quantization.md` 使用：
- `references/deploy_quantization_template.md`

生成时遵守下面规则：

1. 内容基于这次实际导出的权重目录，不要脱离产物空写模板。
2. 如果是直转导出，就明确写“当前系数来自直转量化”。
3. 如果是 PTQ 导出，就明确写“当前系数来自 PTQ 参数加载后导出”。
4. 如果是混合导出，就明确写清：
   - 哪些 target 是 PTQ
   - 哪些 target 回退成了直转
5. 如果某一路当前只是预留语义，例如 `attn-cache`，要明确写“预留”，不要伪装成已完全定义。
6. 文档内容必须与当前导出目录绑定；不要引用其它 run 的结果。

## 边界规则

- 这个 skill 只负责导出，不负责量化方案设计、PTQ 训练和精度判读。
- 当前 deploy 主流程只支持 `granularity=block`。
- 缺少 `attn_linear_param_dir`、`attn_cache_param_dir`、`moe_mlp_param_dir` 不应直接报错；这意味着对应 target 按直转导出。
- 如果用户本意是 PTQ 导出，但实际某个 target 回退成了直转，必须在结论里显式指出。
- 如果用户本意是纯 PTQ 导出，但最终只能做混合导出或直转导出，不能静默继续。
- `config.json` 刷新（`generate_quant_config`）会透传 `w_bits`/`a_bits`，W8A8 等 int 路径写出正确的 `num_bits`/`input_activations`，无需按 bit 额外核对。
- 注意 `is_mx`（mxfp）分支仍硬编码 8-bit，与 int 路径无关。
- 如果输出目录中已有旧结果，不要默认覆盖。
- 引用仓库内路径时，一律使用仓库相对路径。

## 对外说明要求

默认需要产出：

- `deploy_quantization.md`
  - 使用 `references/deploy_quantization_template.md`
  - 作为下游消费者和 agent 共用的主说明文档
  - 默认需要覆盖最小交付物、模块映射、运行时契约、回退策略

## 输出要求

结束时必须给出一张“Deploy 导出卡”，至少包含下面字段：
- `当前导出配置`
- `用户期望导出模式`
- `实际导出模式`
- `输出目录`
- `关键产物`
- `下游接入兼容性`
- `PTQ 参数加载情况`
- `回退情况`
- `deploy_quantization.md 生成情况`
- `当前限制`
- `结果复用情况`

## Deploy 导出卡

```md
# Deploy 导出卡

## 当前导出配置
- `model`：
- `model_name`：
- `quant_target`：
- `quant_dtype`：
- bit 配置：
- `granularity`：

## 用户期望导出模式
- 用户表述：
- 期望模式：

## 实际导出模式
- 直转导出 target：
- PTQ 导出 target：
- 是否存在混合导出：

## 输出目录
- `output_dir`：
- 是否复用旧目录：

## 关键产物
- 支持文件复制：
- `model.safetensors.index.json`：
- `config.json`：
- shard 文件刷新：
- `deploy_quantization.md`：

## 下游接入兼容性
- `quantization_config` 是否可直接复用：
- 模块映射关系：
- 是否存在需下游补实现的模块：
- 是否存在需进一步补充说明的契约：

## PTQ 参数加载情况
- `attn_linear_param_dir`：
- `attn_cache_param_dir`：
- `moe_mlp_param_dir`：

## 回退情况
- 哪些 target 回退为直转：
- 回退原因：

## deploy_quantization.md 生成情况
- 是否已按模板生成：
- 使用模板：
- 是否与本次导出目录一一对应：

## 当前限制
- 当前 deploy 能力边界：
- 需要额外核对的点：

## 结果复用情况
- 是否复用旧结果：
- 如果没有复用，原因：
```

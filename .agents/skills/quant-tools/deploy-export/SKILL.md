---
name: deploy-export
description: 导出 deploy-ready 量化模型目录（HuggingFace safetensors 权重）+ 按模板补齐 deploy_quantization.md 交付文档。触发场景：已明确 模型/quant_target/bit/dtype，要把直转或 PTQ 结果导出给 infer 仓等下游消费。不适用于：模型适配、方案/算法推荐、跑 PTQ、判精度（各→对应 skill）。
---

# Deploy 导出

## 概述

把已接受的量化方案导出成 deploy-ready 模型目录（HuggingFace safetensors 权重 + `config.json` + `index.json`），并由本 skill 按模板补齐 `deploy_quantization.md` 交付文档。

调用即默认：量化方案已接受、任务已进入导出阶段、产物面向 infer 仓及通用下游消费者。**精度是否可行须先由 `$direct-quant-eval` 或 `$quant-workflow` 完成。**

不负责：新模型适配 / 推荐方案 / 跑 PTQ / 评测精度（各 → 对应 skill）。

## 用户调用格式

- 用户明确点名调用，先返回本 skill 的固定输入模板再执行。
- 不替用户改成别的 skill；仅当用户没指定 skill、只描述目标时，才建议是否回 `$quant-workflow`。
- 固定输入模板见 [../references/skill-input-template.md](../references/skill-input-template.md)。

## 先读哪些文件

1. `amct_pytorch/workflows/llm_deploy.py`
2. `amct_pytorch/common/models/llm/common/deploy_export.py`
3. 确认 PTQ 参数按 target 的加载方式，再看 `amct_pytorch/common/models/llm/common/base.py`
4. 生成说明文档时参考 `references/deploy_quantization_template.md`（文档须能被下游推理仓或其他消费者直接理解接入）。

## 命令模板

导出命令以 `examples/deploy.sh` 为唯一权威模板，不要凭空拼命令。按当前方案替换：

- `--model`：原模型目录（分片或单文件 `model.safetensors`）
- `--model_name`：**必填**，已注册适配器名（见 `.agents/docs/repo-map.md`）
- `--granularity block`：**必填**（同 eval，默认 model 不量化）
- `--quant_target` / `--quant_dtype`：与评测阶段一致（`--quant_dtype` 必填 `int`/`mxfp`）
- `--bit_config`：与评测同一份 yaml
- `--output_dir`：导出目录

## 固定流程

1. 先确认用户目标是“只做 deploy 导出”。
   - 还没拿到量化方案 → 回 `$quant-workflow`
   - 只想评测当前方案 → 回 `$direct-quant-eval`
2. 先判定导出模式：用户期望（直转 / PTQ / 未明确）vs 实际（直转 / PTQ / 混合）；**两者不一致必须在执行前说明，并在结论保留差异**。
3. 检查前置：模型已适配 / 目标 `quant_target` 量化通路能正常构图 / 用 `granularity=block` / `quant_dtype` 已实现 `export_deploy()`。
4. 明确导出配置：`model` / `model_name` / `quant_target` / `quant_dtype` / bit 配置 / `output_dir`。
5. 用户没给 `output_dir` → 用稳定默认 `outputs/deploy/<model_name>/<quant_dtype>/<target-tag>`。
6. 目标目录已存在 → 先判能否复用；**需重跑不要静默覆盖**。
7. 运行 `amct_pytorch/cli/llm/deploy.py` 完成权重导出。
8. 导出后至少检查：非权重支持文件已复制 / `model.safetensors.index.json` 已刷新且引用 shard 都在 / `config.json` 已刷新。
9. 导出完成后由 skill 继续生成 `deploy_quantization.md`。

## 产物自检（probe，静态、不碰 infer runtime）

导出后做静态产物核验，**贴输出、不空打勾**；infer 运行期消费（前向 / 算子）由 infer 仓负责，本处不跑推理：

- **object 级**：抽查本次 `--quant_target` 对应模块在 `config.json` 的 `quantization_config` 中确为量化（`format`/`num_bits` 非空、未误入 `ignore`），与方案一致。
- **权重级对账**：safetensors 中量化层的 `*.weight_scale`（int 路径 + `*.weight` int 化 / mxfp + `*.weight_packed`）键数与目标量化层数匹配；`index.json` 0 missing、shard 齐全。
- **PTQ 参数**：带 `--algos` 时核对已消费 ptq 参数（对应层 scale 非占位 / 非全 0），且 deploy 的 `--algos` 与 ptq 一致（否则 load_module KeyError，详见 `$quant-run`）。
- **范围**：止于产物结构与 config 契约；不验证 infer 前向 / 算子生效（交 infer 仓）。

## 边界规则

- **deploy 代码只导权重目录**：`llm_deploy.py` 不生成 `deploy_quantization.md`（早期内嵌 stub 已移除）；该文档由本 skill 按模板生成、放导出目录根、与本次权重一一对应。即 `llm_deploy.py` 管“导出什么”、本 skill 管“把这次导出说清楚”，两者不混。
- 只负责导出，不做方案设计 / PTQ 训练 / 精度判读。
- 当前 deploy 主流程只支持 `granularity=block`。
- 缺 `attn_linear_param_dir` / `attn_cache_param_dir` / `moe_mlp_param_dir` **不应报错**：意味着对应 target 按直转导出。
- **本意 PTQ 但某 target 回退直转 → 结论必须显式指出；本意纯 PTQ 但只能混合 / 直转 → 不能静默继续。**
- `config.json` 刷新（`generate_quant_config`）对 Linear group **硬编码 `num_bits=8/8`**（不按 `bit_config` 透传 `w_bits`/`a_bits`，`llm_deploy.py` 调用还硬编码 `w4a8=False/w4a4=False`）：**W8A8 恰好正确，但 int W4A8/W4A4 的 `num_bits` 会被错标成 w8**——非 W8A8 的 int 路径**必须**在产物自检里核对 `config.json` 的 `num_bits` 与方案一致（对应已知 deploy↔infer config 误标 bug）。`is_mx`（mxfp）分支另走、亦硬编码 8-bit。
- 输出目录已有旧结果不默认覆盖；引用仓库内路径一律用相对路径。

## deploy_quantization.md 交付

本次导出的主说明文档，用 `references/deploy_quantization_template.md`，同时服务下游消费者（完成量化接入）与 agent（复用导出结论）。

**内容要点**（写到下游可据此实现量化推理为止，不停留在“总结”层）：量化系数是什么 / 算法原理 / 系数怎么用 / 下游如何据此实现推理 / `quantization_config` 能否被下游直接复用 / 量化目标 → 下游运行对象映射（`Linear` / `MoEGMM` / KVCache）/ 哪些模块可回退、哪些场景必须停止接入；并显式给出“最小交付物是否齐备”与“下游接入边界”。

**生成规则**：
1. 基于本次实际导出产物，不脱离产物空写模板；内容与当前导出目录绑定，不引用其它 run。
2. 直转导出写“系数来自直转量化”；PTQ 导出写“系数来自 PTQ 参数加载后导出”；混合导出写清哪些 target 是 PTQ、哪些回退直转。
3. `attn-cache` 等仅预留语义的要明确写“预留”，不伪装成已完全定义。
4. 不贴源码、不写成实现注释；写成对通用下游可直接使用的部署规格说明。

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

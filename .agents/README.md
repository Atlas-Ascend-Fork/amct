# amct 大模型量化 Agent Skills

面向昇腾 NPU 大模型量化压缩的 Agent Skills：封装 amct（`amct_pytorch`）的 LLM 量化全流程（`eval / extract_ptq_data / ptq / deploy` + blockwise PTQ），让你用自然语言完成「模型 → 量化方案 → 部署权重」。同时适配 Claude Code 与 OpenCode。

## 快速开始

```bash
bash scripts/init-agent.sh        # 生成客户端视图；--claude / --opencode 可单装
```

装好后直接用自然语言描述任务，例如：

> 把 Qwen3-30B-A3B 量化到 W8A8 并导出部署权重

客户端据此进入编排入口 `quant-workflow`，自动判阶段、串联子能力，并在方案选择、PTQ 升级、导出等关键决策处与你确认。

## 用法：典型场景 → 对应能力

无需记命令；客户端按各 skill / 子代理 frontmatter 的 `description` 匹配诉求自动选用。

| 你的诉求 | 对应能力 |
|---|---|
| 端到端量化 + 导出 | `quant-workflow`（编排入口，下列各步按需调度）|
| 新模型还没适配 | `model-adapter` |
| 要第一轮量化方案 | `scheme-recommendation` |
| 跑直转、看 PPL / delta | `direct-quant-eval` |
| 直转 delta 超阈、要升级 | `algorithm-recommendation` → `algorithm-validation` |
| 导出部署权重 + 交付文档 | `deploy-export` |

编排顺序：适配（未适配时）→ 方案 → 直转（`delta ≤ 0.2` 可直接导出）→ 超阈则升级 PTQ → 导出。

## 量化能力

| 维度 | 取值 |
|---|---|
| 量化目标 | `mlp` / `moe` / `attn-linear` / `attn-cache` |
| 位宽·数据类型 | W8A8 / W4A8 / W4A4；`int` / `mxfp` |
| 算法 | 直转（minmax / smoothquant / awq / gptq 等）、可训练 PTQ（lwc / lac / autoround / omniquant）|
| bit 配置 | yaml `--bit_config`（顶层 `w_bits/a_bits` + `moe.routed/shared` + `attn-cache` 的 q/k/p/v）|
| 评测·阈值 | Wikitext PPL；默认接受 `delta ≤ 0.2` |
| 产物 | compressed-tensors（`config.json` + 分片权重 + `deploy_quantization.md`）|

## 协作结构

编排入口 `quant-workflow` 按职责委派三个子代理，经共享状态文件 `progress.md` 传递上下文：

- `quant-analyzer` —— 适配性分析、方案 / 算法推荐（只读代码）
- `quant-implementer` —— 执行 eval / ptq / deploy、adapter 改造（读写文件）
- `quant-reviewer` —— 精度 / 收益验证、与 casebook 对比（只跑评测，不改方案）

## 仓内组织

`.agents/` 是唯一可信源（tracked）：

```
.agents/
├── agents/        # 子代理：quant-analyzer / quant-implementer / quant-reviewer
├── skills/        # quant-workflow / quant-tools / model-adapter / gitcode-pr / gitcode-issue / default-skills / reference
├── docs/          # casebook（L1/L2/L3 三层量化经验）/ architecture.md / repo-map.md
└── settings.json  # Claude Code 权限白名单
```

`scripts/init-agent.sh` 据此生成 `.claude/`、`.opencode/` 客户端视图（均 gitignored；`.opencode/` 另保留手维护的 `opencode.json`、`README.md`）。架构与设计依据见 [docs/architecture.md](docs/architecture.md)。

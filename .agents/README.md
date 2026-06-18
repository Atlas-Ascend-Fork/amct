# amct 大模型量化 Agent Skills

面向昇腾 NPU 大模型量化压缩的 Agent Skills。基于 amct（`amct_pytorch` 包）的 LLM 量化能力（`eval / extract_ptq_data / ptq / deploy` 四步 CLI + blockwise PTQ），把方案选择、算法升级、直转评测、部署导出等流程模块化，让 Agent 以自然语言完成「模型 → 量化方案 → 部署权重」的端到端工作。命令体基于 amct 真实接口（`python -m amct_pytorch.*` + yaml `--bit_config`），同时适配 Claude Code 与 OpenCode。

## 量化能力概览

| 维度 | 取值 |
|---|---|
| 量化目标 | `mlp` / `moe` / `attn-linear` / `attn-cache` |
| 位宽·数据类型 | W8A8 / W4A8 / W4A4；`int` / `mxfp` |
| bit 配置 | yaml `--bit_config`（顶层 `w_bits/a_bits` + `moe.routed/shared` + `attn-cache` 的 q/k/p/v）|
| 算法 | 直转（minmax / smoothquant / awq / gptq 等）、可训练 PTQ（lwc / lac / autoround / omniquant）|
| 粒度 | `block`（默认）/ `model` |
| 评测·阈值 | Wikitext PPL；默认接受阈值 `delta ≤ 0.2` |
| 产物 | compressed-tensors（`config.json` + `model.safetensors.index.json` + 分片权重 + `deploy_quantization.md`）|

## 能力构成

- **编排入口** `quant-workflow`：唯一入口，负责判阶段 → 串联 / 分流 → 复用 casebook → deploy 交付；不重写叶子逻辑。
- **叶子 skill**（由编排调度）：`model-adapter`（新模型适配）、`scheme-recommendation`（方案推荐）、`direct-quant-eval`（直转评测）、`algorithm-recommendation`（算法推荐）、`algorithm-validation`（算法验证）、`deploy-export`（部署导出）。
- **子代理**（按职责委派，经共享状态文件 `progress.md` 传递上下文）：`quant-analyzer`（分析，只读代码）、`quant-implementer`（实施，读写文件）、`quant-reviewer`（审查，只跑评测不改方案）。
- **通用辅助**：`gitcode-pr` / `gitcode-issue`（gitcode 协作）、`default-skills`（按需安装必备 skill）、`reference`（子 skill 共享输入模板）。

## 典型场景 → 对应能力

安装后客户端会自动发现各视图下的 skills 与 agents（**无需手动注册**），并依据每个 skill / agent frontmatter 的 `description` 匹配用户请求自动选用。用自然语言描述诉求即可：

| 典型诉求 | 自动进入 |
|---|---|
| 端到端「把 Qwen3-30B-A3B 量化到 W8A8 并导出权重」 | `quant-workflow`（全程编排，下列各步按需调度）|
| 「这个新模型还没适配」 | → `model-adapter` |
| 「给我第一轮量化方案」 | → `scheme-recommendation` |
| 「跑直转，看 PPL / delta」 | → `direct-quant-eval` |
| 「直转 delta 超 0.2，怎么升级」 | → `algorithm-recommendation` → `algorithm-validation` |
| 「量化达标了，导出部署权重 + 交付文档」 | → `deploy-export` |

**编排顺序**：`model-adapter`（未适配时）→ `scheme-recommendation` → `direct-quant-eval`（`delta ≤ 0.2` 即可导出）→ 超阈则 `algorithm-recommendation` + `algorithm-validation` → `deploy-export`。其中方案选择、PTQ 升级、是否导出等关键决策均会停下与用户确认。

## 案例库与文档

- `docs/casebook/`：按厂商系列（qwen / deepseek / glm / longcat）归档已适配模型的适配要点与实测精度，按「L1 跨网络 / L2 结构家族 / L3 模型专属」三层组织，方案推荐优先复用。
- `docs/architecture.md`：架构与设计依据。`docs/repo-map.md`：仓内代码导航。

## 目录结构

`.agents/` 是唯一可信源（tracked）：

```
.agents/
├── README.md
├── agents/         # 子代理定义：quant-analyzer / quant-implementer / quant-reviewer
├── skills/         # skill 源：quant-workflow / quant-tools / model-adapter /
│                   #           gitcode-pr / gitcode-issue / default-skills / reference
├── docs/           # casebook / architecture.md / repo-map.md / samples
└── settings.json   # Claude Code 权限白名单
```

## 安装

```bash
bash scripts/init-agent.sh            # 两客户端（默认）
bash scripts/init-agent.sh --claude   # 仅 Claude Code
bash scripts/init-agent.sh --opencode # 仅 OpenCode
```

`init-agent.sh` 从 `.agents/` 生成各客户端视图（均 gitignored），投影方式如下：

- **`.claude/`**：`agents/`、`skills/` 为目录，其下对 `.agents/{agents,skills}/` 的每个条目建**逐文件符号链接**；`docs` 是指向 `.agents/docs` 的**整目录符号链接**；`settings.json` 是 `.agents/settings.json` 的**拷贝**。
- **`.opencode/`**：`agents/`、`skills/`、`docs` 同上；另保留**手维护**的 `opencode.json` 与 `README.md`（tracked，非生成）；OpenCode 插件运行时产物（`node_modules` 等）已 gitignore。
- 仓根生成符号链接 `CLAUDE.md` → `AGENTS.md`（`AGENTS.md` 为 tracked 源）。

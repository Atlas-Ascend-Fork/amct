# amct 大模型量化 Agent Skills

面向昇腾 NPU 大模型量化压缩的 Agent Skills 集合。基于 amct（`amct_pytorch` 包）的 LLM 量化能力（`eval / extract_ptq_data / ptq / deploy` 四步 CLI + blockwise PTQ），将量化方案选择、算法升级、直转评测、部署导出等流程模块化，支持 Agent 完成「模型 → 量化方案 → 部署权重」的端到端工作。命令体基于 amct 真实接口（`python -m amct_pytorch.*` + yaml `--bit_config`），同时适配 Claude Code 与 OpenCode 两类客户端。

## 量化能力概览

- **量化目标**：`mlp` / `moe` / `attn-linear` / `attn-cache`
- **位宽 / 数据类型**：W8A8 / W4A8 / W4A4，`int` / `mxfp`；bit 配置经 yaml `--bit_config`（顶层 `w_bits/a_bits` + `moe.routed/shared` + `attn-cache` 的 q/k/p/v）
- **算法**：直转（minmax / smoothquant / awq / gptq 等）与可训练 PTQ（lwc / lac / autoround / omniquant）
- **粒度**：`block`（默认）/ `model`
- **评测**：Wikitext PPL，默认接受阈值 `delta ≤ 0.2`
- **产物**：compressed-tensors 格式（`config.json` + `model.safetensors.index.json` + 分片权重 + `deploy_quantization.md`）

## Skills

量化主流程由 `quant-workflow` 编排入口调度：

| Skill | 功能 |
|---|---|
| [quant-workflow](skills/quant-workflow/SKILL.md) | 编排入口：判阶段 → 串联 / 分流 → 复用 casebook → deploy 交付 |
| [model-adapter](skills/model-adapter/SKILL.md) | 新模型适配（编写 adapter 代码） |
| [scheme-recommendation](skills/quant-tools/scheme-recommendation/SKILL.md) | 首轮量化方案推荐 |
| [direct-quant-eval](skills/quant-tools/direct-quant-eval/SKILL.md) | 直转评测（PPL / delta） |
| [algorithm-recommendation](skills/quant-tools/algorithm-recommendation/SKILL.md) | 直转超阈时的算法推荐 |
| [algorithm-validation](skills/quant-tools/algorithm-validation/SKILL.md) | 验证 PTQ 算法相对直转的收益 |
| [deploy-export](skills/quant-tools/deploy-export/SKILL.md) | 部署导出（compressed-tensors 权重 + 文档） |

通用辅助 skill（非量化主流程）：`gitcode-pr` / `gitcode-issue`（gitcode 协作）、`default-skills`（按需安装必备 skill）、`reference`（子 skill 共享输入模板）。

### 编排流程

`quant-workflow` 按阶段推进：model-adapter（未适配）→ scheme-recommendation（首轮方案）→ direct-quant-eval（直转，`delta ≤ 0.2` 即可导出）→ algorithm-recommendation + algorithm-validation（超阈时升级 PTQ）→ deploy-export（导出产物）。

### Agent 协作

编排入口为 supervisor，按分析 / 实施 / 审查职责委派三类 SubAgent，经共享状态文件 `progress.md` 传递上下文：

| Agent | 职责 |
|---|---|
| [quant-analyzer](agents/quant-analyzer.md) | 适配性分析、方案 / 算法推荐；只读代码，仅写 progress.md |
| [quant-implementer](agents/quant-implementer.md) | 执行 eval / extract / ptq / deploy、adapter 改造、产物导出；读写文件 |
| [quant-reviewer](agents/quant-reviewer.md) | 精度 / 收益验证、与 casebook 对比；运行评测，不改代码与方案 |

## 案例库与文档

- `docs/casebook/`：按厂商系列（qwen / deepseek / glm / longcat）归档已适配模型的适配要点与实测精度，按「L1 跨网络 / L2 结构家族 / L3 模型专属」三层组织，方案推荐优先复用；部分模型仅完成代码适配、实测待补（如 DeepSeek-V4）。
- `docs/architecture.md`：架构与设计依据。`docs/repo-map.md`：仓内导航。

## 目录结构

`.agents/` 为唯一可信源（tracked）：

```
.agents/
├── README.md
├── agents/         # SubAgent 定义：quant-analyzer / quant-implementer / quant-reviewer
├── skills/         # Skill 源：quant-workflow / quant-tools / model-adapter /
│                   #            gitcode-pr / gitcode-issue / default-skills / reference
├── docs/           # casebook / architecture.md / repo-map.md / samples
└── settings.json   # Claude Code 权限白名单
```

## 安装

```bash
bash scripts/init-agent.sh            # 两客户端（默认）
bash scripts/init-agent.sh --claude   # 仅 Claude Code
bash scripts/init-agent.sh --opencode # 仅 OpenCode
```

`init-agent.sh` 由 `.agents/` 生成各客户端视图（gitignored）。`.claude/` 全量生成；`.opencode/` 保留手维护的 `opencode.json` 与 `README.md`，仅生成 `skills / agents / docs`。执行后实际结构：

```
.claude/                            .opencode/
├── agents/      -> .agents/agents  ├── opencode.json   (tracked，手维护)
├── skills/      -> .agents/skills  ├── README.md       (tracked，手维护)
├── docs/        -> .agents/docs    ├── agents/   -> .agents/agents
└── settings.json (copy)            ├── skills/   -> .agents/skills
                                    └── docs/     -> .agents/docs

根 CLAUDE.md -> AGENTS.md
```

> 说明：`agents / skills / docs` 为指向 `.agents/` 的符号链接，`.claude/settings.json` 为拷贝；`.agents/` 不含 `hooks/`，故视图中也无 `hooks/`。
</content>

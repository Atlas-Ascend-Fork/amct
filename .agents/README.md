# amct 大模型量化 Agent Skills

## 项目概述

### 项目定位

面向昇腾 NPU 大模型量化压缩场景的 Agent Skills 集合。基于 amct（AMCT 9.0，`amct_pytorch` 包）的 LLM 量化能力（`eval / extract_ptq_data / ptq / deploy` 四步 CLI + blockwise PTQ），把量化方案选择、算法升级、直转评测、部署导出等知识与流程模块化，支持 Agent 按流程完成「模型 → 量化方案 → 权重产物」的端到端工作。

本能力的命令体基于 amct 真实接口（`python -m amct_pytorch.*` + yaml `--bit_config`），按 Claude/OpenCode 双客户端的 `.agents/` 玩法组织（参考 cann-recipes-infer），随 amct 量化能力落地在上游 `cann/amct`。

### 量化能力概览

- **量化目标**：`mlp` / `moe` / `attn-linear` / `attn-cache`，一次聚焦一个角色
- **位宽与数据类型**：W8A8 / W4A8 / W4A4，`int` / `mxfp`；bit 配置走 yaml `--bit_config`（顶层 `w_bits/a_bits` + `moe.routed/shared` 分组 + `attn-cache` q/k/p/v）
- **算法**：直转（minmax/smoothquant/awq/gptq…）与可训练 PTQ（lwc/lac/autoround/omniquant）
- **粒度**：`block`（主流）/ `model`
- **评测**：Wikitext PPL，默认阈值 `delta ≤ 0.2`
- **产物**：compressed-tensors 格式 `config.json` + `model.safetensors.index.json` + 分片权重 + `deploy_quantization.md`

## Skills 列表

### 量化主流程 Skills（`skills/quant-tools/` + `skills/tasks/`）

由 `amct-quantization` 编排入口调度：

| Skill | 功能 | 触发场景 |
|-------|------|---------|
| [amct-quantization](skills/tasks/amct-quantization/) | 量化适配统一入口（编排） | 端到端量化适配、输入模型输出方案与权重 |
| [model-adapter](skills/tasks/model-adapter/) | 新模型适配 | 模型未适配时写 adapter 代码 |
| [model-workflow](skills/tasks/model-workflow/) | 量化工作流总览 | 了解四步 CLI 的串联与产物流转 |
| [quantization](skills/tasks/quantization/) | 量化实验总控 | 直转→升级 PTQ→导出的实验编排 |
| [scheme-recommendation](skills/quant-tools/scheme-recommendation/) | 量化方案推荐 | 首轮候选方案、复用 casebook、风险/回退 |
| [direct-quant-eval](skills/quant-tools/direct-quant-eval/) | 直转评测 | 跑 PPL，判 delta 是否可接受 |
| [algorithm-recommendation](skills/quant-tools/algorithm-recommendation/) | 算法推荐 | 直转超阈，推荐下一步算法 |
| [algorithm-validation](skills/quant-tools/algorithm-validation/) | 算法验证 | 验证 PTQ 算法相对直转是否真改善 |
| [deploy-export](skills/quant-tools/deploy-export/) | 部署导出 | 导出 compressed-tensors 权重 + 文档 |
| [dynamic-quant-qmm-analysis](skills/quant-tools/dynamic-quant-qmm-analysis/) | Linear dynamic quant + qmm 理论性能评估 | 候选方案启用动态量化时判收益方向 |

### 调度关系

```
amct-quantization（编排入口）
  ├── Step 1: model-adapter（未适配时）
  ├── Step 2: scheme-recommendation（首轮方案，按需 dynamic-quant-qmm-analysis 评估收益）
  ├── Step 3: direct-quant-eval（直转，delta ≤ 0.2 → 直接导出）
  ├── Step 4: algorithm-recommendation + algorithm-validation（直转超阈时升级 PTQ）
  └── Step 5: deploy-export（导出权重产物）
```

### Agent 协作架构

主流程采用「编排 Agent + 三类专业 SubAgent」协作，按分析/实施/验证职责隔离：

| Agent | 角色 | 职责 |
|-------|------|------|
| 主 Agent | 编排者 | 阶段推进、用户确认、进度管理，不直接跑命令改代码 |
| [quant-analyzer](agents/quant-analyzer.md) | 分析专家 | 适配性分析、方案/算法推荐、收益评估；只读代码，只写 progress.md |
| [quant-implementer](agents/quant-implementer.md) | 实施工程师 | 跑 eval/extract/ptq/deploy、adapter 改造、产物导出；读写全部文件 |
| [quant-reviewer](agents/quant-reviewer.md) | 审查专家 | 精度/收益验证、与 casebook 对比；运行评测但不改代码与方案 |

SubAgent 之间通过共享状态文件 `progress.md` 传递上下文。

### 通用辅助 Skills（随仓既有，非量化主流程）

`skills/default-skills/`（会话启动自安装通用 skill）、`skills/gitcode-pr/`、`skills/gitcode-issue/`（gitcode PR / issue 协作）。

## 案例库与文档

- `docs/agent/casebook/<series>/`：7+ 模型实测精度归档（Qwen3 系列 / LongCat / DeepSeek-V4 等，含 BF16/Quant PPL Δ + 部署产物清单），方案推荐优先复用
- `docs/agent/`：`amct-onepage.md`（一页速查）、`amct-10min-quickstart.md`（10 分钟上手）、`amct-quantization-framework-guide.md`（完整框架原理）、`repo-map.md`（仓内导航）、`amct-to-infer-execution-flow.md` + `infer-integration-checklist.md`（交付 infer 推理仓）

## 目录结构

```
.agents/
├── README.md                 # 本文件
├── agents/                   # SubAgent 角色定义（Claude/OpenCode 共用）
│   ├── quant-analyzer.md
│   ├── quant-implementer.md
│   └── quant-reviewer.md
├── hooks/                    # Hook 脚本（可选）
├── settings.json            # Claude Code 设置（权限 + hook 注册）
└── skills/                   # Skills 源（quant-tools / tasks / reference + 通用 skills）
```

`.agents/` 为唯一可信源（tracked）；`.claude/` 与 `.opencode/` 由 `init-agent.sh` 生成（gitignored）。

## 使用方式

```bash
bash scripts/init-agent.sh --claude     # Claude Code 用户
bash scripts/init-agent.sh --opencode   # OpenCode 用户
bash scripts/init-agent.sh              # 两个客户端都装（默认）
```

安装后：`.claude/{skills,agents,hooks,settings.json}` 与 `.opencode/{skills,agents}` 指向 `.agents/` 源，根 `CLAUDE.md -> AGENTS.md`。

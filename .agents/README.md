# amct Agent Skills

amct 仓的 Agent Skills 基础设施（`.agents/`），以 skill 为单元封装各类研发能力，自然语言触发、agent 自主执行，clone 即可用。

当前内置：
- **quant-workflow**：LLM 量化全流程编排主入口（方案推荐 → PTQ → 部署导出，见下文）
- **gitcode-pr / gitcode-issue**：GitCode 仓库协作操作

后续按需扩充。适配 Claude Code 与 OpenCode。

## 快速开始

clone 本仓即可直接使用，Agent Skills 自动加载（机制见 [docs/architecture.md](docs/architecture.md) §3）。

装好后直接用自然语言描述任务，例如：

> 把 Qwen3-30B-A3B 量化到 W8A8 并导出部署权重

客户端据此进入编排入口 `quant-workflow`，自动判阶段、串联子能力，并在方案选择、PTQ 升级、导出等关键决策处与你确认。

> 前置：`amct_pytorch` 可导入；NPU 设备由你的环境提供（命令以 `--device` 指定）；评测数据可达（不可达时可设 `HF_ENDPOINT` 镜像 / 用 modelscope / 指本地路径）。

## 用法：自然语言示例 → 自动进入

无需记命令；客户端按各 skill / 子代理 frontmatter 的 `description` 匹配诉求自动选用。换个说法、换个模型都行，例如：

| 你这样说（示例） | 自动进入 |
|---|---|
| 「把 Qwen3-30B-A3B 量化到 W8A8 并导出部署权重」 | `quant-workflow` 全程编排（含确认门）|
| 「DeepSeek-V2-Lite 还没适配，先接进来」 | `model-adapter` 适配 |
| 「给我这个模型第一轮量化方案」/「W4A8 大概到多少 delta」 | `scheme-recommendation` 方案推荐 |
| 「跑下 W8A8 直转，看 PPL 和 delta」 | `direct-quant-eval` 直转评测 |
| 「直转 delta 超 0.2 了怎么升级」/「要不要上 GPTQ」 | `algorithm-recommendation` → `algorithm-validation` |
| 「PTQ 跑完了，比直转有没有真改善」 | `algorithm-validation` 收益验证 |
| 「达标了，导出权重 + 写交付文档」 | `deploy-export` 导出 + `deploy_quantization.md` |
| 「deploy 目录有了，就差交付文档」 | `deploy-export`（阶段 5 补文档）|
| 「量化后 PPL 比 BF16 还低，正常吗」 | `quant-workflow` 先查链路再判定 |

整句覆盖全流程 → 走完整编排；只说其中一段 → 直接分流到对应叶子，不强行走完整流程。

编排顺序：适配（未适配时）→ 方案 → 直转（`delta ≤ 0.2` 可直接导出）→ 超阈则升级 PTQ → 导出。

## 量化能力

| 维度 | 取值 |
|---|---|
| 量化目标 | `mlp` / `moe` / `attn-linear` / `attn-cache` |
| 位宽·数据类型 | W8A8 / W4A8 / W4A4；`int` / `mxfp` |
| 算法 | 直转（minmax 等）、可训练 PTQ（lwc / lac / autoround / omniquant；其余如 gptq/awq 视分支移植）|
| bit 配置 | yaml `--bit_config`（顶层 `w_bits/a_bits` + `moe.routed/shared` + `attn-cache` 的 q/k/p/v）|
| 评测·阈值 | Wikitext PPL；默认接受 `delta ≤ 0.2` |
| 产物 | compressed-tensors（`config.json` + 分片权重 + `deploy_quantization.md`）|

## 协作结构

编排入口 `quant-workflow` 按职责委派三个子代理，经共享状态文件 `progress.md` 传递上下文：

- `quant-analyzer` —— 适配性分析、方案 / 算法推荐（只读代码）
- `quant-implementer` —— 执行全部量化命令（统一走 `quant-run`：直转评测 / 校准 / ptq / 导出）、adapter 改造（读写文件）
- `quant-reviewer` —— 精度 / 收益判读、与 casebook 对比（只读 `quant-run` 跑出的结果判定，不跑评测/ptq、不改方案）

## 仓内组织

`.agents/` 是唯一可信源（tracked）：

```
.agents/
├── agents/        # 子代理：quant-analyzer / quant-implementer / quant-reviewer
├── skills/        # quant-workflow / quant-tools（叶子含 model-adapter + references 共享）/ gitcode-pr / gitcode-issue / default-skills
├── docs/          # casebook（L1/L2/L3）/ architecture.md / repo-map.md
├── hooks/         # pre_tool_use / subagent_stop（Claude Code）
├── settings.json  # Claude Code 配置（权限 + hooks）
└── opencode.json  # OpenCode 配置（plugins + 权限）
```

`.claude/`、`.opencode/` 是真实目录；其中 **配置文件**（`settings.json`、`opencode.json`）为真实文件，**内容目录**（skills/agents/hooks/docs）为 git-tracked symlink 指向 `.agents/`，随 clone 自动生效。架构与设计见 [docs/architecture.md](docs/architecture.md)。

> **Windows 注意**：`settings.json`/`opencode.json` 已是真实文件，Windows 下客户端可正常启动。内容目录（skills/agents/hooks/docs）仍为 symlink，Windows 默认（`core.symlinks=false`）clone 后会退化为文本文件，skills 不会自动加载。需开启开发者模式，再 `git config core.symlinks true` 后重新 checkout 即可恢复。主场景为 Linux 服务器，Windows 按需处理。

## 面向 agent 集成

多 agent / 上游集成只需对接唯一入口 `quant-workflow`（黑盒）；契约（输入 / 状态面 / 确认门 / 前置）见 [docs/architecture.md](docs/architecture.md) 第 8 节。

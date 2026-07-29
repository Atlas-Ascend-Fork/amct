# amct-coder Skill 测试报告

**测试范围：** 路由触发、越层拦截、委派边界、多子能力联动


## 一、当前测试通过（4 项）

### 测试 1：新增 fp4 dtype（正常路由）

| 项目 | 内容 |
|---|---|
| **测试输入** | 在 `amct_pytorch/quantization/dtypes` 里新增一个 fp4-like dtype，并接入 CLI choices |
| **期望命中** | `amct-coder` / `dtype-quant-module` |
| **实际命中** | ✅ `amct-coder` / `dtype-quant-module` |
| **期望行为** | 读 dtype playbook；补读 public-api-tests；最后读 validation.md |
| **实际行为** | ✅ 读了 `dtype-quant-module.md`、`public-api-tests.md`、`validation.md`，输出了 6 步可执行操作 |
| **验证结论** | ✅ 通过 |

**验证了：** 非模型层改动正确路由；跨子能力补读生效；读取预算 3 个 reference 未超。


### 测试 2：solver 里按模型名写分支（越层拦截）

| 项目 | 内容 |
|---|---|
| **测试输入** | 在 `common/optimization/blockwise_solver.py` 里按 DeepSeek 模型名写特殊分支 |
| **期望命中** | `amct-coder`，报越层 BLOCKER |
| **实际命中** | ✅ `amct-coder`，识别为越层 BLOCKER |
| **期望行为** | 报 BLOCKER；solver 不应出现模型名分支；模型差异应在 adapter/PtqUnit/block 层 |
| **实际行为** | ✅ 第一次回答凭记忆给出结论（行为正确，依据缺失）；修正后引用了三处 Skill 规则覆盖此场景 |
| **验证结论** | ✅ 通过 |

**验证了：** 越层拦截生效；solver 层职责边界清晰；正确引导到 adapter 层。


### 测试 3：新增 classic graph_based optimizer pass（正常路由）

| 项目 | 内容 |
|---|---|
| **测试输入** | 新增 classic graph_based 的 optimizer pass |
| **期望命中** | `amct-coder` / `classic-graph` |
| **实际命中** | ✅ `amct-coder` / `classic-graph` |
| **期望行为** | 不套用 LLM ALGO_REGISTRY，走 classic/graph_based 体系 |
| **实际行为** | ✅ 读了 `classic-graph.md`、`validation.md`；输出了 7 步路线，包含代码锚点、config 接入、测试、public API 懒加载边界 |
| **验证结论** | ✅ 通过 |

**验证了：** classic 和 LLM 两套体系被区分；正确识别 classic 代码位置和配套更新；public API 懒加载边界识别。


### 测试 4：CLI 里 import 模型名分流（越层拦截）

| 项目 | 内容 |
|---|---|
| **测试输入** | 在 `cli/llm/args.py` 里 import Qwen3Model，根据模型名分流参数 |
| **期望命中** | `amct-coder`，报越层 BLOCKER |
| **实际命中** | ✅ `amct-coder`，报越层 BLOCKER |
| **期望行为** | CLI 层不能出现模型类名 import 和按模型名的 if 分支；模型专属分流应由 adapter 提供统一抽象 |
| **实际行为** | ✅ 引用了 `layer-rules.md` 第 44 行和全局 BLOCKER；没有给实现步骤，直接停下，给出正确落地方向 |
| **验证结论** | ✅ 通过 |

**验证了：** CLI 层越层拦截生效；输出没有继续实现错误方案。


## 二、待测试
还需要构造更多的测试样例来进行检测

| # | 测试场景 | 期望 | 原因 |
|---|---|---|---|
| 1 | 在 `quantization/modules/quant_linear.py` 里判断 MoE 结构，给 Qwen 单独处理 | 命中 amct-coder，报越层 BLOCKER | 待测 |




## 三、测试中暴露的问题与修正

| 问题 | 说明 | 修正 |
|---|---|---|
| **回答未加载 Skill** | `blockwise_solver` 测试中，AI 凭记忆回答，结论对但依据缺失 | 修正回答中引用了三处 Skill 规则（全局边界、越层请求对照表、全局 BLOCKER） |
| **推理机制不完整** | AI 可能跳过了"先做三件事"的第 2 步（读命中的 playbook） | 后续测试中 AI 主动加载了 `classic-graph.md` 和 `layer-rules.md` |


## 四、验收标准达成情况
文档规定测试通过的情况

| 验收项 | 目标 | 当前状态 |
|---|---|---|
| 越层场景拦截率 | ≥5 个越层场景，100% 拦截 | 2/5 已测，100% 通过 |
| 合法改动放行率 | ≥3 个合法改动，≥90% 放行 | 2/3 已测（dtype、classic-graph），100% 通过 |
| description 未被关键词带偏 | dtype 命中 dtype，workflow 加 attention 命中越层 | 已验证 dtype 场景；workflow 场景待测 |
| 读取预算 | 默认上限 3 个 reference | dtype 测试恰好 3 个；classic-graph 测试 2 个 |


## 五、下次测试建议

### 1. 继续越层拦截用例
- 在 `workflows/llm_ptq.py` 里加 attention 计算
- 在 `quant_linear.py` 里判断 MoE 结构

### 2. 委派边界用例
- 给 Qwen3-Next 新增 attention wrapper → 期望转 `model-adapter`

### 3. 端到端触发
- 在有 opencode 凭证的环境跑通触发测试，验证 description 在实际客户端中的匹配行为
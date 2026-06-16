# DeepSeek-V2-Lite：量化 Agent 自主适配样例（实测记录）

**日期**：2026-06-16
**结论**：✅ 通过。量化 agent 能把一个**未适配的新模型**端到端适配 + 量化达标。

## 1. 目的与方法

- **目的**：用一个 `register_llm_models()` 中**不存在**的新模型（DeepSeek-V2-Lite，casebook 曾因"未适配"移除）实测 agent 真实能力 —— 不是跑已知样例，是从零适配。
- **运行**：opencode 1.14.19 + `alibaba-cn/glm-5`，加载本仓 `.agents` 的 quant-workflow skill + 三层 casebook。模式 = **agent 自主跑、卡住人接管**（human-in-the-loop）。
- **模型**：DeepSeek-V2-Lite（~16B，2.4B active；MLA + DeepSeekMoE，**无 DSA Indexer**，`q_lora_rank=null`）。

## 2. 结果（real weights，Wikitext PPL seq_len=4096）

| 步骤 | 结果 |
| --- | --- |
| BF16 baseline | **9.103** |
| 关闭量化等价 attn-linear / moe（bits=16） | 均 **9.103**（逐位=BF16）|
| W8A8 attn-linear+moe 直转 | **9.2374**，`delta=0.134` 达标 |

PPL 9.10 量级合理（小 active MoE），且关闭量化逐位等于 BF16 → **agent 写的 MLA modeling（RoPE/YaRN/mscale）数值忠实**。

## 3. Agent 自主能力（强）

agent 在无人干预下正确完成：
- 三层 casebook 检索：读 L1/L2(MLA+MoE)/L3(deepseek-v3.2)，判 **V2-Lite≈v3.2、差异=无 Indexer**。
- 结构要点：**`q_lora_rank=null` 非压缩 Q 分支**（没盲抄 v3.2 压缩 Q）、`first_k_dense_replace=1`、64+2 expert。
- 落地 adapter（deepseekv2.py/quant_module.py/modeling/）+ 注册 `DeepseekV2` + 空载 smoke 6/6。
- **自主调试修了 4 个自己的 bug**：上游 modeling import 错、`base_prefix`、`QuantDeepseekV2MLP` 缺 `hidden_size`、`freqs_cis` buffer。
- 守 supervisor 流程，主动停在卡点①（直转方案）等用户确认。

## 4. 暴露的问题与处置

**A. Agent 基础设施 bug（已修，提交在 feat/quant-agent-migration）**
1. casebook 三层路径漏 `.agents/` 前缀（消费侧 quant-workflow/quant-analyzer）→ 修。
2. `init-agent.sh` 没投影 `docs/` → skill 里 `../docs/repo-map.md` 等链接在 `.opencode/.claude` 投影下断裂 → 加 `{base}/docs` 符号链。
3. opencode 沙箱默认挡 model/work 外部目录 → `.opencode/opencode.json` 放行 `/mnt/data/**`（本地 env）。

**B. 环境（人接管，非 agent 逻辑）**
- Wikitext egress 被墙 → `get_wikitext2` 本地 parquet fallback（属 offline-dataset-fallback 关注点，本次仅本地 env，不进 MR）。
- NPU HBM 被占满（共享机，孤儿显存）→ 换空闲卡（npu:12-15）。

**C. 既有代码潜伏 bug（本次实测顺带发现，已修）**
- `deepseek_v3_2.py` 的 `get_embed_load_specs` 用了**全仓未定义**的 `self.base_prefix`（应为 `self._embed_base_prefix()`）。glm5 继承 v3.2，同样受影响。**确定性必崩**（agent 抄了这行，v2 真跑时崩 `AttributeError: no attribute 'base_prefix'` 实证），只因 v3.2/glm5 在本仓"从没真跑过"（casebook 标待补）而潜伏。修复 = 1 行，已应用 v3.2（glm5 经继承一并修）。

## 5. 收编

- `deepseek_v2/` adapter 收编：清理 agent 的运行期 hack —— `empty_weights_model` 改为**实例化仓内已注册 `DeepseekV2ForCausalLM`**（`init_empty_weights`），不再依赖覆盖权重目录的上游 modeling（验证：移开权重目录 modeling 后 BF16 仍 9.103）。
- 新增 casebook L3 `deepseek/deepseek-v2.md`（含 q_lora=null / empty_weights 两条专属陷阱 + 精度）。

## 6. 总评

agent 把全新模型从"读 config"到"W8A8 量化达标"基本自主走通；人接管的多是环境（数据集墙、NPU 占用）与一处既有代码潜伏 bug。**casebook 三层 + supervisor + human-in-the-loop 在真实新模型上验证有效。**
</content>

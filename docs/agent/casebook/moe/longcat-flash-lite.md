# LongCat-Flash-Lite

## 为什么这个案例重要

- 这是仓库里第一个 LongCat 系列适配案例。
- 这类模型的输入路径不是普通的 token embedding，而是依赖 `ngram_embeddings`。
- 每个 decoder block 同时包含双 self-attn 分支、双 dense MLP 分支，以及一条 packed MoE shortcut 路径。

## 参考关系

- 所属系列：LongCat
- 复用的仓内抽象：`BaseModel`、`PtqUnit`、`QuantLinear`、`QuantGatedMLP`
- 最接近的参考思路：Qwen 的 dense/MoE wrapper，但 LongCat 仍然需要自己的 attention 和 packed expert wrapper

## 适配结论

- 复用了这些已有抽象：
  - `BaseModel` 的 blockwise 工作流
  - `PtqUnit` 对 `attn / mlp / moe` 的路由
  - `QuantLinear` 和 `QuantGatedMLP`
- 新增的最小模型专属实现：
  - `QuantLongcatMLA`
  - `LongcatPackedExperts`
  - `LongcatPackedExpertView`
  - LongCat 专属的 embedding 加载和 expert 重组逻辑
- 风险最高的区域：
  - 输入路径和 block 拓扑都比较特殊，所以必须做源码公式重放和逐层 stepwise 校验
- 最终验证状态：
  - BF16 baseline：通过
  - 关闭量化后的浮点等价：通过
  - 最小 PTQ unit smoke：通过

## 精度与验证

- BF16 baseline：
  - Wikitext PPL，blockwise，`seq_len=2048`
  - `ppl_bf16 = 12.910730361938477`
- stepwise / 源码路径验证：
  - `ngram_embeddings` 的源码公式重放与适配器路径一致
  - blockwise 输出逐层对齐 through 全部 decoder stack
  - 最后的 `norm + lm_head` logits 也一致
- 挂上 Quant wrapper 但关闭量化后的整链验证：
  - 全量 Wikitext PPL 仍然是 `12.910730361938477`

## 量化结论

- 第一轮直转量化方案：
  - Wikitext PPL，blockwise，`seq_len=2048`
  - `quant_target = attn mlp moe`
  - `quant_dtype = int`
  - `a_bits = 8`
  - `w_bits = 8`
- BF16 baseline 复用说明：
  - 模型、代码路径和评测口径都没有变化，所以直接复用适配阶段拿到的 BF16 baseline
- 直转量化结果：
  - `ppl_quant = 12.9460`
  - `delta = 0.0353`
- PTQ 决策：
  - 第一轮不需要进入 PTQ，因为 `delta <= 0.2`
- 最终采用方案：
  - `attn + mlp + moe` 的直转 `a8w8-int8`

## 典型问题

- 问题：
  - Ascend 运行时在量化 block 执行前初始化失败
- 现象：
  - `_npu_init()` 报 `SetPrecisionMode` 和 AOE/TBE 初始化错误
- 根因：
  - `/usr/local/Ascend/ascend-toolkit/set_env.sh` 没有导出这条运行路径所需的全部编译侧环境变量
- 修复：
  - 导出 `ASCEND_COMPILER_PATH=$ASCEND_TOOLKIT_HOME/compiler`
  - 导出 `TBE_IMPL_PATH=$ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe`
  - 导出 `AOE_DIR=$ASCEND_TOOLKIT_HOME/tools`
- 后续建议：
  - 大任务前先跑一次最小的 `torch.ones(...).npu()` smoke

- 问题：
  - int activation quant 路径在第一个 MLP block 就报错
- 现象：
  - `weight_quant()` 收到了 activation tensor，并触发 shape 断言
- 根因：
  - `ActivationQuantizer` 实例化 dtype 时没有传 `is_act=True`，同时 `dynamic_per_token_quant()` 错误地用了只适用于正数的 clamp
- 修复：
  - activation quantizer 传入 `is_act=True`
  - 去掉 signed activation quant 里的正数 clamp
- 后续建议：
  - 不要一上来跑全模型，先 smoke 一层 quant block

## 最终建议

- 下一个 LongCat 系列模型，优先从这个案例和 LongCat 系列 README 开始。
- 先验证自定义输入路径，再验证逐层 block 等价，最后再跑直转量化。
- 在做第一轮 `a8w8-int8` 之前不要急着上 PTQ，因为这个系列在最简单方案下已经能保持近乎无损。

## 第二轮量化结论(2026-05-12, mlp+moe 直转 a8w8-int8)

- 量化方案变更原因:
  - 业务侧调整,Longcat-Flash-Lite 量化范围由全量 `attn+mlp+moe` 收窄为仅 `mlp+moe`
- 第二轮直转量化方案:
  - Wikitext PPL,blockwise,`seq_len=4096`(SKILL `quantization` 强约束)
  - `quant_target = mlp moe`
  - `quant_dtype = int`
  - `a_bits = 8`,`w_bits = 8`
  - 无 algos,纯直转
- BF16 baseline 重跑说明:
  - 旧 baseline `12.9107` 来自 `seq_len=2048`,与新口径 `seq_len=4096` 不可比;按 SKILL 强约束重跑
- 评测结果:
  - `ppl_bf16 = 14.6243`
  - `ppl_quant = 14.8006`
  - `delta = 0.1763`(≤ 阈值 0.2,达标)
- PTQ 决策:
  - 不需要进入 PTQ,因为 `delta <= 0.2`
- 最终采用方案:
  - `mlp + moe` 的直转 `a8w8-int8` @ `seq_len=4096`
- Deploy 导出:
  - 输出目录 `<deploy artifact: A8W8 INT8 MLP+MoE (seq4096)>`
  - 关键产物:14 个 `layer_*.safetensors` + 1 个 `rest_00000.safetensors` + `config.json`(刷新 `quantization_config`,`format=int-quantized`,`group_0.targets=["Linear"]`)+ `model.safetensors.index.json` + `deploy_quantization.md`(由 deploy 主流程自动生成)

## 性能评估(prefill, M=4096)

按 SKILL `dynamic-quant-qmm-analysis` 真机 profiler 实测(`validate_qmm.py`,80 iter/30 warmup,数据见 `<perf data: prefill M=4096>`),得出本次方案 vs 旧方案的 prefill 性能对比结论:

| 方案 | 含义 | 每层耗时 (us) | vs BF16 |
|---|---|---|---|
| BF16 基线 | 不量化 | 16411.5 | 1.00× |
| **A. mlp+moe(第二轮)** | attn 留 BF16 | **28146.9** | **0.583×**(−71%)⚠️ |
| **B. attn+mlp+moe(第一轮)** | 全量化 | **28147.4** | **0.583×**(−71%) |
| Alt. 只 mlp | — | 15258.6 | **1.076×**(+7%) ✓ |

主要结论:
- **A 与 B 在 prefill 下性能差异 < 0.5 us/层**:attn 内部的 5 个 Linear,正负 speedup 互相抵消(`group speedup ≈ 1.000`),量化 attn 既不增益也不损益
- **两方案都是负收益**:MoE 单 expert M=384 太小,`dynamic_quant + qmm_fixed + antiquant` 开销 > BF16 matmul,且 MoE 每层 ×128 被放大(0.43× speedup)
- **MLP 是唯一明显正收益组**(+7%/层)

负收益主因:
- MoE per-expert: small-M(M=384 远低于 cube tile 阈值),INT8 路径主导项不再是 cube,而是 dq + qmm 头开销
- attn.kv_a(N=576)/ kv_b(K=512):维度过小,L1→L0 搬运效率被破坏
- mlp / attn 大 shape: cube-bound,INT8 cube 2× 体现为 ~1.4× speedup

关键假设(影响结论):
1. MoE 部署路径为 per-expert 独立 dynamic_quant + Linear QMM;若 inference 仓有 grouped MoEGMM-INT8,M_eff 可放大到 49152,MoE 可能反转为正收益 — 注意 `generate_quant_config` 的 INT 路径 `config_groups` 只标 `targets=["Linear"]`,**未声明 `MoEGMM`**(mxfp 路径才有),这是 INT 部署可能走 per-expert 的强信号
2. Prefill 视角 M=4096;Decode(M=1)未测,decode 时所有 Linear weight-bound,INT8 权重 ½ 大概率全正收益,可能整体翻转
3. 平台假设按 A2/A3 bucket;若实际 SKU 是 910B3/B4 之一,需单独确认

详细评估卡见 `<perf evaluation card>`。

## 量化经验补充

- 第一轮的 `attn+mlp+moe`(seq_len=2048)与第二轮的 `mlp+moe`(seq_len=4096)在 prefill 性能上**几乎无差**;选 attn 进/不进量化,首要决策依据应当是 **精度 delta** 与 **下游 infer 路径覆盖**,不是 prefill 性能
- 在 prefill 视角下,**MoE 量化不一定是性能加分项**,需要先确认 inference 仓的 MoE INT 路径
- 同一系列下次再做直转量化时,建议先用 `dynamic-quant-qmm-analysis` 跑一遍方案推荐前的可行性,避免在精度过关但性能反向的方案上走完整 pipeline

## 精度速查表

> ppl 口径 seq_len=4096（2048→4096 后精度反而变差）。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 14.6133 |
| INT | A4w4: mlp, moe, attn-linear | 无 | 1053.832 |
| INT | A8w4: mlp, moe, attn-linear | 无 | 13.1681 |
| INT | A8w8: mlp, moe, attn-linear | 无 | 14.0334 |
| MXFP | A4w4: mlp, moe, attn-linear | 无 | 13.2715/12.283 |
| MXFP | A8w4: mlp, moe, attn-linear | 无 | 10.4672/10.061 |
| MXFP | A8w8: mlp, moe, attn-linear | 无 | 14.8785/15.091 |
| HiFP | A4w4: mlp, moe, attn-linear | 无 | 14.258 |
| HiFP | A8w4: mlp, moe, attn-linear | 无 |  |
| HiFP | A8w8: mlp, moe, attn-linear | 无 |  |

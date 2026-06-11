---
name: amct-quantization
description: "amct 量化适配 subagent。输入需要量化的模型（含融合算子兼容性报告作为参考），输出量化方案和权重。量化方案决策需要用户确认后再执行。"
---

# amct 量化适配 Subagent

这个 skill 是 amct 的统一入口，用于将整个量化工作流封装为可被外部调用的 subagent。

## 输入

用户需要提供：
- **模型路径**：`--model /path/to/model`
- **模型名称**：`--model_name qwen3_5`（可选，用于方案推荐）
- **融合算子分析报告**（可选，作为量化方案选择的参考输入）
- 其他可选参数

## 工作流程

```
用户输入（模型 + 融合算子报告作为参考）
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: 检查模型是否已适配                          │
│         - 已适配 → 进入 Step 2                      │
│         - 未适配 → 调用 $model-adapter               │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: 方案推荐 (调用 $scheme-recommendation)     │
│         - 输出多个候选方案                           │
│         - 综合融合算子兼容性作为方案选择参考         │
│         - 用户选择方案                               │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: 直转量化评测 (调用 $direct-quant-eval)      │
│         - 产出 ppl_bf16 / ppl_quant / delta         │
│         - delta ≤ 0.2? → 可接受 → 进入 Step 5       │
│         - delta > 0.2? → 不可接受 → 进入 Step 4     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: 算法推荐 (调用 $algorithm-recommendation)  │
│         - 推荐下一步算法                             │
│         - 用户选择是否继续                           │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Step 4b: 算法验证 (调用 $algorithm-validation)      │
│         - 验证用户选择的算法                         │
│         - 产出新 delta                               │
│         - 用户选择是否导出                           │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Step 5: Deploy 导出 (调用 $deploy-export)           │
│         - 导出量化权重                               │
│         - 输出下游可接入的部署目录                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
完成
```

## Skill 调用关系

```
$amct-quantization (本 skill)
    │
    ├── $model-adapter (新模型适配)
    │
    └── $quantization (量化实验总控)
            │
            ├── $scheme-recommendation (方案推荐)
            │       └── 输入包含融合算子兼容性作为参考
            ├── $direct-quant-eval (直转评测)
            ├── $algorithm-recommendation (算法推荐)
            ├── $algorithm-validation (算法验证)
            │
            ▼
            $deploy-export (导出)
```

## 融合算子兼容性（作为参考输入）

### 输入作用

融合算子分析报告作为**方案选择的参考输入**，不单独评估，不增加用户确认环节。

在方案推荐时，综合考虑：
- 融合算子生效情况
- 与目标量化模式的潜在兼容性
- 作为方案推荐的辅助信息

### 输入格式

```json
{
  "session_and_graph_id_0_0": {
    "graph_fusion": {
      "RefreshInt64ToInt32FusionPass": {
        "effect_times": "1",
        "match_times": "1"
      },
      "TransdataFz2FzgFusionPass": {
        "effect_times": "1",
        "match_times": "3"
      }
    }
  }
}
```

### 方案推荐时的参考逻辑

- **Pass 生效率高**（>80%）：可按标准推荐方案执行
- **Pass 生效率中等**（50%-80%）：方案推荐时适当保守
- **Pass 生效率低**（<50%）：方案推荐时优先保守方案

## 决策点（需要用户确认）

1. **方案选择**：scheme-recommendation 输出候选方案后，用户选择
2. **是否升级 PTQ**：直转 delta > 0.2 时，询问是否继续升级
3. **算法选择**：algorithm-recommendation 输出候选后，用户选择
4. **是否导出**：算法验证后，用户确认是否导出权重

## 输出

完成量化后，输出：
- 量化方案详情
- 评测结果：`ppl_bf16 / ppl_quant / delta`
- 导出目录路径
- 下游可消费的交付物：`config.json` / `model.safetensors.index.json` / `deploy_quantization.md`
- 下一步建议（如果有）

## 使用示例

```bash
# 基本使用
python -m amct_pytorch.eval --model /path/to/model --model_name qwen3_5 --device npu:0

# 指定量化方案
python -m amct_pytorch.eval --model /path/to/model --model_name qwen3_5 \
  --quant_target mlp --quant_dtype int --bit_config amct_pytorch/configs/w8a8.yaml --eval_mode quant

# PTQ 训练
python -m amct_pytorch.ptq --model /path/to/model --model_name qwen3_5 \
  --quant_target mlp --quant_dtype int --bit_config amct_pytorch/configs/w8a8.yaml

# Deploy 导出
python -m amct_pytorch.deploy --model /path/to/model --model_name qwen3_5 \
  --quant_target mlp --quant_dtype int --bit_config amct_pytorch/configs/w8a8.yaml \
  --output_dir ./deploy_out
```

## 注意事项

- 融合算子分析报告作为**方案选择的参考输入**，不单独评估
- 量化方案决策需要用户参与确认（选择具体方案）
- 默认使用 Wikitext PPL 作为评测指标
- 默认 threshold 为 delta ≤ 0.2
- 默认 granularity 为 block
- 当前支持的目标：mlp / moe / attn-linear / attn-cache

# AMCT 结构化剪枝（Pruning）

## 1 剪枝前提

### 1.1 安装依赖

本 sample 依赖包可参考 [requirements.txt](requirements.txt)。

需要注意的是 torch_npu 包版本需要与 Python、torch 包版本相匹配，需要安装 CANN 包；
若仅在 CPU 环境下做剪枝离线流程，可以不安装 torch_npu。

### 1.2 模型和数据集准备

本 sample 提供两条示例：

| 脚本 | 模型 | 数据集 | 说明 |
|------|------|--------|------|
| [run_distilbert_boolq_prune.py](src/run_distilbert_boolq_prune.py) | DistilBERT (SequenceClassification) | BoolQ | 稠密层 low_variance 50% 剪枝 |
| [run_qwen_prune.py](src/run_qwen_prune.py) | Qwen2-7B | pile-val | `PrunePass` + `InsertQuantizeModulePass`（AWQ-int4）串接 |

模型需要用户自行下载并通过 `--model_path` 指定本地路径；数据集为在线加载。

### 1.3 简易剪枝配置

本 sample 中使用的剪枝配置内置在工具中，可通过下述方式获取并使用：

```python
from amct_pytorch import (
    CNN_VARIANCE_PRUNE_CFG,
    DENSE_LOWVAR_PRUNE_CFG,
    MOE_MASSVAR_PRUNE_CFG,
    FULL_STRUCTURED_PRUNE_CFG,
)
```

支持的领域 / 方法 / 关键字段：

| 领域 | 方法 | 关键 kwargs | 说明 |
|------|------|-------------|------|
| `cnn` | `tp_variance_channel` | `prune_ratio` | 基于激活方差的通道剪枝（Conv→BN→Conv / Conv→flatten→Linear 自动联动） |
| `dense` | `low_variance` | `prune_ratio` | 稠密 MLP 低方差神经元裁剪 |
| `moe` | `activation_count` | `prune_ratio` | 保留高频激活专家 |
| `moe` | `mass_variance` | `prune_ratio`, `boundary` | 浅层用权重质量、深层用路由方差 |

完整字段说明参见 [`amct_pytorch/pruning/config.py`](../../../amct_pytorch/pruning/config.py)。

## 2 剪枝示例

### 2.1 使用接口方式调用 —— DistilBERT + BoolQ

**step 1.** 请在当前目录执行如下命令运行示例程序：

```bash
python3 src/run_distilbert_boolq_prune.py \
    --model_path=/data/distilbert-base-uncased-qa-boolq \
    --n_samples=128 \
    --save_dir=./pruned_out
```

若出现如下信息，则说明剪枝成功：

```none
Prune done: params XXXX -> YYYY (reduction 30.00%)
Saved pruned model to ./pruned_out
```

### 2.2 剪枝 + 量化串接 —— Qwen2-7B (NPU)

**step 1.** 请在 NPU 环境下执行：

```bash
python3 src/run_qwen_prune.py --model_path=/data/Qwen2-7b/
```

脚本会按下列顺序对模型做处理（均在同一 `ModelOptimizer` 流水线中）：

1. `PrunePass`：在 dense / moe 域分别剪 30%
2. `amct.quantize(..., INT4_AWQ_WEIGHT_QUANT_CFG)`：插入 AWQ-int4 量化算子
3. `amct.convert(...)`：转换为部署模型

成功后将看到：

```none
Calibration time taken:  X min  Y s
Done: pruned + INT4-AWQ Qwen ready for NPU deploy.
```

## 3 接口概览

| 接口 | 等价于量化 | 说明 |
|------|------------|------|
| `amct.prune(model, cfg, data=...)` | `amct.quantize(model, cfg)` | in-place 修改模型，返回 None |
| `amct.get_last_report()` | — | 获取上一次 `amct.prune(...)` 的 `PruneReport`（参数量、警告等） |
| `amct.PrunePass(cfg, data=...)` | `InsertQuantizeModulePass(layer_cfg)` | 可被 `ModelOptimizer.add_pass(...)` 加入流水线；其 `last_report` 属性可读 |
| `amct.FULL_STRUCTURED_PRUNE_CFG` | `INT8_MINMAX_WEIGHT_QUANT_CFG` 等 | 预定义全栈剪枝配置 |

## 4 模型加载

`HuggingFaceSequenceClassificationAdapter` / `ModelScopeSequenceClassificationAdapter` 已下线 —
模型下载与本地加载请通过 `examples/algorithms/prune/src/utils.py` 中的 `get_seqcls / get_causal_lm`
按 AMCT 约定完成。

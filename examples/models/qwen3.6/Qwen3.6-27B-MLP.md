# Qwen3.6-27B Dense A4W4 MXFP4 MLP 量化实践 CaseBook

## 1. 实践概述

本实践基于 AMCT PyTorch，对 Qwen3.6-27B Dense 模型执行
A4W4、MXFP4、MLP 训练后量化，并比较不同 PTQ 算法的
量化精度、运行耗时、Deploy 结果和 clean reload 完整性。

| 配置项 | 本次配置 |
|---|---|
| 模型 | Qwen3.6-27B Dense |
| 模型层数 | 64 |
| 位宽 | A4W4 |
| 数据格式 | MXFP4 |
| 量化目标 | MLP |
| 量化粒度 | Block |
| Block 范围 | `[0, 64)` |
| 校准数据 | PileVal |
| 精度评测数据 | WikiText2 test |
| 核心精度指标 | Perplexity（PPL） |
| AMCT 内部适配名称 | `qwen3_5` |

本轮正式执行的量化方案包括 Direct、LWC、LAC、LWC+LAC、
OmniQuant、AutoRound 和 FlatQuant。
## 2. 校准与 PTQ 配置

### 2.1 基础配置

本轮实践针对 Qwen3.6-27B Dense 模型的全部 64 层 MLP 模块执行 A4W4 MXFP4 量化。

| 配置项 | 配置值 |
|---|---|
| 模型 | Qwen3.6-27B Dense |
| `architectures` | `Qwen3_5ForConditionalGeneration` |
| `model_type` | `qwen3_5` |
| AMCT 模型适配名 | `qwen3_5` |
| 量化位宽 | A4W4 |
| 数据格式 | MXFP4 |
| 量化目标 | MLP |
| `quant_target` | `mlp` |
| 量化粒度 | block |
| Block 范围 | `[0, 64)` |
| 模型层数 | 64 |
| Bit 配置 | `amct_pytorch/configs/w4a4.yaml` |
| 校准数据 | PileVal |
| 评测数据 | WikiText2 test |
| `nsamples` | 16 |
| `seq_len` | 1024 |
| `epochs` | 15 |

`qwen3_5` 是 Transformers 配置和 AMCT 模型适配代码使用的内部架构标识，本轮实际模型为 Qwen3.6。

### 2.2 MLP 校准数据

正式校准目录：

```text
outputs/formal/extract_mlp_n16_s1024_v2
```

采集结果：

| 指标 | 结果 |
|---|---:|
| `EXIT_CODE` | 0 |
| `ELAPSED_SECONDS` | 4013 |
| `BLOCK_COUNT` | 64 |
| 目录大小 | 约 11G |

### 2.3 PTQ 参数目录与 CLI 接口

各算法生成的 MLP PTQ 参数位于以下目录模式：

```text
outputs/formal/<algo>_a4w4_mxfp4_mlp_full64/ptq_params/qwen3_5/mlp
```

虽然本轮处理对象是 Dense 模型中的 MLP，但当前经过验证可用的 AMCT CLI 参数入口仍为：

```text
--moe_mlp_param_dir
```
Eval 和 Deploy 必须继续使用 `--moe_mlp_param_dir` 指定 MLP PTQ 参数目录。

### 2.4 Direct 参数规则

Direct 表示不启用额外优化算法的直接量化基线。

因此，Direct 的 Eval 和 Deploy 命令需要省略 `--algos` 参数。错误传入：

```text
--algos direct
```

会触发：

```text
KeyError: 'direct' is not registered in 'algo'
```

### 2.5 算法范围

本轮已完成以下七种正式方案：

- Direct
- LWC
- LAC
- LWC+LAC
- OmniQuant
- AutoRound
- FlatQuant

## 3. 流程与指标说明

### 3.1 实验流程

本轮正式流程为：

```text
PileVal 校准数据采集
→ 全 64 层 MLP PTQ
→ WikiText2 PPL Eval
→ Deploy 模型导出
→ clean reload 完整性检查
```

MLP 每层包含三个目标 Linear：

- `gate_proj`
- `up_proj`
- `down_proj`

因此，全 64 层量化后应包含：

```text
64 × 3 = 192
```

个量化模块，第 0 层应包含 3 个量化模块。

### 3.2 PPL 指标

WikiText2 test PPL 用于衡量模型在当前语言建模评测配置下的困惑度。PPL 越低，表示模型在该数据集和当前评测流程下的预测结果越好。

相对 BF16 的变化定义为：

```text
当前算法 PPL - BF16 PPL
```

相对 Direct 的恢复量定义为：

```text
Direct PPL - 当前算法 PPL
```

恢复量为正，表示当前算法 PPL 低于 Direct；恢复量为负，表示当前算法 PPL 高于 Direct。

PPL 结果只反映当前模型、WikiText2 test 数据集和当前评测配置，不能直接推广为算法在其他模型、数据集或任务上的一般精度结论。

### 3.3 Eval 耗时

完整 WikiText2 Eval 流程包括：

```text
Chunking
→ Embedding Processing
→ Block Processing
→ Head Processing
→ Evaluating
→ 输出 PPL
```

本文记录的 Eval 耗时是上述完整流程的端到端时间，不等同于在线推理 tokens/s、首 Token 延迟或单 Token 延迟。

### 3.4 PTQ 耗时

PTQ 耗时表示从正式 PTQ 任务启动到对应算法的量化参数生成完成的工程运行时间。

### 3.5 Deploy 与 clean reload

Deploy 用于根据原模型和 PTQ 参数生成可部署模型目录。本文记录 Deploy 是否成功、端到端耗时、完成时的原始文件数和目录近似大小。

Deploy 目录还包含模型配置、权重索引和其他部署文件，因此目录大小不能简单等同于纯 4bit 权重大小。

clean reload 用于验证 Deploy 结果能否在独立加载流程中完成模型结构初始化、量化权重匹配和状态字典加载。检查内容包括：

- 缺失键；
- 非预期键；
- 不匹配键；
- 加载错误；
- 总量化模块数；
- 第 0 层量化模块数；
- Vision 分支量化情况；
- `lm_head` 权重缩放情况。

## 4. 正式实验结果

### 4.1 WikiText2 PPL

BF16 基线 PPL 为：

```text
7.782411575317383
```

七种正式量化方案的 WikiText2 test PPL 如下。

| 方法 | PPL | 相对 BF16 | 相对 Direct 恢复量 |
|---|---:|---:|---:|
| Direct | 7.893825054168701 | +0.111413 | 0.000000 |
| LWC | 8.021615028381348 | +0.239203 | -0.127790 |
| LAC | 7.833244323730469 | +0.050833 | +0.060581 |
| LWC+LAC | 7.982542037963867 | +0.200130 | -0.088717 |
| OmniQuant | 7.849410057067871 | +0.066998 | +0.044415 |
| AutoRound | 7.941936492919922 | +0.159525 | -0.048111 |
| FlatQuant | 7.695919036865234 | -0.086493 | +0.197906 |

### 4.2 Eval 耗时与运行条件

各方案的正式 WikiText2 Eval 结果如下。

| 方法 | PPL | Eval 耗时/s | 运行环境 | 运行条件 |
|---|---:|---:|---|---|
| Direct | 7.893825054168701 | 1695 | 新服务器，Ascend 910B1 | 物理 NPU 1，单卡独占 |
| LWC | 8.021615028381348 | 2042 | 旧服务器，Ascend 910B3 | 正式运行记录 |
| LAC | 7.833244323730469 | 2586 | 旧服务器，Ascend 910B3 | 正式运行记录 |
| LWC+LAC | 7.982542037963867 | 2215 | 新服务器，Ascend 910B1 | 物理 NPU 1，单卡独占 |
| OmniQuant | 7.849410057067871 | 2014 | 旧服务器，Ascend 910B3 | 正式运行记录 |
| AutoRound | 7.941936492919922 | 1820 | 新服务器，Ascend 910B1 | 物理 NPU 1，单卡独占 |
| FlatQuant | 7.695919036865234 | 1551 | 新服务器，Ascend 910B1 | 物理 NPU 1，单卡独占 |

上述 Eval 结果跨越两台服务器：

- Direct、LWC+LAC、AutoRound 和 FlatQuant 来自新服务器物理 NPU 1 单卡独占运行；
- LWC、LAC 和 OmniQuant 来自旧服务器正式运行记录。

因此，表中耗时可用于记录各正式任务的端到端运行情况，但不能视为完全统一硬件、软件负载和资源状态下的严格算法速度排名。

### 4.3 PTQ、Deploy 与 clean reload

#### 4.3.1 PTQ 结果

七种方案均已完成全部 64 层 MLP PTQ，且 `PARAM_COUNT` 均为 64。

| 方法 | PTQ 耗时/s | `PARAM_COUNT` | 参数目录大小 | 正式输出目录 |
|---|---:|---:|---:|---|
| Direct | 727 | 64 | 324K | `outputs/formal/direct_a4w4_mxfp4_mlp_full64` |
| LWC | 2237 | 64 | 4.0G | `outputs/formal/lwc_a4w4_mxfp4_mlp_full64` |
| LAC | 1713 | 64 | 452K | `outputs/formal/lac_a4w4_mxfp4_mlp_full64` |
| LWC+LAC | 2429 | 64 | 4.0G | `outputs/formal/lwc_lac_a4w4_mxfp4_mlp_full64` |
| OmniQuant | 2301 | 64 | 6.0M | `outputs/formal/omniquant_a4w4_mxfp4_mlp_full64` |
| AutoRound | 2439 | 64 | 68G | `outputs/formal/autoround_a4w4_mxfp4_mlp_full64` |
| FlatQuant | 2517 | 64 | 19M | `outputs/formal/flatquant_a4w4_mxfp4_mlp_full64` |

AutoRound 的约 68G 磁盘占用主要来自：

```text
ptq_params/qwen3_5/mlp
```

#### 4.3.2 Deploy 结果

七种方案全部完成 Deploy，`EXIT_CODE` 均为 0。

| 方法 | Deploy 耗时/s | 原始 Deploy 文件数 | 目录大小 |
|---|---:|---:|---:|
| Direct | 1396 | 80 | 约 29G |
| LWC | 1801 | 80 | 约 29G |
| LAC | 1577 | 80 | 约 29G |
| LWC+LAC | 1511 | 80 | 约 29G |
| OmniQuant | 1418 | 80 | 约 29G |
| AutoRound | 1475 | 80 | 约 29G |
| FlatQuant | 1445 | 80 | 约 29G |

#### 4.3.3 clean reload 结果

七种方案全部通过 clean reload。

| 方法 | clean reload | 耗时/s |
|---|---|---:|
| Direct | PASS | 17 |
| LWC | PASS | 18 |
| LAC | PASS | 19 |
| LWC+LAC | PASS | 19 |
| OmniQuant | PASS | 18 |
| AutoRound | PASS | 19 |
| FlatQuant | PASS | 18 |

七种方案的统一检查结果如下。

```text
REMOVED_ACTIVATION_USE_CLIP=2
FINAL_IGNORE_COUNT=416
MISSING_KEYS=Set()
UNEXPECTED_KEYS_COUNT=0
MISMATCHED_KEYS_COUNT=0
ERROR_MSGS_COUNT=0
TOTAL_QUANTIZED_COUNT=192
LAYER0_QUANTIZED_COUNT=3
VISION_QUANTIZED_COUNT=0
LM_HEAD_HAS_WEIGHT_SCALE=False
CLEAN_RELOAD=PASS
EXIT_CODE=0
```

Qwen3.6-27B Dense 的每层 MLP 包含以下三个量化 Linear：

```text
gate_proj
up_proj
down_proj
```

因此全部 64 层对应的量化模块数为：

```text
64 × 3 = 192
```

第 0 层包含 `gate_proj`、`up_proj` 和 `down_proj`，因此：

```text
LAYER0_QUANTIZED_COUNT=3
```
clean reload 检查确认：

- 没有缺失键；
- 没有非预期键；
- 没有权重形状不匹配；
- 没有加载错误；
- Vision 分支没有被错误量化；
- `lm_head` 没有附带本轮 MLP MXFP4 权重缩放参数；
- 192 个 MLP 量化模块均被正确识别。

上述结果说明七种 Deploy 产物均通过当前独立加载与完整性检查。

## 5. 运行环境

### 5.1 Python 与 AMCT 环境

| 组件 | 版本或状态 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0+cpu |
| `torch_npu` | import PASS |
| Transformers | 5.5.4 |
| Datasets | 4.0.0 |
| AMCT | import PASS |
| `NPU_AVAILABLE` | True |

本环境通过 `torch_npu` 接入 Ascend 设备；`torch_npu` 导入成功且 `NPU_AVAILABLE=True`，因此 AMCT 可以使用 NPU 执行模型处理。

### 5.2 环境进入方式

环境初始化脚本和 AMCT 仓库位置由实际部署环境决定。进入运行环境并切换到仓库根目录的通用形式如下：

```bash
source "${AMCT_ENV_SCRIPT}"
cd "${AMCT_REPO_DIR}"
```

其中：

- `AMCT_ENV_SCRIPT` 表示实际环境初始化脚本；
- `AMCT_REPO_DIR` 表示 AMCT 仓库根目录；
- 本文档不记录服务器专属绝对路径。

## 6. 已知警告和兼容性问题

### 6.1 内部架构名称

模型配置中的：

```text
model_type=qwen3_5
architectures=Qwen3_5ForConditionalGeneration
```

以及 AMCT 使用的模型适配名：

```text
qwen3_5
```

均为当前 Transformers 和 AMCT 代码中的内部架构标识。

本轮实际模型为 Qwen3.6-27B Dense，不能仅根据内部标识将其描述为 Qwen3.5 模型。

### 6.2 Dense MLP 的 CLI 参数名称

本轮量化目标是 Dense 模型中的 MLP，但经过正式流程验证可用的参数入口仍为：

```text
--moe_mlp_param_dir
```

参数名称中的 `moe` 不表示本轮模型是 MoE 模型。

### 6.3 Direct 不是注册算法

Direct 是不启用额外优化算法的直接量化基线，不是 AMCT 注册算法名。

Direct Eval 和 Deploy 必须不传：

```text
--algos
```

错误写法：

```text
--algos direct
```

会导致：

```text
KeyError: 'direct' is not registered in 'algo'
```

目录名中的 `direct` 只是工程输出标识，不能作为算法注册名使用。

### 6.4 MXFP block size 约束

MXFP 使用的 block size 为 32。相关 Linear 的输入维度必须满足对应整除要求。

第一次 Direct clean reload 中出现：

```text
ValueError:
tensor column shape must be divisible by group_size 32 but got 4304
```

相关权重属于视觉分支，不属于本轮文本 MLP 量化目标。

通过补充正确的 Vision、Visual、Image 和不满足 block size 整除条件的 Linear ignore 规则后，七种方案全部通过 clean reload。

### 6.5 FlatQuant 低于 BF16 的解释

FlatQuant 的 WikiText2 PPL 为：

```text
7.695919036865234
```

BF16 基线为：

```text
7.782411575317383
```

FlatQuant 在当前评测中的 PPL 低于 BF16，但该结果只适用于：

- 当前 Qwen3.6-27B 模型；
- 当前 MLP 量化目标；
- 当前 A4W4 MXFP4 配置；
- 当前 WikiText2 test 数据；
- 当前评测脚本和参数。

不能据此推导量化通常会提高精度，也不能直接推广到其他模型、任务或数据集。

## 7. 阶段结论

### 7.1 完成情况

本轮已完成 Qwen3.6-27B Dense 模型的 MLP A4W4 MXFP4 全 64 层量化实践。

正式完成的方案包括：

- Direct；
- LWC；
- LAC；
- LWC+LAC；
- OmniQuant；
- AutoRound；
- FlatQuant。

七种方案均完成：

- MLP 校准数据使用；
- 全 64 层 PTQ；
- WikiText2 test PPL Eval；
- Deploy；
- clean reload。

### 7.2 精度结果

BF16 基线 PPL 为：

```text
7.782411575317383
```

七种方案中，FlatQuant 获得当前最低 PPL：

```text
7.695919036865234
```

相对 BF16 变化为：

```text
-0.086493
```

相对 Direct 恢复量为：

```text
+0.197906
```

LAC 和 OmniQuant 的 PPL 也低于 Direct，分别为：

```text
LAC=7.833244323730469
OmniQuant=7.849410057067871
```

LWC、LWC+LAC 和 AutoRound 在当前评测中的 PPL 高于 Direct。

上述结果仅表示当前 WikiText2 test 配置下的实测表现。

### 7.3 工程完整性

七种方案的 Deploy 均成功完成：

```text
EXIT_CODE=0
```

每种方案的正式 Deploy 原始文件数均为：

```text
80
```

目录大小均约为：

```text
29G
```

七种方案均通过 clean reload，并统一得到：

```text
MISSING_KEYS=Set()
UNEXPECTED_KEYS_COUNT=0
MISMATCHED_KEYS_COUNT=0
ERROR_MSGS_COUNT=0
TOTAL_QUANTIZED_COUNT=192
LAYER0_QUANTIZED_COUNT=3
VISION_QUANTIZED_COUNT=0
LM_HEAD_HAS_WEIGHT_SCALE=False
CLEAN_RELOAD=PASS
EXIT_CODE=0
```

这表明本轮文本 MLP 的 192 个目标量化模块均被正确识别，且视觉分支与 `lm_head` 未被错误纳入量化范围。

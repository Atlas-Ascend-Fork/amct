# 简介

本项目提供了不同场景的调用样例，搭建完环境后，可以根据实际场景尝试运行：

| 样例 | 算法 | 说明 |
| ------ |------ |------ |
| 使用MIN-MAX算法量化模型 | [Min-Max](algorithms/minmax/README.md) | 基于极值的简单量化，入门首选 |
| 使用AWQ算法量化模型 | [AWQ](algorithms/awq/README.md) | 激活感知的权重量化，适合大模型 PTQ |
| 使用GPTQ算法量化模型 | [GPTQ](algorithms/gptq/README.md) | 基于二阶信息的权重量化，逐层优化 |
| 使用SmoothQuant算法量化模型 | [SmoothQuant](algorithms/smoothquant/README.md) | 平滑激活分布的 W8A8 量化 |
|使用Cast直转算法量化模型 | [Cast](algorithms/cast/README.md) | HiFloat8 数据直转 |
| 使用Quantile算法量化模型 | [Quantile](algorithms/quantile/README.md) | HiFloat8 分位量化 |
| 使用ofmr算法量化模型 | [OFMR](algorithms/ofmr/README.md) | 输出特征 Min-Max 量化 |
| 使用mxquant算法量化模型 | [MXQuant](algorithms/mxquant/README.md) | 微缩浮点量化（MXFP8/MXFP4） |
| 使用FlatQuant算法量化模型（试验特性） | [FlatQuant](algorithms/flatquant/README.md) | 通过仿射变换平整化分布的量化 |

> **注意**：标注"试验特性"的样例依赖 `amct_pytorch/experimental/` 目录下的内容，需使用
> `bash build.sh --torch --experimental`（或 `--pkg --experimental`）构建安装包后方可使用。

## 快速验证

`run_example.sh` 提供一键式端到端验证，覆盖 cast、minmax、smoothquant、awq 四种算法的完整量化流程（quantize → calibrate → convert → forward），使用随机校准数据，无需准备数据集。

**用法**

```bash
# 默认：下载 Qwen3-0.6B 并运行全部用例
bash examples/run_example.sh

# 使用本地模型（跳过下载）
bash examples/run_example.sh --model /data/Qwen3-0.6B

# 指定镜像加速下载
bash examples/run_example.sh --hf_endpoint https://hf-mirror.com

# 使用 ModelScope 下载（国内推荐）
bash examples/run_example.sh --model_source modelscope

# 只运行指定用例
bash examples/run_example.sh --cases minmax,smoothquant --device npu:0
```

**参数说明**

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-0.6B` | HuggingFace 模型名或本地路径，本地路径时跳过下载 |
| `--model_dir` | `/tmp/amct_verify_model` | 模型下载目标目录 |
| `--device` | `npu:0` | 运行设备 |
| `--cases` | `cast,minmax,smoothquant,awq` | 逗号分隔的用例列表 |
| `--model_source` | `huggingface` | 模型下载来源：`huggingface` 或 `modelscope`（国内推荐） |
| `--hf_endpoint` | 无 | HuggingFace 镜像地址（`--model_source huggingface` 时有效） |

**说明**

- cast 用例依赖原生 hifloat8 支持或 amct_ops，环境不满足时自动标记为 `SKIP`，不影响整体结果
- 全部通过时输出 `[AMCT_EXAMPLE_ALL_PASS]`，任意用例 FAIL 时输出 `[AMCT_EXAMPLE_FAILED]`

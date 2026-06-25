# MiniCPM on Ascend NPU — 低比特量化与部署

本项目在 **昇腾（Ascend）910B NPU** 上对 MiniCPM 系列模型做端到端的低比特训练后量化
（PTQ）与评测。所采用的核心量化技术是
[QuIP#](https://github.com/Cornell-RelaxML/quip-sharp)（incoherence processing + 格码本
的 2-bit PTQ），并在其基础上做了面向多架构与 NPU 的工程化与算法扩展。

目标：把 QuIP# 风格的低比特 PTQ 推广到 MiniCPM 等 **Llama 以外的模型架构**，并让整条
量化—微调—评测链路在昇腾 NPU 上直接跑通。

## 技术基础：QuIP# 三段式量化

量化主流程沿用 QuIP# 的三段式：

1. **Stage 1 — Hessian**：用校准数据前向，按层累积各 Linear 输入的二阶统计量（Hessian）。
2. **Stage 2 — 量化 + 微调**：LDLQ 逐层量化到格码本，并对 incoherence 旋转的缩放因子
   （SU/SV）做少量端到端微调补偿。
3. **Stage 3 — 导出**：还原成标准 HuggingFace 权重目录，可直接 eval / 推理。

## 本项目的扩展

在上游 QuIP# 之上，本项目针对「MiniCPM + 昇腾 NPU」这一目标做了四方面扩展：

- **支持 Llama 以外的架构。** 上游量化驱动假设了 Llama 的独立 `q/k/v/o` 与
  `gate/up/down` 投影。本项目的 `create_init_ckpt/model/general_model.py` 同时支持
  **独立投影**（`q_proj`/`k_proj`/`v_proj`、`gate_proj`/`up_proj`）和**融合投影**
  （`qkv_proj`、`gate_up_proj`）两类布局，按模型结构自动选择形状字典，从而覆盖
  MiniCPM 等使用融合投影的模型。

- **以减少 IO 的方式加速 Hessian 计算。** Stage 1 用 forward-pre-hook 在前向过程中
  **分块（chunked）流式累积** 每个 Linear 的 Hessian，按层即时落盘、用完即释放，
  避免把全部中间激活物化到磁盘/内存；累加精度按设备自适应（NPU fp32 / CUDA fp64）。
  这显著降低了 Hessian 阶段的 IO 与峰值内存。

- **引入 Gaussian 与 Identical 线性码本。** 在码本层
  （`create_init_ckpt/lib/codebook/`）新增两种 4 维线性格码本：
  - `linear_guassian`（默认）：码点经一个正交投影排列，使量化更贴近白化后输入的
    各向同性分布；
  - `identical`：轴对齐的规则立方格，作为对照/基线。

  两者通过 `--codebook` 选择，无需改动量化主流程。

- **移植到昇腾 NPU。** Python 入口（`hessian_general_model.py` /
  `quantize_finetune.py`）**自动探测设备**（`npu → cuda → cpu`），同一套代码在
  NPU 与 GPU 上无差别运行；评测包装器（`eval/run_lm_eval.py`）内置一组 NPU 兼容补丁，
  使 `lm-eval` 可直接在 NPU 上跑。

## 环境需求

参考 `requirements.txt`。关键依赖：

- Python 3.10
- `torch==2.5.1`、`transformers==4.52.4`
- `datasets`、`lm_eval==0.4.11`、`glog`、`huggingface_hub`、`scipy`、`numpy` 等

```bash
pip install -r requirements.txt
```

### 昇腾 NPU

本项目在如下昇腾环境上验证：

- 硬件：Ascend 910B（snt9b），aarch64
- CANN：`8.1.RC1`
- `torch==2.5.1` + `torch_npu==2.5.1`
- Python 3.10，HCE 2.0（OS）

需先 `source` CANN 的环境脚本（如 `/usr/local/Ascend/ascend-toolkit/set_env.sh`）使
`torch_npu` 可用；设备由 Python 入口自动探测，无需手动指定。

### CUDA

以 `TORCH_DEVICE=cuda` 运行即可（`torch==2.5.1`，无需 `torch_npu`）。

## 快速开始（以 MiniCPM4-0.5B 为例）

每个模型对应一个设备无关的「一模型一入口」脚本（`create_init_ckpt/scripts/run_*.sh`），
封装了该模型的配方常量。下面以可公开下载的 **MiniCPM4-0.5B** 走完整流程。

> **关于配方**：SOTA 实验线是 `run_1b.sh` / `run_3b.sh`，使用 `linear_guassian` 码本与
> `devset=4096 / ctx=2048 / ft_epochs=5 / scale_search_iters=3 / scale_override=0.83`
> 配方（Hessian 用 fp32、量化用 fp16）。`run_0.5b.sh` 是探索线（码本为 `identical`，
> 尚未对齐到 1B/3B 的 SOTA 精度基准），此处仅作端到端可跑通示例。把
> `BASE_MODEL`/`WORKSPACE`/`DATASET` 指向你的模型即可换用 `run_1b.sh` 等。

### 1. 下载基座模型

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download openbmb/MiniCPM4-0.5B \
    --local-dir ./models/MiniCPM4-0.5B
```

> 国内网络可设镜像：`export HF_ENDPOINT=https://hf-mirror.com`

### 2. 三段式量化

`run_0.5b.sh <stage>`：`1`=Hessian，`2`=量化+微调，`3`=导出 HF。路径通过环境变量覆盖，
配方常量（码本、量化 dtype、batch 等）已写在脚本里。

```bash
cd create_init_ckpt

export BASE_MODEL=../models/MiniCPM4-0.5B        # 第 1 步下载的基座
export WORKSPACE=../workspaces/minicpm4-0.5b     # 产物输出根目录
export DATASET=/path/to/calib.jsonl              # 校准数据（jsonl 或 pajama）

bash scripts/run_0.5b.sh 1   # Stage 1: Hessian
bash scripts/run_0.5b.sh 2   # Stage 2: 量化 + 微调
bash scripts/run_0.5b.sh 3   # Stage 3: 导出为 HF 权重
```

量化后的 HF 模型位于
`$WORKSPACE/exp/quip_sharp/MiniCPM4-0.5B/hf_npu_fast_aligned`。

> **在 CUDA 上运行**：同一脚本加前缀即可，设备会自动切换：
> ```bash
> TORCH_DEVICE=cuda BASE_MODEL=/path/to/MiniCPM4-0.5B HARNESS_Q_BATCH=32 \
>     bash scripts/run_0.5b.sh 1   # 2 / 3 同理
> ```
>
> **复现 SOTA**：换 `run_1b.sh` / `run_3b.sh`（`linear_guassian` 码本 + 上述 SOTA 配方）。
> 1B 建议额外设 `HARNESS_EMB_LEGACY_COPY=1`（见脚本注释）。

### 3. 评测

困惑度（WikiText-2）：

```bash
cd eval
python eval_ppl.py \
    --model-path $WORKSPACE/exp/quip_sharp/MiniCPM4-0.5B/hf_npu_fast_aligned \
    --dataset /path/to/wikitext-2-raw-v1/test-*.parquet \
    --max-length 2048 --stride 2048 --trust-remote-code
```

下游任务（lm-eval：arc / boolq / piqa / hellaswag / winogrande / ceval / cmmlu /
mmlu / openbookqa 等）。`run_lm_eval.py` 是 `lm-eval` CLI 的 NPU 兼容包装器，接受标准
`lm-eval` 参数：

```bash
cd eval
python run_lm_eval.py \
    --model hf \
    --model_args pretrained=$WORKSPACE/exp/quip_sharp/MiniCPM4-0.5B/hf_npu_fast_aligned,trust_remote_code=True \
    --tasks arc_easy,arc_challenge,boolq,piqa,hellaswag,winogrande,openbookqa \
    --batch_size 8
```

### 精度结果

下表为 MiniCPM4-0.5B 在常见下游基准上的得分，FP16 为未量化基线，PTQ 为本项目
2-bit 量化后的结果（`identical` 码本探索线）。`Per.` 为相对 FP16 的保留比例。

| 数据集 | FP16 | PTQ |
|---|---:|---:|
| ARC-C | 32.93 | 25.43 |
| ARC-E | 54.54 | 42.85 |
| BoolQ | 66.24 | 62.14 |
| CEval | 66.64 | 29.42 |
| CMMLU | 66.24 | 28.55 |
| HellaSwag | 45.17 | 35.29 |
| M3KE | 44.23 | 25.63 |
| MMLU | 47.19 | 27.45 |
| OpenBookQA | 31.40 | 28.80 |
| PIQA | 65.89 | 60.39 |
| WinoGrande | 52.57 | 52.25 |
| WPLC | 3.46 | 0.00 |
| **Avg.** | **48.04** | **34.85** |
| **Per.** | **1.0000** | **0.7254** |

## 常用环境开关

`create_init_ckpt/scripts/_ptq_stage_runner.sh` 顶部列出了全部开关，常用的有：

| 变量 | 作用 |
| --- | --- |
| `TORCH_DEVICE=cuda\|npu` | 强制设备（默认自动探测 `npu→cuda→cpu`） |
| `HARNESS_SMOKE=1` | `devset=256/ctx=1024/ft_epochs=1` 的全链路冒烟（仅验证管线，非精度） |
| `HARNESS_EMB_LEGACY_COPY=1` | 回退 embedding 内存机制（1B 推荐） |
| `HARNESS_Q_BATCH` / `HARNESS_HESS_BATCH` | 按设备覆盖 Stage 2 / Stage 1 的 batch |
| `HARNESS_MAX_LAYERS=N` | 仅量化前 N 层（profiling） |

## 目录结构

- `create_init_ckpt/` — 量化主流程（Hessian / LDLQ 量化 / 微调 / 导出），含码本、
  Hadamard 工具、按模型入口脚本（`scripts/run_*.sh`）。
- `eval/` — 困惑度与 lm-eval 评测入口（含 NPU 兼容补丁）。

## 许可证

本项目以 **GPLv3** 授权，完整条款见 [`LICENSE`](./LICENSE)。

量化主流程的核心代码改编自 Cornell RelaxML 的
[QuIP#](https://github.com/Cornell-RelaxML/quip-sharp)（同为 GPLv3）。所有改编自上游的
源文件均在文件头部标注了对应的上游来源；本仓库的修改（多架构支持、IO 优化的 Hessian、
Gaussian/Identical 码本、昇腾 NPU 移植等）在 GPLv3 下继续以相同许可证发布。

## 致谢

本项目基于 Cornell RelaxML 的 [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp)。
感谢原作者在 incoherence processing 与格码本量化上的开创性工作。

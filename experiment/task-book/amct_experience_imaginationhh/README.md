# Qwen2.5-3B HiFloat8 量化体验

使用 AMCT（Ascend Model Compression Toolkit）对 **Qwen2.5-3B-Instruct** 执行
**HiFloat8** 量化，对比量化前后在 wikitext2 上的困惑度（PPL），完成从环境搭建、
量化脚本编写到 benchmark 结果输出的全流程体验，并记录过程中的兼容性问题与优化建议。

> 一句话结论：在当前 CANN 9.1.0 环境下，AMCT 内置的 NPU HiFloat8 量化接口
> （`HIFP8_CAST_CFG`）因底层算子内核缺失而无法直接跑通；本实践改用 **amct_ops 的
> NPU 自定义 cast 算子**实现真·NPU HiFloat8 伪量化，成功跑通全流程，量化后 PPL
> 仅劣化约 **1.3%**。

---

## 1. 目录结构

```
amct_experience_imaginationhh/
├── README.md                          # 本文档（目录结构/环境说明/达成情况/体验反馈）
├── quantize.py                        # 量化执行脚本（三后端：npu_op / cpu_sim / amct）
├── npu_hifloat8_fakequant_linear.py   # 基于 amct_ops NPU 算子的 HiFloat8 伪量化 Linear
├── eval_common.py                     # 离线友好的模型/数据集加载 + PPL 评估工具
├── run.sh                             # 运行封装（自动补齐 PYTHONPATH，降低使用门槛）
└── result_npuop_full.json             # 全量量化前后 PPL 对比结果（脚本产出）
```

各文件职责：

| 文件 | 说明 |
|------|------|
| `quantize.py` | 全流程主脚本：加载模型 → 量化前 baseline PPL → HiFloat8 量化 → 量化后 PPL → 输出对比 JSON。通过 `--backend` 选择量化后端 |
| `npu_hifloat8_fakequant_linear.py` | 自定义量化模块，用 `amct_ops.hifloat8_cast` 的 NPU 算子做 HiFloat8 编解码伪量化，注册进 AMCT 量化流程 |
| `eval_common.py` | 封装模型加载、wikitext2 离线加载（本地 parquet 优先）、PPL 计算 |
| `run.sh` | 设置 `amct_pytorch`/`amct_ops`/`torch_npu(post4)` 等所需 PYTHONPATH 后调用 `quantize.py` |

---

## 2. 环境说明

### 2.1 硬件环境

| 项 | 值 |
|----|----|
| NPU 型号 | Ascend 910B3（A2 系列，soc=ascend910b） |
| 可见设备 | 1 卡（容器内逻辑 device id = 0；`npu-smi` 物理编号为 7） |
| 架构 | aarch64 |

### 2.2 软件环境

| 组件 | 版本 | 备注 |
|------|------|------|
| CANN | 9.1.0 | `/home/developer/Ascend/cann-9.1.0`，`version.info` Version=9.0.0 |
| Python | 3.12.9 | |
| PyTorch | 2.7.1+cpu | |
| torch_npu（系统） | 2.7.1.post2.dev20251226 | dev 快照，**未注册** `torch_npu.hifloat8` |
| torch_npu（实跑） | **2.7.1.post4** | requirements 指定版本，注册了 `hifloat8` dtype；隔离安装于 `/tmp/tnpu_post4`，由 `run.sh` 经 PYTHONPATH 优先加载 |
| transformers | 5.5.4 | |
| AMCT (amct_pytorch) | 源码仓（master，commit 57c5f99） | 需先生成 protobuf `*_pb2.py` 才能导入 |
| 模型 | Qwen2.5-3B-Instruct | hidden=2048, layers=36，从 ModelScope 下载 |
| 数据集 | wikitext-2-raw-v1（test） | 本地 parquet 离线加载 |

### 2.3 环境搭建步骤

```bash
# 0) 仓库根目录
cd <amct repo root>

# 1) Python 依赖（不要重装已配好的 torch/torch_npu）
pip install transformers==5.5.4 datasets==4.8.4 accelerate tqdm einops \
            sentencepiece pyyaml scipy pandas

# 2) 生成 protobuf *_pb2.py（仓库未含，import amct_pytorch 必需）
pip install grpcio-tools
python3 -m grpc_tools.protoc -I. --python_out=. \
  amct_pytorch/classic/graph_based/amct_pytorch/proto/*.proto

# 3) 编译 amct_ops NPU 自定义算子（HiFloat8 cast）
bash amct_ops/ops_build.sh hifloat8_cast    # 产出 amct_ops/staging 与 dist/*.whl

# 4)（可选）编译 CPU 仿真扩展，用于 cpu_sim 后端对照
cd amct_pytorch/experimental/hifloat8 && bash build.sh && cd -

# 5) 安装提供 hifloat8 dtype 的 torch_npu（隔离安装，不污染系统环境）
pip download torch_npu==2.7.1.post4 --no-deps -d /tmp/tnpu_probe
pip install --no-deps --target=/tmp/tnpu_post4 \
  /tmp/tnpu_probe/torch_npu-2.7.1.post4-*.whl

# 6) 准备 wikitext2 离线数据（HF 不通时用镜像）
mkdir -p /home/developer/datasets/wikitext2
curl -L -o /home/developer/datasets/wikitext2/test.parquet \
  "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
```

---

## 3. 运行方式

推荐用 `run.sh`（自动补齐 PYTHONPATH）：

```bash
cd experiment/task-book/amct_experience_imaginationhh

# 默认 NPU 自定义算子伪量化，全量评估
bash run.sh --model_path /home/developer/models/Qwen2.5-3B-Instruct

# 快速冒烟（前 5 段）
bash run.sh --model_path /home/developer/models/Qwen2.5-3B-Instruct --max_samples 5

# CPU 仿真后端对照
bash run.sh --model_path /home/developer/models/Qwen2.5-3B-Instruct --backend cpu_sim --max_samples 5
```

`run.sh` 支持环境变量 `AMCT_REPO_ROOT`、`TORCH_NPU_POST4` 覆盖默认路径；
`--dataset_path` 指定 wikitext2 本地 parquet。

也可直接调用 `quantize.py`（需自行设置 PYTHONPATH，详见 `run.sh`）。

### 后端说明

| `--backend` | 量化实现 | 当前环境 | 说明 |
|-------------|----------|----------|------|
| `npu_op`（默认） | amct_ops NPU 自定义 cast 算子伪量化 | ✅ 可跑通 | 独立 ascendc LUT kernel，不依赖 aclnnQuantize，全程 NPU |
| `cpu_sim` | experimental/hifloat8 CPU 扩展伪量化 | ✅ 可跑通 | 逐层 CPU↔NPU 搬运，慢，仅作精度对照 |
| `amct` | AMCT 内置 `HIFP8_CAST_CFG` | ❌ 失败 | 依赖 CANN aclnnQuantize 的 HiFloat8 内核，当前 CANN 9.1.0 未编入（见第 5 节） |

---

## 4. 任务达成情况

### 4.1 量化输入

| 项 | 值 |
|----|----|
| 模型 | Qwen2.5-3B-Instruct（hidden=2048, layers=36, FP16） |
| 量化数据格式 | HiFloat8（8-bit 浮点，变长 Dot 域 + 锥形精度） |
| 量化算法 | Cast（数据直转，无需校准数据） |
| 量化配置 | 权重 per-tensor 对称（scale = max(\|W\|)/16）；激活直转；跳过 `lm_head` |
| 量化粒度 | weights: channel/tensor；inputs: tensor |
| 校准数据 | 无（Cast 算法不需要） |
| 评估数据集 | wikitext-2-raw-v1（test，146 段 × 2048 token） |

### 4.2 量化结果（原始 vs 量化模型精度对比）

| 模型 | 数据集 | PPL | 相对劣化 |
|------|--------|-----|----------|
| Qwen2.5-3B（FP16 原始） | wikitext2 | **8.5570** | — |
| Qwen2.5-3B（HiFloat8 量化） | wikitext2 | **8.6724** | **+1.349%** |

> 完整结果见 [`result_npuop_full.json`](result_npuop_full.json)。
> HiFloat8 量化后 PPL 仅劣化约 1.35%，验证了其在 8-bit 低比特下凭借大动态范围与
> 锥形精度，对 LLM 推理精度影响很小。

耗时（910B3，146 段）：量化 4.6s，baseline 评估 16.1s，量化后评估 383.1s
（伪量化每个 Linear 含一次 HiFloat8 编解码，故评估更慢；真实部署态由硬件 matmul
直接消费 HiFloat8，无此开销）。

### 4.3 量化执行指令

```bash
bash run.sh --model_path /home/developer/models/Qwen2.5-3B-Instruct \
            --backend npu_op \
            --output result_npuop_full.json
```

### 4.4 任务详情达成

- [x] 详情 1：理解 AMCT 中 HiFloat8 量化核心逻辑（格式编码、Cast/Quantile/OFMR
  三算法、`HIFP8_*_CFG` 配置、`amct.quantize` / `algorithm_register` 用法）
- [x] 详情 2：完成 HiFloat8 量化全流程（环境搭建 → 量化脚本 → benchmark 输出）
- [x] 详情 3：原始 vs 量化模型精度对比 + 兼容性反馈与优化建议（见第 5 节）

---
## 5. 体验反馈、问题与优化建议

以下为实践中实际复现的主要问题，按严重程度排序。

### 5.1 致命阻塞

**【问题 1】AMCT 内置 NPU HiFloat8 量化（`HIFP8_CAST_CFG`）在当前环境无法跑通**

核心兼容性问题。`amct.quantize` 依赖 `torch_npu.hifloat8` dtype 及 CANN 的
HiFloat8 量化算子内核，二者缺一不可，而当前环境两层均不满足：

- 预装 `torch_npu`（dev 快照）未注册 `hifloat8` dtype（requirements 实际要求
  `2.7.1.post4`）；
- CANN 9.1.0 的 `quantize` 算子内核只支持 `int8`，HiFloat8 仅在算子原型中声明、
  未编入 kernel（声明与实现不一致），报 `DT_HIFLOAT8 not in [INT8,UINT8,INT32]`。

影响 AMCT 全部 NPU HiFloat8 路径。建议：强约束并对齐 `torch_npu` / CANN 版本，
提供“HiFloat8 算子是否可用”的自检，并在内核不支持时给出可读的兼容性提示而非
底层错误码。

**【问题 2】`import amct_pytorch` 开箱即崩，且报错误导**

protobuf 的 `*_pb2.py` 是构建产物但仓库未含、`build.sh` 默认也不生成，导致首次
import 报“循环导入”（实为文件缺失，误导排查）。建议在文档/构建流程中补齐 pb2
生成步骤，并在缺失时给出明确提示。

### 5.2 易用性问题

- **【问题 3】`requirements.txt` 依赖冲突**：`transformers==5.5.4` 与
  `datasets==4.8.4` 在 huggingface_hub 版本上互斥，严格安装会导致数据集加载失败。
  建议给出一组自洽的版本组合。
- **【问题 4】数据集硬编码在线加载**：样例写死 `load_dataset('wikitext', ...)`，
  离线/内网环境直接超时。建议支持本地 parquet（本实践 `eval_common.py` 已实现）。
- **【问题 5】CPU 仿真路径性能过低**：`Hifloat8FakequantLinear` 每次前向都
  CPU↔NPU 搬运，大模型评估极慢。建议明确其“精度调试”定位，评估走 NPU 算子路径。
- **【问题 6】NPU 逻辑/物理设备号不一致**：容器内应使用逻辑 device id（通常从 0
  开始），用物理号会报 `Invalid device ID`。建议文档提示。

### 5.3 亮点

- **amct_ops 的 NPU 自定义 cast 算子设计良好**：独立 ascendc LUT kernel，不依赖
  `aclnnQuantize`，在 CANN HiFloat8 内核缺失时仍可用，且与 CPU 实现编码字节完全
  一致，是本实践跑通真·NPU HiFloat8 伪量化的关键。
- **HiFloat8 精度表现优秀**：Cast 直转、无需校准，Qwen2.5-3B 量化后 PPL 仅劣化
  1.35%，体现了大动态范围 + 锥形精度的优势。

---

## 6. 附：HiFloat8 核心逻辑理解

- **格式**：8-bit = 符号(1) + 点位 Dot(2~4，前缀编码指示阶码位宽) + 阶码(0~4，
  符号-幅度编码) + 尾数(1~3)。变长字段实现**锥形精度**：高精度区间 E∈[-3,3] 用
  3-bit 尾数，阶码增大时尾数收缩至 1-bit，覆盖指数范围 [-22,15]（近 FP16）。
- **三算法**：
  - **Cast**：高精度浮点直转 HiFloat8，权重按 `max/16` 缩放，无需校准，最快。
  - **Quantile**：对激活最大值做指数滑动平均（0.99/0.01）抑制离群点，需校准。
  - **OFMR**：以输出特征图误差最小化搜索最优量化因子，精度最高、开销最大。
- **AMCT 用法**：`amct.quantize(model, HIFP8_*_CFG)` 一键量化；
  `amct.algorithm_register(name, 'Linear', QuantModule, DeployModule)` 注册
  自定义量化算法（本实践据此接入 NPU 自定义算子伪量化模块）。

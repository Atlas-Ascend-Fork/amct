# MR-GPTQ MXFP4 低比特量化算法（7月社区任务）

- **任务**：AMCT 低比特量化算法开发（量化算法方向，进阶项"未适配新算法"）
- **算法**：MR-GPTQ（Micro-Rotated GPTQ，arXiv:2509.23202），子类化 AMCT `GPTQuant` 实现，运行时注册、**零改 amct 源码**
- **数据类型**：MXFP4（block=32，E2M1 元素 + E8M0 共享指数）。主线 **W4A16**（weight-only），另实现 **W4A4**（权重+激活均 4-bit），**8B 与 35B 均达标**
- **测试模型**：Qwen3-8B（dense）、Qwen3.6-35B-A3B（`qwen3_5_moe` 混合 MoE）　|　**数据集**：WikiText2　|　**指标**：PPL（seq_len=4096）
- **精度目标**：`delta = ppl_quant − ppl_bf16 ≤ 0.4`；量化层（nn.Linear）占比 ≥ 70%；显存降低 ≥ 50%
- **提交者**：程逸雷 / gitcode `asp1r1n1`

## 当前状态：✅ 8B 与 35B 三项硬指标全部达标

**W4A16 主线**

| 模型 | bf16 PPL | MR-GPTQ PPL | **delta** | 量化层覆盖 | **权重压缩率** | 判定 |
|------|----------|-------------|-----------|------------|--------------|------|
| Qwen3-8B（dense） | 8.9893 | 9.2621 | **0.273** | 99.6%（nn.Linear） | **62.3%** | ✅ |
| Qwen3.6-35B-A3B（MoE） | 6.3111 | 6.3781 | **0.0671** | 71.4% nn.Linear / 全参数 97% | **71.2%** | ✅ |

**W4A4（权重+激活均 4-bit）**

| 模型 | bf16 PPL | W4A4 PPL | **delta** | **真 4-bit 激活层**（任务书口径） | 判定 |
|------|----------|----------|-----------|------------------|------|
| Qwen3-8B | 8.9893 | 9.3790 | **0.3897** | 180/252 = **71.4%** | ✅ |
| Qwen3.6-35B-A3B | 6.3111 | 6.6352 | **0.3241**（三 seed 0.2842–0.3241） | 250/350 = **71.4%**（**无保护层**） | ✅ |

要求：delta ≤ 0.4、量化层（nn.Linear）占比 ≥ 70%、显存降 ≥ 50%。

### 权重压缩率怎么算的

MXFP4 每 32 个元素 = 32 × 4 bit（E2M1 元素）+ 8 bit（E8M0 共享指数）
→ **4.25 bit/参数**；未量化部分（embedding / lm_head / norm / router / 维度不整除 32 的层）保 fp16 = 16 bit。

```
压缩率 = 1 − (量化参数 × 4.25 + 未量化参数 × 16) / (全部参数 × 16)
```

| 模型 | 量化参数占比 | 代入 | 压缩率 |
|---|---|---|---|
| Qwen3-8B | 84.8%（6.95B / 8.19B） | 1 − (0.848×4.25 + 0.152×16) / 16 | **62.3%** |
| Qwen3.6-35B-A3B | 97.0%（33.62B / 34.66B） | 1 − (0.970×4.25 + 0.030×16) / 16 | **71.2%** |

全部参数都量化时的上限为 `1 − 4.25/16 =` **73.4%**。35B 更接近上限，是因为
embedding/lm_head 在更大模型中占比更低（8B 中约 15%，35B 中约 3%），与 MoE 量化质量无关。
实现见 `src/eval_utils.py` 的 `report_memory_full`，运行时打印 `[memory-full]`。

⚠️ **该指标是按位宽的解析推算，不是运行时实测。** 本工作是 **fake quantization**：
权重数值已被量化到 MXFP4 的 16 个格点上（**精度损失真实发生，PPL 为实测**），
但张量仍以 fp16 承载，故 `torch` 运行时显存不变。要兑现该压缩率需 4-bit 打包存储格式
与配套反量化算子，属 AMCT `deploy` 环节，不在本任务范围。
此为量化算法工作的通行口径（GPTQ / AWQ / QuaRot 等均以位宽比报告压缩比）。

### 两个覆盖率口径，勿混淆

| 数字 | 含义 | 不是什么 |
|---|---|---|
| 35B **全参数 97%** | **MXFP4 数据类型**的参数覆盖率 | 不是 MR-GPTQ **算法**的覆盖率——完整算法（旋转+scale fitting+GPTQ 补偿）只作用于 250 个 `nn.Linear`，约占 **4%** 参数；其余 32B 融合专家走 RTN + scale fitting |
| **71.4% 的层跑真 4-bit** | **层数**口径（任务书定义为 `torch.nn.Linear` 占比） | 不是**算力**口径——35B 算力大头是 32B 融合专家，其激活仍为 fp16 |

### 测量分辨率与归因判据

**「测量分辨率」= 本实验能分辨的最小 delta 差异。** 类比游标卡尺的最小刻度：
最小刻度 0.02 mm 的卡尺量不出 0.01 mm 的差别——不是被测物没有差别，而是量具分辨不出来。

⚠️ 它**不是一种消减波动的方法**，而是一把**判定尺**：先量出本实验的噪声有多大，
再据此判断哪些实测差异算数、哪些只是噪声。波动本身并未被消除，只是被划入了「不归因」。

W4A4 的达标余量较小，故先测定实验自身的噪声水平——同一配置重复测量时，结果本身能波动多大：

- 固定配置、只更换校准语料的抽样种子（`--calib_seed`），8B W4A4 的 delta 在
  0.3131 / 0.3897 / 0.4241 之间，**极差 0.111**；
- 固定种子、扫描 GPTQ 阻尼系数，相邻取值间的结果**非单调**、摆动约 **0.06**。

**本实验分辨不了小于约 0.06 的差异**，故效应量低于该量级者不作归因。

各手段的**配对复测**结果（固定其余变量、只改一项，并跨三个校准种子）：

| 手段 | 效应量 | 结论 |
|---|---|---|
| 混合精度保护（8B） | 0.224 | ✅ 远超噪声，**确证有效** |
| 旋转强度（k=128 → 全宽） | 0.147 | ✅ **确证有效**（k=128 下 delta 0.4993 不达标） |
| GPTQ 阻尼调优 | +0.0146 | ❌ **证伪**：三种子配对下 2/3 变差，效应方向随数据变号 |
| 其余四项 | < 0.06 | ⚪ **不归因**——低于分辨率，不足以支撑结论 |

「不归因」指：这些手段**未被证明有害**，只是本实验的精度不足以证明它们有益，
故不作为"贡献了多少精度"陈述。各项实测效应量见报告 §4.3 ④。

**该分辨率是配置相关的**：同样的测量在 W4A16 主线上极差仅 0.0350（余量 0.127，主线结论稳固）、
在 35B W4A4 上为 0.0399（三种子全部达标）。完整数据见报告 §4.3 ⑤。

详见 [自验证报告](docs/self-verification-report.md)、[实验记录](docs/experiment-log.md)。

## 目录

```
mr-gptq-mxfp4_asp1r1n1/
├── README.md
├── docs/
│   ├── design.md                    # 设计文档（评审对象，v1.0 最终版）
│   ├── task-book.md                 # 任务书原文存档
│   ├── experiment-log.md            # 实验流水与踩坑
│   ├── self-verification-report.md  # 自验证报告（数据+截图）
│   └── img/                         # 运行截图
└── src/
    ├── run_eval.py         # 评测入口（bf16 / gptq_mxfp4 / mr_gptq）
    ├── mr_gptq_module.py   # ★ MR-GPTQ 实现（核心交付）
    ├── eval_utils.py       # 加载 / 校准 / PPL / 覆盖率与显存统计
    ├── data.py             # WikiText2 / pile 数据集加载
    ├── run_ablations.sh    # 消融批跑（分组 / 断点续跑 / 自动汇总）
    └── run_night.sh        # 夜间队列（测量分辨率复测与 35B W4A4）
```

> **关于日志文件**：运行日志按评审要求不纳入提交（批跑脚本会在 `src/logs/` 下自行创建）。
> 各文档中出现的 `xxx.log` 是**运行标识**，用于标注每个数值出自哪一次运行、
> 便于按上文命令复现，不指向仓内文件。所有被引用的数值均已在文档正文中列出。

## 方案概述

MR-GPTQ = GPTQ 骨架 + ① **block-wise Hadamard 微旋转**（正态化权重分布、摊平离群值）② **MXFP scale fitting**（附录 H，E8M0 scale 网格重映射为次幂次）。论文的第三项增量「静态激活重排」**有意未采用**——本实现已对每层施加全宽旋转，旋转后各通道近似可交换、按幅度分组已无物可分（报告 §5）。以仓内 GPTQ-MXFP4 为 baseline 对照证明精度提升。

W4A4 在此基础上另加激活量化与四项优化。经三 seed **配对复测**，其中两项确证、一项证伪：

| 手段 | 效应量 | 判定 |
|---|---|---|
| **混合精度保护**（`o_proj`/`down_proj` 保高精度激活） | **0.224** | ✅ 8B 不可省；**35B 必须弃用**（覆盖率口径，见报告 §4.4） |
| **旋转强度**（k=128 → 全宽；`down_proj` 的 12288 走 Paley H₁₂ ⊗ Sylvester H₁₀₂₄） | **0.147** | ✅ k=128 达 0.4993、**不达标** |
| GPTQ 阻尼调优 | +0.0146 | ❌ **证伪**（2/3 seed 变差、方向随数据变号） |

⚠️ 达标依赖全宽旋转，而其在线开销在朴素矩阵乘下约 **+104%**（`k=128` 仅 1.6% 但不达标）——
**可部署性取决于是否实现 FWHT**（O(k log k)，开销 < 1%），本项目未实现。

工程要点：MXFP4 是纯 torch fake-quant（无需 NPU 算子）；**GPTQ 误差补偿整体移至 CPU 并预计算 shared exponent，端到端量化耗时由约 2–3 小时降至分钟级（~10×，详见下节）**；MoE 融合路由专家（非 nn.Linear、amct 覆盖不到）额外实现 MXFP4 量化以达显存指标（RTN + scale fitting，无旋转/无补偿——完整 MR-GPTQ 仍只作用于 250 个 nn.Linear，口径见报告 §4.2）。详见 [design.md](docs/design.md)。

### 「~10× 提速」指的是哪里的开销

**位置**：GPTQ 误差补偿的**逐列内层循环**（`mr_gptq_module.py` 步骤 7）。GPTQ 必须一列一列地量化并把误差补偿到未量化列，每列上要做的都是**小张量运算**——形状仅 `[out_features]` 的除法、`float_to_fp4e2m1`、乘法、以及一次 `[out,1]×[1,n]` 的外积更新。

**规模**：Qwen3-8B 每层需处理 `Σ in_features = 6×4096 + 12288 = 36864` 列，36 层合计约 **133 万列**；每列若干个小算子，总计**千万量级的算子调用**。原实现每列还要重算一次 shared exponent。

**改动**（三项）：

| 改动 | 内容 |
|---|---|
| shared exponent **预计算** | 由逐列重算改为整层算一次——**与运行设备无关** |
| 补偿循环**移至 CPU** | 全程 CPU float32，含 Cholesky（约 0.7 s/层） |
| 改用**标准 block-32 MXFP4** | 替换非标准的 per-column 缩放 |

**机制已由微基准实测确证**（`src/bench_compensate.py`，日志 `src/logs/bench_compensate.log`）：

| out_features | 64 | 1024 | 4096 | 65536 | 规模 ×1024 后 |
|---|---|---|---|---|---|
| **NPU** `_quant_col` 耗时 | 1568.9 μs | 1700.1 μs | 1718.4 μs | 1939.0 μs | **×1.24** |
| CPU | 175.5 μs | 207.0 μs | 291.5 μs | 1468.3 μs | ×8.37 |

**计算量放大 1024 倍、NPU 耗时仅增 24%** ⇒ 时间几乎全花在算子启动/分发上，**launch-bound 成立**。实测 NPU 单算子启动开销 **≈ 13.6 μs**，反推 `_quant_col` 每列约 **115 个 kernel**（`float_to_fp4e2m1` 为纯位运算实现，16 个格点需多次掩码比较）。out=64 时 **CPU 比 NPU 快约 9 倍**。

**通信开销**：实测 D2H 约 3.9 GB/s、H2D 约 4.6 GB/s；8B 全网补偿搬运约 64 GB、耗时约 **16 s**，占校准总耗时 1849.5 s 的 **0.85%**，已包含在端到端的 10× 之内。

**⚠️ 仍未完全分离的部分**：微基准显示**单换设备**只贡献 **2.5–2.9×**（补偿循环整体，因其中的外积更新是 compute-bound、NPU 不吃亏），而端到端为 ~10×；差额应主要来自 **shared exponent 预计算**（与设备无关），推算约 3.7× ——**该拆分由两个实测值相除得到，未单独实测**。

故准确表述为：**该重构使端到端量化耗时降低约一个数量级（实测）；「小算子在 NPU 上受启动开销主导」已获直接测量支持；「设备切换」与「预计算」两项的贡献比例为推算。** 详见报告 §6。

## 运行方式

```bash
cd src
export ASCEND_RT_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1

# --- Qwen3-8B（单卡）---
python3 -u run_eval.py --model_path <Qwen3-8B> --mode bf16     --seq_len 4096
python3 -u run_eval.py --model_path <Qwen3-8B> --mode mr_gptq  --increments hadamard scale_fitting --seq_len 4096

# --- Qwen3.6-35B-A3B（双卡，35B 单卡放不下）---
export ASCEND_RT_VISIBLE_DEVICES=0,1
python3 -u run_eval.py --model_path <35B> --mode bf16    --seq_len 4096 --device_map auto
python3 -u run_eval.py --model_path <35B> --mode mr_gptq --increments hadamard scale_fitting --seq_len 4096 --device_map auto
```

```bash
# --- W4A4 达标配置（Qwen3-8B，单卡；delta 0.3897）---
python3 -u run_eval.py --model_path <Qwen3-8B> --mode mr_gptq --seq_len 4096     --hadamard_k 4096 --hadamard_full --perc_damp 0.2     --act_protect down_proj o_proj     --increments hadamard scale_fitting act_quant act_scale_fit joint_comp
```

运行即打印：量化层覆盖率（`[coverage]`/`[actual-coverage]`）、融合专家量化（`[fused-experts]`）、显存降低（`[memory-full]`）、WikiText2 PPL（`Score:`）。环境搭建见 [experiment-log.md](docs/experiment-log.md) §2。

## ⚠️ 安全提示

本项目对 `qwen3_5_moe` 等**自定义模型架构**，在 `eval_utils.py` 加载时使用了
`trust_remote_code=True`（该架构未并入 transformers 主干，不开则无法加载）。此参数会让
HuggingFace 模型仓内的 `modeling_*.py` / `tokenization_*.py` 等自定义代码在加载阶段**直接执行**，
等同于以当前进程权限运行任意 Python 代码。若 `--model_path` 指向被篡改或来路不明的模型，可能导致
**主机被控、数据泄露**。

使用建议：
- **仅从受信任来源获取模型**（如官方 ModelScope `Qwen/...`、HuggingFace 官方仓）；
- 尽量**固定到已知安全的版本 / commit hash**，避免加载被篡改的最新快照；
- 不要把 `--model_path` 指向未经核验的第三方模型目录。

## 交付说明

设计文档、算法代码与自验证报告**统一由单个 PR 交付**：分支 `feat/mr-gptq-impl` →
`cann/amct:feature/community-tasks`，目录 `experiment/task-book/mr-gptq-mxfp4_asp1r1n1/`。

- 交付 #1 自测用例 / 测试报告 / 步骤文档 → `docs/self-verification-report.md`（§3 为可复现步骤）
- 交付 #2 算法代码 → `src/`（核心为 `mr_gptq_module.py`）
- 交付 #3 设计文档 → `docs/design.md`

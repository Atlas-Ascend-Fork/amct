# MR-GPTQ：面向 MXFP4 的 LLM 低比特量化算法设计文档

| 项                | 内容                                                               |
| ----------------- | ------------------------------------------------------------------ |
| 任务              | 7月社区任务 — 低比特量化算法开发                                  |
| 数据类型          | MXFP4（微缩浮点）；**W4A16 主线 + W4A4 补充；8B 与 35B 在两种位宽组合下均已达标**（见 §4.4） |
| 目标算法          | MR-GPTQ（Micro-Rotated GPTQ），AMCT 仓未适配算法（进阶项）         |
| 测试模型          | Qwen3-8B（dense）、Qwen3.6-35B-A3B（MoE）                          |
| 测试集 / 指标     | WikiText2 / PPL（seq_len=4096, block 粒度）                        |
| 精度目标          | delta = ppl_quant − ppl_bf16 ≤ 0.4；量化层(nn.Linear)占比 ≥ 70% |
| 作者 / gitcode_id | 程逸雷 / asp1r1n1                                                  |
| 版本              | **v1.0（最终版）** —— 8B/35B 在 W4A16 与 W4A4 下均已达标；全部归因经配对复测重述 |

> 交付目录 `experiment/task-book/mr-gptq-mxfp4_asp1r1n1/`；gitcode 账号 `asp1r1n1`，私仓 fork：https://gitcode.com/asp1r1n1/amct 。最终 PR 目标 `cann/amct` 的 `feature/community-tasks` 分支。

---

## 1. 背景与目标

### 1.1 背景

大模型推理的显存与带宽瓶颈主要来自权重与激活的搬运。将 `nn.Linear` 的权重/激活从 BF16 压到 4bit，可把显存占用降低 50% 以上，并在昇腾 NPU 上使能低比特运算。**MXFP4（Microscaling FP4）** 是业界主推的 4bit 浮点格式：每 32 个元素共享一个 2 的幂次（power-of-two, PoT）缩放因子，单元素为 E2M1（1 符号 + 2 指数 + 1 尾数）。相比 INT4，MXFP4 的分段浮点表示对权重的长尾分布更友好，且硬件可原生支持。

但 MXFP4 有一个结构性难点：**共享 scale 被限制为 2 的幂次**，直接量化时 scale 的舍入误差会被整块放大，导致精度显著劣化（这一点是 MR-GPTQ 论文的核心动机）。因此简单的 min-max / round-to-nearest 直转在 MXFP4 上往往不达标，需要专门的算法补偿。

### 1.2 目标

在 AMCT 中实现 **MR-GPTQ** 算法，用于 MXFP4 低比特量化，并在两个测试模型上满足：

- **绝对精度门**：`delta ≤ 0.4`（相对原始 BF16 网络）；
- **量化覆盖门**：被量化的 `torch.nn.Linear` 占比 ≥ 70%；
- **相对收益（进阶项要求"精度优化"）**：在相同 MXFP4 配置下，MR-GPTQ 的 PPL 明显优于仓内已有的 **GPTQ-MXFP4** baseline；
- **可复现**：提供独立评测脚本与完整日志（不走 AMCT 标准 CLI，理由见 §4.5）。

### 1.3 非目标

- 不做 hifloat8（≤0.1 门槛过严，非本任务主线）；hifloat8 仅作为可选附加场景，直接复用仓内 OFMR 现成结果（Qwen3-8B delta≈0.093），不额外开发。
- 不做自定义 NPU 算子；MXFP4 的 cast/量化算子复用仓内 `quantization/dtypes/mxfp_impl.py` 与已有 deploy 算子。
- 不改动 workflow / solver / quant module 主干；算法作为可插拔单元接入。

---

## 2. 现状分析（复用面）

AMCT 仓已具备接入 MR-GPTQ 所需的大部分积木，本方案是"改造 + 组合"，而非从零：

| 已有能力       | 位置                                                                                                            | 在本方案中的角色                                                                                                                                     |
| -------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 算法注册框架   | `amct_pytorch/algorithms/quant/__init__.py` 的 `register_algorithms()` + `registry_factory.ALGO_REGISTRY` | MR-GPTQ 按同样方式注册为`--algos mr_gptq`                                                                                                          |
| 算法接入范本   | `amct_pytorch/classic/quantize_op/gptq_module.py`                                                             | **本设计直接子类化该类**（MR-GPTQ 是 GPTQ 变体）。`algorithms/quant/flatquant.py` 属另一套注册系统（可学习结构变换），本设计不走该通路，见 §2.1 |
| GPTQ 误差补偿  | `amct_pytorch/classic/quantize_op/gptq_module.py`（已支持 MXFP4，见算法矩阵）                                 | 作为 baseline**对照组**，并提供误差补偿实现参考                                                                                                |
| MXFP4 数据类型 | `amct_pytorch/quantization/dtypes/mxfp.py` / `mxfp_impl.py`                                                 | 提供 MXFP4 的 fake-quant / cast                                                                                                                      |
| 全链路 CLI     | `amct_pytorch.eval / extract_ptq_data / ptq / deploy`                                                         | **本设计不使用**（运行时注册的算法不在 CLI 白名单内，且 W4A4 需按增量开关做消融），改提供独立脚本，理由见 §4.5 |
| 模型适配       | `common/models/llm/qwen/`（qwen3、qwen3_6_moe 已注册）                                                        | 两个测试模型均已适配，无需新写 adapter                                                                                                               |

**算法支持矩阵佐证**（`docs/zh/algorithm_brief.md`）：GPTQ 已支持 MXFP4 权重量化 → baseline 对照组现成可用；这是本方案"证明有精度提升"的关键前提。

### 2.1 baseline 代码级验证结论（本地无 NPU，已从代码确认链路）

> 以下为接入所依据的代码事实（一并给出源码位置，便于核对）：

- **CLI 合法**：`cli/llm/args.py` 中 `--quant_dtype` 的 `choices=['int','mxfp']`、`--quant_target` 的 `choices=['mlp','moe','attn-linear','attn-cache']`，与本方案命令行一致 ✅。
- **GPTQ 原生支持 MXFP4**：`gptq_module.py:26` 导入 `MXFP4_E2M1`；`:192` 对 MXFP4 走特殊分支（`cal_scale_offset_static` 返回 `None,None`，即 scale 由 per-block（group_size=32）动态计算，不走静态 min-max）✅。
- **MXFP4 微块大小 = 32**：`common/utils/vars.py:50` `MXFP4_E2M1: [32]`，`:64` mxfp4 可用算法为 `['awq','gptq','mxquant']` → GPTQ-MXFP4 坐实为合法 baseline ✅。
- **fake-quant / 部署路径齐全**：`common/utils/quant_util.py` 的 `quant_dequant_weight/tensor` 已处理 MXFP4_E2M1；部署 `classic/deploy_op/weight_npu_quant_module.py`、`npu_mx_quantization_linear.py` 已支持 MXFP4 导出 ✅。
- **两套注册系统（关键架构认知）**：
  - `AlgorithmRegistry`（`algorithms/__init__.py`）= **算子级权重量化算法**（minmax/awq/**gptq**/mxquant/ofmr/...），签名 `register(name, module_type, quant_module, deploy_module)`。GPTQ 在此：`register('gptq','Linear',GPTQuant,NpuWeightQuantizedLinear)`。
  - `ALGO_REGISTRY`（`algorithms/quant/`）= **可学习结构变换**（flatquant/lwc/lac/omniquant/autoround）。
  - **MR-GPTQ 属前者**（GPTQ 变体），不走 flatquant 那套注册。
- **依赖 NPU 环境**：`gptq_module.py:132` 硬 `import torch_npu`，数值实验必须在昇腾环境完成。

---

## 3. 算法原理

MR-GPTQ = GPTQ 骨架 + 三个专为 FP4 微缩格式设计的增量（论文 §4.1）。本设计采用其中 ① 与 ③，不采用 ②。

### 3.1 MXFP4 格式与难点

- MXFP4 = block size **32** + 元素 E2M1（1符1指2尾，7个正值 {0.5,1,1.5,2,3,4,6}）+ 共享 scale 用 **E8M0**（纯指数、无尾数 → scale 被逼成 2 的幂次 PoT）。
- 元素量化：`q = fp4_e2m1(x / s)`，反量化 `x̂ = s · dequant(q)`。
- **难点（论文 §3 定量证明）**：native 权重/激活是 Laplace 重尾分布；absmax scaling 下块内 outlier 会挤压其余元素精度。更要命的是 E8M0 的 PoT scale：能表示 2⁻¹²⁷~2¹²⁸ 但实际数据范围很窄，网格太粗 → scale 量化误差大。这是 MXFP4 掉点（RTN 下 ~10% 相对）的主因。

### 3.2 GPTQ 误差补偿（基线，直接复用）

逐层 GPTQ：用校准激活构造 Hessian `H = 2XᵀX + λI`（λ=1% 均值阻尼，与仓内 `perc_damp=0.01` 一致），固定列序贪心量化，每量化一列用 `H⁻¹` 的 Cholesky 把误差补偿到未量化列，最小化 `‖XW − XŴ‖²`。仓内 `gptq_module.py` 已实现（含 MXFP4 分支）。这既是 baseline，也是 MR-GPTQ 的骨架。

### 3.3 MR-GPTQ 的三个增量（论文 §4.1）

**① Block-wise Hadamard 微旋转（Ingredient 3）**
对线性层 `Y = XWᵀ`，用 **block 对角 Hadamard** `H_k`（k×k 分块，k 为 2 的幂）：`Q(WH_k)·Q(XH_k)ᵀ`。
- 原理（论文 §3 证明）：Laplace 重尾经 Hadamard 后趋于 Normal，把块内 outlier 能量摊到整个 block → 旋转后 MXFP4 的 RTN 误差更小。
- **与 QuaRot/FlatQuant 的"全 hidden 大旋转"不同**：大旋转会把 outlier 跨 32-block 扩散、反破坏微缩块结构，故用 **micro（块内）** 旋转，k 对齐 block size。
- **论文的 k 取值**：主表用 Had32（对齐 block），Platinum bench（更低噪）显示 MXFP4 上 Had128 明显优于 Had32（跨 4 个 block 的更大块旋转）。
- **fuse 方式**：`WH_k` **离线**融进权重；`XH_k` **在线**（见 §4.3 我们的处理，精度评测无需快核）。
- **本设计的取值：全宽旋转**（与论文取小 k 不同，理由如下）。在 8B 上把 k 从 128 一路加到 4096
  （= 全 hidden 宽），再把 `down_proj` 补到 12288（全宽）。**「k=128 → 全宽」的整步效应量
  为 0.147（已确证）：k=128 下 W4A4 delta 达 0.4993、不达标，全宽下 0.3519、达标。**
  论文取小 k 的真实约束是**在线旋转算子的推理开销**，而非"大旋转破坏微缩块结构"。
  数据见自验证报告 §4.3 ③(a)。

  ⚠️ 该发现的适用条件必须一并说明：本任务只做 fake-quant 精度评估、旋转不计推理成本。
  真部署时开销 = `k / out_features`，**朴素矩阵乘下全宽旋转约 +104% 在线算力**
  （`down_proj` 一层即 300%），而 k=128 仅 +1.6%。故「取满 k」不是无代价的选择，
  其可行性取决于**快速 Hadamard 变换（FWHT，O(k log k)，开销 < 1%）**是否可用。
  中间的细分（如"全宽 vs k=4096"的 0.0305）低于本实验测量分辨率，不单独归因。

**② 静态激活重排（Static Activation Reordering, Ingredient 2）—— 本设计不采用**

> **口径统一**：论文三项增量中，本设计实现 ① 与 ③，**不实现 ②**。理由见本条末尾。
> （GPTQ 父类自带的 dynamic act-order 始终启用，与本条所述的"静态"重排是两件事。）

论文的做法：GPTQ 原生 dynamic act-order（按 Hessian 对角降序重排）精度好但推理要动态 reshuffle（慢 10-20%）。MR-GPTQ 改**静态**：先按原始列序定 scale/grid，再 shuffle 列跑 GPTQ，跑完 shuffle 回去，保持微缩 group 结构。**同等精度、零推理开销**。

**不采用的理由**：该增量的作用是"让同量级的通道共享一个 32 块"，以缓解块内幅度悬殊。
而本设计已对每层施加**全宽 Hadamard 旋转**（§3.3①）——旋转后各通道近似同分布、可交换，
按幅度分组已无物可分，与 SmoothQuant 式的难度迁移被旋转吸收同理。
故本设计有意不实现该项，并在自验证报告 §5 给出说明。

**③ MXFP scale fitting（Ingredient 1 / 附录 H）—— 需与 Hadamard 配合，不能单独用于 GPTQ**
针对 3.1 的 PoT scale 误差，把 E8M0 的 256 个码位从 [2⁻¹²⁷,2¹²⁸] 重映射到数据实际范围，记作 **MXFP4†**。**确切公式（附录 H 式(1)(2)(3)）**：
- 标准 E8M0（含 4/3 去偏）：`s_E8M0 = (4/3)·2^clamp(round(log₂s),−128,127)`（式1）。
- 拟合网格：码位 `q = clamp(round(255·(log₂s − log₂s_min)/(log₂s_max − log₂s_min)), 0, 255)`；拟合后 `s_fit = 2^(α·q + β)`，其中 **α = (log₂s_max − log₂s_min)/255**（斜率 <1 → 次幂次、比幂次细）、**β = log₂s_min**（式2/3）。`s_min/s_max` 取该张量所有块 scale 的最小/最大值。
- 效果（论文表 11）：RTN 下 Qwen3-8B **93.7→96.3**、Llama3-8B 87.8→94.3；MR-GPTQ 下 Qwen3-8B **95.2→98.5**。
- **⚠️ 顺序约束（本设计据此排序）**：scale fitting **不能单独用于 GPTQ**。论文表 11 GPTQ 行：**Qwen3-8B 94.1→92.3（−1.8，变差）**；只有叠在 **Hadamard 之上**（MR-GPTQ）才发力（95.2→98.5）。RTN 下才单独有效。**本项目实测复现了这点**：GPTQ-mxfp4 10.13 → 单加 scale fitting 10.26（变差）。故本设计中 scale fitting 必须叠加在 Hadamard 之上。

> 校准设置：论文用 1024 条 FineWeb 序列、λ=1e-2，与仓内 GPTQ 默认基本一致。
> **论文自证我们的靶子成立**：Qwen3-8B 上 MXFP4 MR-GPTQ(95.2%) > GPTQ(94.1%) > RTN(93.7%)（表10）——"相比 baseline 有提升"在测试模型上已被原作者验证，我们是在 AMCT 里复现这个已知正结果。

---

## 4. AMCT 接入设计

### 4.1 文件与改动范围

**对 amct 的改动为零**：算法以运行时注册接入，代码全部位于交付目录内。

| 文件 | 性质 | 说明 |
|---|---|---|
| `src/mr_gptq_module.py` | **新增（核心交付）** | `class MRGPTQuant(GPTQuant)` + `register()`；`import` 即完成注册 |
| `src/run_eval.py` | 新增 | 评测入口，`--mode {bf16, gptq_mxfp4, mr_gptq}`，增量与 W4A4 开关均为命令行参数 |
| `src/eval_utils.py` / `src/data.py` | 新增 | 模型加载与校准 / PPL 评测 / 覆盖率与显存统计；数据集加载 |
| `src/run_ablations.sh` / `run_night.sh` / `bench_compensate.py` | 新增 | 批跑与性能微基准 |
| `amct_pytorch/**` | **不修改** | 仅 import 四个符号，见下 |

对 amct 的**全部耦合**为四行 import：

```python
from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant       # 父类
from amct_pytorch.common.utils.quant_util  import cal_shared_exponent   # MX 共享指数
from amct_pytorch.common.utils.data_utils  import float_to_fp4e2m1      # E2M1 cast
from amct_pytorch.common.utils.vars        import MXFP4_E2M1
```

### 4.2 注册（运行时，零改源码）

MR-GPTQ 是 GPTQ 变体，属 `AlgorithmRegistry`（算子级权重量化算法）而非 `ALGO_REGISTRY`
（可学习结构变换，flatquant 那套），见 §2.1。注册在 `mr_gptq_module.register()` 内完成，
`import mr_gptq_module` 时执行：

```python
def register():
    AlgorithmRegistry.register('mr_gptq', 'Linear', MRGPTQuant, NpuWeightQuantizedLinear)
    BUILT_IN_ALGORITHM.add('mr_gptq')                       # 运行时向白名单追加
    ALGORITHM_SUPPORTED_QUANT_TYPE_COMB['mr_gptq'] = {'NOT_QUANTIZE mxfp4_e2m1'}
```

三步分别解决：算法查找、白名单校验、算法 × 数据类型的支持表。
部署模块沿用 `NpuWeightQuantizedLinear`（已支持 `MXFP4_E2M1`）。

### 4.3 MRGPTQuant 实现要点（复用父类 GPTQuant）

父类 `GPTQuant` 已提供：Hessian 累积（`update_hessian`）、求逆（`cal_hessian_inverse`，Cholesky）、贪心量化 + 误差补偿主循环（`get_opt_weight_and_quant_factor`）、MXFP4 分支（`cal_scale_offset_static` 返回 `None,None`）、fake-quant 缓存。本设计在其上 override 三个方法：

| override 的方法 | 加入的处理 |
|---|---|
| `update_hessian` | 先对输入施加块对角 Hadamard `x·R`，使 `H_r = RᵀHR` 与旋转后的权重一致（转输入比转 Hessian 便宜）；W4A4 且开启联合补偿时，在此处先量化激活再累积 Hessian |
| `get_opt_weight_and_quant_factor` | 旋转权重 `W·R` → 预计算每块 scale（scale fitting，§3.3③）→ 逐列量化并补偿；全程 CPU float32 |
| `fake_quant_forward` | 对输入同样施加 `x·R`；W4A4 时再对旋转后的激活做逐 token、逐 32 通道块的动态 MXFP4 |

**两项增量的实现与依赖关系**：

1. **Block-wise Hadamard 微旋转（§3.3①）**：进 GPTQ 列循环前 `W' = W·R`，补偿在 `W'` 上做。
   `R` 为块对角 Hadamard，块大小取满 `in_features`（全宽）；2 的幂走 Sylvester 递归构造，
   `down_proj` 的 12288 走 `12×1024` 的 Paley H₁₂ ⊗ Sylvester H₁₀₂₄。旋转矩阵按 `(k, device)`
   全局缓存，同一 k 只构造一次。
2. **MXFP scale fitting（§3.3③）**：叠加在 Hadamard 之上，替换 MXFP4 的 per-block scale，
   用式(2)/(3) 把 E8M0 幂次网格重映射为 256 级次幂次细网格。
   **顺序不可颠倒**——单独用于 GPTQ 会变差（§3.3③）。

**激活侧旋转 `X·R` 的落点**：本任务目标是精度（PPL），不涉及推理加速，故 `X·R` 直接在
fake-quant 前向里用朴素 PyTorch 块对角 `matmul` 实现（数值正确即可）。旋转的 matmul 在
fp32 下做再回原 dtype——k 较大时 fp16 累加会引入可观误差、污染消融。

> ⚠️ 该实现形态的代价：朴素矩阵乘下全宽旋转的在线开销约 +104%（§3.3①），
> 真实部署需快速 Hadamard 变换（FWHT）。本任务不实现该算子，属落地前置工作。

**校准型、无梯度训练**：与 GPTQ 一样只需一遍校准前向求解，不跑训练循环——
这是相对 BATQuant（可学习）在算力预算上的核心优势。

### 4.4 目标位宽决策：W4A16 为主线，W4A4 为进阶（8B / 35B 均已达标）

论文数据给了明确指引（表2 vs 表1，Llama3-8B）：**weight-only MXFP `GPTQ` recovery 96.76% ≫ W4A4 的 89.47%**——激活量化贡献了约一半误差。对本任务：

| 方案 | 权重 | 激活 | 满足显存≥50%? | 满足量化层≥70%? | PPL 达标难度 | 定位 |
|---|---|---|---|---|---|---|
| **W4A16（weight-only）** | mxfp4 | bf16 | ✅ 权重 16 → 4.25 bit/参数，整体 >50% | ✅ | **低**（无激活量化+无在线激活旋转） | **首选/保底** |
| W4A4（全量化） | mxfp4 | mxfp4 | ✅ 更省 | ✅ | 高（激活敏感，需在线旋转） | 进阶/加分 |

任务书硬指标（delta≤0.4 / 量化层≥70% / 显存≥50%）**未限定位宽组合**，W4A16 已全部满足且风险最低 → **8B/35B 主线走 W4A16**（已达标：8B delta 0.273、35B 0.0671）。

**为何仍需 W4A4**：weight-only 只拿到存储收益——matmul 前权重要反量化回 bf16，
计算量与访存均不变，推理反而可能变慢；只有激活也量化，matmul 才能真正在低比特上算。
故两个模型均实现并评估 W4A4（下文），**均已达标**：

| 模型 | delta | 真 4-bit 激活层（任务书口径） | 保护层 | 报告 |
|---|---|---|---|---|
| Qwen3-8B | **0.3897** ≤ 0.4 | 180/252 = **71.4%** | `down_proj` + `o_proj`（**不可省**，去掉则 0.5761） | §4.3 |
| Qwen3.6-35B-A3B | **0.3241** ≤ 0.4 | 250/350 = **71.4%** | **无**（加了会压到 57.1% 不达标） | §4.4 |

⚠️ **同一手段在两个模型上结论相反**：8B 上混合精度保护不可省（效应量 0.224），
35B 上则必须弃用——该模型有 100 个 `nn.Linear` 因维度不整除 32 而无法量化、却仍占分母，
任何保护都会扣掉达标余量。**配置不可跨模型默认继承，须按各模型自身约束重新推导。**

⚠️ **35B 的 W4A4 只作用于 250 个 `nn.Linear`**：32B 融合路由专家不是 `nn.Linear`，
`quantize_fused_experts` 是纯权重改写通路、不接管 forward，无处施加激活量化，其激活仍为 fp16。
故「71.4% 的层跑真 4-bit」≠「71.4% 的算力跑在 4-bit 上」，详见报告 §4.4。

**W4A4 精度评估**：在**已实现的 Hadamard 在线旋转**基础上补齐激活量化：
- **方法**：`fake_quant_forward` 内，对旋转后的激活 `X·H_k` 做**逐 token、逐 32 通道块的动态 MXFP4**（E8M0 shared exponent 每次前向重算）——旋转把激活离群值摊平，是激活能 4-bit 的前提；
- **实现**：`--increments hadamard scale_fitting act_quant` 启用；amct 配置仍为 weight-only 组合（放行），激活量化在自实现 fake-quant 路径完成，用于**精度评估**（真部署需低比特激活算子，属另一课题）；
- **消融**：W4A4「旋转开 vs 关」对照，量化 Hadamard 对激活量化的关键作用；
- W4A4 精度数据见自验证报告 §4.3（8B）与 §4.4（35B）。delta 明显大于 W4A16（激活敏感），
  但**两个模型均独立达标**，不再仅作"进阶补充"；W4A16 主线结论亦不受影响。

**W4A4 精度优化设计**：在上述基础实现之上，针对 4-bit 激活的主要误差源（离群值 + scale 粗糙 +
补偿输入失配）设计 4 项优化，均为可独立开关的增量，便于消融：

| 优化 | 开关 | 设计要点 |
|---|---|---|
| 旋转块加强 | `--hadamard_k K` | 论文取 k=128 是**在线旋转算子**的开销权衡；本任务只做 fake-quant 精度评估、旋转无推理开销，故把 k 提升为可调（512/1024/4096），块越大离群值摊得越开。注意 `in_features % k == 0` 才旋转，且大 k 下旋转 matmul 须在 fp32 做（fp16 累加 4096 项误差会污染消融） |
| 激活 scale-fitting | `--increments act_scale_fit` | 把权重侧已验证的附录 H 网格重映射（式2/3）复用到激活块 scale：以次幂次细网格替代 E8M0 幂次网格，减小激活 scale 的舍入误差 |
| 联合补偿 | `--increments joint_comp` | `update_hessian` 中先对（旋转后）激活做 MXFP4 量化再累积 Hessian。GPTQ 补偿的最优性依赖 Hessian 与推理期输入分布一致；W4A4 下推理输入是量化激活，用 bf16 输入建的 Hessian 会系统性失配 |
| 混合精度保护 | `--act_protect <子串...>` | 按层名子串把指定层的激活保留高精度（权重仍全程 MXFP4）。默认候选为 `down_proj`（输入 = 门控后中间激活）与 `o_proj`（输入 = 注意力输出）——两处离群值最重。需同时报告"仍跑真 4-bit 激活的层占比"，避免用保护面换数字。**8B 上实测不可省**（去掉则 0.3519 → 0.5761，效应量 0.224）；**35B 上必须弃用**（会把占比压到 57.1%）——`report_act_coverage` 同时打印两个分母（已量化模块 / 全部 nn.Linear），后者为任务书口径 |
| **全宽旋转** | `--hadamard_full` | 每层旋转块取满 `in_features`。`down_proj` 的 in=12288 在 k=4096 下只能切 3 个对角块，是全网**唯一未获全宽旋转**的层、而它最大。12288 非 2 的幂，且 `3×4096` 不可行（3 阶 Hadamard 不存在），走 `12×1024`：Paley H₁₂ ⊗ Sylvester H₁₀₂₄——正交矩阵的 Kronecker 积仍正交。**「k=128 → 全宽」整步效应量 0.147（确证，k=128 达 0.4993 不达标）**；其中"全宽 vs k=4096"的细分（0.0305）低于测量分辨率，不单独归因 |
| GPTQ 阻尼系数（**经复测证伪，非精度手段**） | `--perc_damp` | `damp = perc_damp × mean(diag(H))`。原设计假设：`down_proj` 的样本/维度比仅约 1.1:1、Hessian 近奇异，加大阻尼可压制噪声主导的补偿方向。**经三 seed 配对复测已证伪**——0.2 vs 0.4 在 2/3 的校准集上变差、均值亦变差、效应方向随数据变号。开关保留（交付配置固化为 0.2），但**不再作为一项精度手段**。详见报告 §4.3 ③(c) |

另实现并消融了 **5 项未采纳的手段**（激活 scale 裁剪搜索、加大校准集、权重 scale 裁剪、
增加保护层、randomized Hadamard），开关默认关闭、代码保留作消融记录。

⚠️ **这 5 项的效应量多数低于本实验分辨率**（见下方"归因方法说明"），故其中大部分只能表述为
「在本配置下未观察到收益」，**不足以支撑"该手段有害"**。经复测后仍站得住的只有一条：

> **randomized Hadamard 在 block-32 粒度下有害**（效应量 0.138，接近判据上界）——DC 分量由
> 集中于一个坐标变为摊到全部块，而 scale 是逐块的，**集中优于摊匀**，与其在 per-channel
> 粒度下的通行做法结论相反。这是一条与**量化粒度相关**的一般性结论：跨工作迁移做法前须核对粒度。

该结论有一处限定：本实验只测了 block-32 一条臂，未在同一管线下复现 per-channel 的正结果，
故只能陈述「该粒度下随机化未见收益」。逐项数据见自验证报告 §4.3 ④。

各优化的复测、最终精度、以及与公开工作的同协议对比见自验证报告 §4.3 / §4.4。

**⚠️ 归因判据**：本实验的**测量分辨率**为——同配置仅换 `--calib_seed` 时 delta 极差 **0.111**，
同 seed 下阻尼响应面相邻点摆动 **0.06**。故上表各项的贡献均以**配对复测**（固定其余变量、
只改一项，跨三个校准种子）判定，且只对效应量大于该分辨率者归因：
**混合精度保护 0.224、旋转强度 0.147 为确证**，**阻尼为证伪**，其余不归因。
**该分辨率是配置相关的**（W4A16 主线极差仅 0.0350、35B W4A4 为 0.0399），需各自测定。
详见报告 §4.3 ⑤。

**⚠️ 可部署性前提**：W4A4 达标依赖全宽旋转，而旋转的在线开销 = `k / out_features`——
朴素矩阵乘下 Qwen3-8B 全宽旋转约 **+104% 在线算力**（`down_proj` 一层即 300%），`k=128` 仅 +1.6%
但**不达标**（0.4993）。故本配置的可部署性取决于是否实现**快速 Hadamard 变换（FWHT，O(k log k)）**，
其下开销降至 **< 1%**。本项目为 fake-quant 精度评估，未实现该算子，此为落地的前置工作。

### 4.5 全链路调用（实际实现：独立评测脚本）

**为何不走 AMCT 标准 CLI**：本设计提供独立脚本 `src/run_eval.py` 而非复用
`eval / extract_ptq_data / ptq / deploy`，原因有二：① MR-GPTQ 采用**运行时注册**（`import` 即注册、
零改 amct 源码），标准 CLI 的算法白名单在包内，不改源码则无法识别 `mr_gptq`；② W4A4 的激活量化需
在 `fake_quant_forward` 内介入，且需按增量开关做消融，标准 CLI 无对应参数面。
**量化本身仍完全走 amct 的 `amct.quantize()` 与 `GPTQuant` 状态机，未绕过工具链。**

```bash
cd src
export ASCEND_RT_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

# 1. BF16 基准
python3 -u run_eval.py --model_path <M> --mode bf16 --seq_len 4096

# 2. baseline 对照：GPTQ-MXFP4 直转（仓内已有算法）
python3 -u run_eval.py --model_path <M> --mode gptq_mxfp4 --seq_len 4096

# 3. MR-GPTQ 主线（W4A16）
python3 -u run_eval.py --model_path <M> --mode mr_gptq --seq_len 4096 \
    --increments hadamard scale_fitting

# 4. MR-GPTQ W4A4 达标配置（Qwen3-8B）—— 保护层不可省
python3 -u run_eval.py --model_path <Qwen3-8B> --mode mr_gptq --seq_len 4096 \
    --hadamard_k 4096 --hadamard_full --perc_damp 0.2 \
    --act_protect down_proj o_proj \
    --increments hadamard scale_fitting act_quant act_scale_fit joint_comp

# 5. MR-GPTQ W4A4 达标配置（Qwen3.6-35B-A3B）—— 双卡，且【无 --act_protect】
export ASCEND_RT_VISIBLE_DEVICES=0,1
python3 -u run_eval.py --model_path <35B> --mode mr_gptq --seq_len 4096 \
    --device_map auto --calib_chunk 32 \
    --hadamard_k 4096 --hadamard_full --perc_damp 0.4 \
    --increments hadamard scale_fitting act_quant act_scale_fit joint_comp

# 35B 的 W4A16：同上去掉 act_quant 系增量，保留 --device_map auto
# 校准集敏感性复测：追加 --calib_seed <n>（默认 42，交付结果以此为准）
```

**未做导出部署**：本任务验收指标为精度（PPL）与覆盖率 / 显存，均由 fake-quant 确定性数学决定；
`deploy` 导出属推理落地环节，不在本任务范围（W4A4 的真实部署另需低比特激活算子）。

---

## 5. 量化范围与覆盖率（实测）

覆盖率定义（任务书口径）：**被量化的 `nn.Linear` 数 / 全网 `nn.Linear` 数**（跳 `lm_head`）。

| 模型 | 全网 nn.Linear（跳 lm_head） | 实际量化 | 覆盖率 | 判定 |
|---|---|---|---|---|
| Qwen3-8B（dense） | 252 | 252 | **99.6%**（252/253） | ✅ |
| Qwen3.6-35B-A3B（MoE） | 350 | 250 | **71.4%** | ✅（余量仅 1.4 个百分点） |

**两个模型的覆盖率来源完全不同：**

- **8B**：每个 decoder layer 的 7 个 Linear（q/k/v/o、gate/up/down）维度均整除 32，
  全部可量化，仅跳过 `lm_head`。
- **35B**：⚠️ **路由专家不是 `nn.Linear`**——它们是 `Qwen3_5MoeExperts` 的融合 3D 张量
  `gate_up_proj`/`down_proj`（约占 35B 参数中的 32B），**既不在覆盖率的分子里、也不在分母里**，
  amct 编排层匹配不到。分母中的 350 个 `nn.Linear` 是 attn / 线性注意力投影 / 共享专家；
  其中 **100 个因 in/out 维不整除 32 而无法做 block-32 MXFP4**（部分 `linear_attn` 投影、
  `shared_expert_gate`(out=1)），**结构性地永远进不了分子**，故上限即 250/350 = 71.4%。

**融合专家单独走一条通路**（`quantize_fused_experts`，§4.3）：不处理则显存指标必然不达标。
处理后 MXFP4 的**参数量**覆盖达 97.0%（33.62B/34.66B）、权重压缩率 71.2%。

> ⚠️ **三个口径不可混用**：覆盖率 71.4%（`nn.Linear` **层数**）、参数量覆盖 97.0%
> （**MXFP4 数据类型**）、MR-GPTQ **算法**覆盖约 4%（仅 250 个 `nn.Linear`，
> 融合专家走 RTN + scale fitting）。详见自验证报告 §4.2。

### 5.1 量化层 ↔ 量化算法 对应关系

**并非所有被量化的参数都走同一套算法。** 逐类列明：

**Qwen3.6-35B-A3B（全参数 34.66B）**

| 层类别 | 计数 | 参数量 | 施加的算法 | 权重类型 | W4A4 下激活 |
|---|---|---|---|---|---|
| attn / `linear_attn` 投影 / `shared_expert` 中维度整除 32 的 `nn.Linear` | **250** | 1.41B（4.1%） | **完整 MR-GPTQ**：全宽 Hadamard 旋转 → MXFP scale fitting → GPTQ Hessian 补偿 | MXFP4 | **4-bit**（逐 token、逐 32 通道块动态） |
| 融合路由专家 `Qwen3_5MoeExperts.{gate_up_proj, down_proj}` | 40 模块 × 2 张量 | **32.21B（92.9%）** | **MXFP4 RTN + scale fitting**（**无旋转、无 GPTQ 补偿**） | MXFP4 | **fp16**（不量化） |
| 维度不整除 32 的 `nn.Linear`（部分 `linear_attn` 投影、`shared_expert_gate` out=1） | **100** | \} 合计 1.04B（3.0%） | 无 | bf16 | fp16 |
| `router` / `embedding` / `norm` / `lm_head` | — |  | 无 | bf16 | fp16 |

**Qwen3-8B（dense，全参数 8.19B）**：结构单一——每 decoder layer 的 7 个 `nn.Linear`
（q/k/v/o、gate/up/down）维度均整除 32，**252 个全部走完整 MR-GPTQ**（6.95B，84.8%）；
`embedding` / `lm_head` 保 bf16。不存在融合专家，无第二条通路。

**⚠️ 为什么融合专家只能走 RTN + scale fitting，拿不到完整算法**

`quantize_fused_experts` 是**纯权重改写**通路——只重写 `W.data`，**不接管 forward**。
而 MR-GPTQ 的另两项增量都需要在前向上挂东西：

- **Hadamard 旋转**要求推理时对专家输入施加 `x·R`，纯权重改写没有挂载点；
- **GPTQ 补偿**需要前向累积 Hessian `H = 2XᵀX`，同样需要接管 forward。

只有 **scale fitting 是纯权重侧变换**，故可施加（论文 Table 11 表明它对 RTN 本就是加分项）。
要让这 32B 也拿到完整算法，需自行接管 `Qwen3_5MoeExperts.forward`——属后续工作。

**由此得到「97% 是数据类型覆盖率，不是算法覆盖率」**：92.9% 的参数拿到的是 MXFP4
**数据类型**（RTN 量化），只有 4.1% 拿到了论文的完整 MR-GPTQ **算法**。
两者在文档中始终分开陈述。

**覆盖率余量是 35B 的主要风险**：71.4% 距门槛仅 1.4 个百分点（≈5 层），已无冗余。
故 35B 上禁用混合精度保护（会压到 200/350 = 57.1%），见 §7 与报告 §4.4。

---

## 6. 实验设计与验收

### 6.1 对照实验矩阵

| 实验 | 配置 | 目的 | 实测（Qwen3-8B, seq_len=2048 全量 146 块） |
| --- | --- | --- | --- |
| E0 | BF16 baseline | 得 `ppl_bf16`（精度门分母） | 9.7154 |
| E1 | GPTQ-MXFP4 直转 | baseline 对照，得 `ppl_gptq` | 10.1332（delta +0.418，**不达标**） |
| E2 | MR-GPTQ MXFP4（+Hadamard） | 主方案 | 10.0747（delta +0.359 ✅） |
| E3 | MR-GPTQ MXFP4（+Hadamard +scale fitting） | 主方案（完整） | 10.0499（delta +0.3345 ✅） |

> 交叉验证：E1 的 10.1332 与 amct 自带 GPTQ-MXFP4 的 10.1369 相差 0.004，
> 佐证本实现的正确性。**纯 GPTQ 不达标、增量使其达标**，即进阶项要求的"相对提升"。
> 交付数值另按 §6.2 的口径在 `seq_len=4096` 下产出，见自验证报告 §4。

**判定**：

- 绝对达标：`ppl_mrgptq − ppl_bf16 ≤ 0.4`
- 相对提升（进阶项）：`ppl_mrgptq < ppl_gptq`（越低越好）
- 覆盖达标：Linear 占比 ≥ 70%（脚本统计并写入报告）

### 6.2 评测口径

统一口径：`seq_len=4096`、block 粒度、WikiText2 **全量分块**评测（8B 73 块 / 35B 72 块），
默认 `--calib_seed 42`。所有交付数值均按本口径产出，可按 §4.5 的命令逐位复现。

**不作定量依据的两类数据**（详见自验证报告 §4.3 ⑤）：

- **测试集子集**：以前 20 块做快速筛选时，其相对全量的偏移随配置摆动达 0.053、大于多数待测
  增量，**连排序都不可靠**；
- **低于测量分辨率的差异**：同配置换校准语料抽样，8B W4A4 的 delta 极差达 0.111；同一校准集下
  阻尼响应面相邻点摆动约 0.06。故**效应量小于约 0.06 的差异不作归因**。

### 6.3 性能数据

由评测脚本直接统计并打印：量化层覆盖率（`[coverage]` / `[actual-coverage]`）、权重显存降低
（`[memory]` Linear 口径 / `[memory-full]` 全模型口径）、校准与补偿耗时、PPL 评测耗时，
连同原始日志写入自验证报告 §6。本任务验收指标为精度与显存、且量化为 fake-quant，
脚本内建统计已足够，未引入算子级采集工具；补偿循环的性能归因另附微基准
`src/bench_compensate.py`。

---

## 7. 风险与规避

以下为交付时点的风险状态（历史风险项的处置记录见附录 B）：

| 风险 | 影响 | 现状与规避 |
|---|---|---|
| **融合专家量化的瞬时显存峰值** | 双卡虽装得下 35B（70 GB / 128 GB），但 `device_map="auto"` 贪心填充后 card 0 常仅剩约 1 GB，而单个 Experts 模块转 fp32 后约 2 GB、含中间量峰值更高 | 量化计算全程在 CPU 完成（**先搬 CPU 再转 fp32**，顺序颠倒会先在卡上 OOM）；回写采用三步法（先赋 CPU 张量释放旧权重 → `empty_cache` → 再挪回），避免新旧两份同时在卡上。见报告 §4.2 |
| **35B 覆盖率余量仅 1.4 个百分点**（250/350 = 71.4% vs 门槛 70%） | 任一配置变动使 5 层掉出即不达标 | 100 个 `nn.Linear` 因 in/out 维不整除 32 **结构性**无法量化、却始终占分母，已无冗余。故 35B 上禁用混合精度保护（会压到 57.1%）；`report_act_coverage` 同时打印两个分母以防误判 |
| **8B 的 W4A4 对校准语料抽样敏感** | 换一批校准语料可能越过门槛（三 seed 实测 0.3131 / 0.3897 / 0.4241） | 交付件锁定默认 `--calib_seed 42`，按 §4.5 命令可逐位复现；已实测"加大校准集""调整阻尼"两个方向均无法改善最差值，边界在报告 §4.3 ① 如实披露。W4A16 主线（极差 0.0350）与 35B W4A4（极差 0.0399）无此问题 |
| **全宽旋转的在线开销** | 精度达标配置在朴素实现下约 +104% 在线算力，`k=128` 虽仅 +1.6% 但不达标 | 本任务为 fake-quant 精度评估，不计推理成本；真实落地需实现 FWHT（O(k log k)，开销 < 1%），属部署前置工作，已在报告 §4.3 ③(a) 与 §7 标注 |
| **压缩率为解析推算而非实测** | 与"实际显存下降"不等价 | fake-quant 下张量仍以 fp16 承载；兑现压缩率需打包存储格式与配套反量化算子，属 `deploy` 环节。口径已在 README 与报告 §4 显式声明 |
| **提速的机制未经分离验证** | "10×"由三项改动打包取得，且 kernel 启动这一机制为推断 | 端到端墙钟为实测且含主机-设备传输；已附微基准 `src/bench_compensate.py` 供确证，口径见报告 §6 |

---

## 8. 交付件

> 算法以**运行时注册**方式接入、**零改 amct 源码**，故代码全部位于交付目录内。
> 交付目录：`experiment/task-book/mr-gptq-mxfp4_asp1r1n1/`

| 交付件     | 位置                                                          |
| ---------- | ------------------------------------------------------------- |
| 算法代码   | `src/mr_gptq_module.py`（子类化 `GPTQuant`，`import` 即注册，**零改 amct 源码**） |
| 评测脚本   | `src/run_eval.py`（入口）、`src/eval_utils.py`（加载/校准/PPL/覆盖率与显存统计）、`src/data.py`（数据集加载） |
| 实验日志   | 运行日志按评审要求不纳入提交；各数值均已在报告正文列出，可按 §6.2 命令复现 |
| 自验证报告 | `docs/self-verification-report.md`（结果 + 截图 + §3 可复现步骤 + §4.3 / §4.4 W4A4） |
| 实验记录   | `docs/experiment-log.md`（环境搭建与踩坑流水）                |
| 设计文档   | 本文 `docs/design.md`                                         |
| README     | `README.md`（目标、结果、目录、运行方式、安全提示）           |
| PR         | `feat/mr-gptq-impl` → `cann/amct:feature/community-tasks`（设计 + 代码 + 报告，**单 PR**） |

---

## 附录 A：参考

- MR-GPTQ: *Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization*, arXiv:2509.23202
- GPTQ: arXiv:2210.17323
- MXFP4 微缩格式: arXiv:2310.10537
- FlatQuant: arXiv:2410.09426
- BATQuant（备选路线）: arXiv:2603.16590
- AMCT 算法矩阵: `docs/zh/algorithm_brief.md`
- Qwen3.6-MoE 量化样例: `examples/models/qwen3.6/Qwen3.6-Moe.md`

---

## 附录 B：设计演进记录（历史，非最终设计）

> 本附录仅记录设计过程中被修正或废弃的判断，供追溯。**最终设计以正文为准**，
> 二者若有出入，以正文为准。

| # | 初期判断 | 最终结论 | 修正依据 |
|---|---|---|---|
| 1 | MR-GPTQ 按 `ALGO_REGISTRY`（flatquant 那套可学习结构变换）注册 | 改走 `AlgorithmRegistry`（算子级权重量化算法），子类化 `GPTQuant` | MR-GPTQ 是 GPTQ 变体，非可学习变换；见 §2.1 |
| 2 | scale fitting 收益最大，应最先做、可单独用 | 必须叠加在 Hadamard 之上；单独用于 GPTQ 会变差 | 论文表 11 GPTQ 行 Qwen3-8B 94.1→92.3；本项目实测 10.13→10.26 |
| 3 | 实现论文三项增量（含静态激活重排） | **不实现静态激活重排**——已施加全宽旋转，旋转后按幅度分组已无物可分 | §3.3② |
| 4 | 沿用论文的小 k（32/128）微旋转 | 取满 `in_features` 的全宽旋转 | 「k=128 → 全宽」效应量 0.147；k=128 下 W4A4 不达标（§3.3①） |
| 5 | 复用 AMCT 标准 CLI 全链路 | 提供独立评测脚本 | 运行时注册的算法不在 CLI 白名单；W4A4 需按增量开关做消融（§4.5） |
| 6 | 算法代码落进 `amct_pytorch/` 并改注册文件 | 运行时注册、零改 amct 源码，代码全部在交付目录内 | §4.2 |
| 7 | 加大 GPTQ 阻尼是达标关键手段（单项 −0.0261） | **证伪**——三 seed 配对下 2/3 变差、效应方向随数据变号 | 报告 §4.3 ③(c) |
| 8 | 全宽旋转单项贡献 −0.0305 | 该细分低于测量分辨率，不单独归因；仅「k=128 → 全宽」整步可归因 | 报告 §4.3 ⑤ |
| 9 | 以测试集前 20 块做快速筛选 | 全部定量结论改用全量评测 | 该子集偏移随配置摆动达 0.053，连排序都不可靠 |
| 10 | 35B 的 MoE expert 量化敏感、可能掉点超 0.4 | 实测未发生（W4A16 delta 0.0671 / W4A4 delta 0.3241）；真实风险转移到覆盖率口径 | §7 |
| 11 | 用 `cann-perf` 采集算子级耗时 | 未引入；脚本内建统计已足够，性能归因另附 `bench_compensate.py` | §6.3 |

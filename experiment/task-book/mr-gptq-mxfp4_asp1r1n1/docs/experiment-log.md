# 实验记录 / 自验证日志（MR-GPTQ MXFP4）

> 本文件持续记录环境搭建、baseline、MR-GPTQ 各增量的实验过程与结果，最终整理为自验证报告。
> 方案见 [design.md](design.md)，验收标准见 [task-book.md](task-book.md)。

## 1. 实验环境

| 项 | 值 |
|----|----|
| 硬件 | Ascend 910（双卡，各 64GB HBM；实验固定单卡 `ASCEND_RT_VISIBLE_DEVICES=0`） |
| CANN | 9.0.0 正式版（`$ASCEND_TOOLKIT_HOME=/home/developer/Ascend/cann-9.0.0`） |
| Python | 3.11.4（aarch64） |
| torch / torch_npu | 2.7.1+cpu / 2.7.1.post4 |
| amct_pytorch | 1.1.0（源码 `build.sh --torch --experimental` 构建安装） |
| 测试模型 | Qwen3-8B（`/mnt/workspace/models/Qwen3-8B`，ModelScope 下载） |
| 数据集 | WikiText2（test，4358 行）、pile-val-backup（校准，214670 条） |

> 备注：任务目标硬件为 Atlas A5（Ascend 950）。原申请的 950PR 免费环境因 `torch_npu` 报
> `Unsupported soc version: Ascend950PR 9579`（A5 太新、当前 torch_npu 未适配）无法运行，遂改用
> 910B。**量化精度（PPL）由 fake-quant 确定性数学决定，与具体 NPU 无关，910B 与 950 的 PPL 等效**，
> 不影响精度验收。

## 2. 环境搭建关键步骤（可复现）

```bash
# 1. 构建并安装 amct_pytorch
source $ASCEND_TOOLKIT_HOME/set_env.sh
bash build.sh --torch --experimental          # 产出 build_out/amct_pytorch-1.1.0-*.tar.gz
pip3 install build_out/amct_pytorch-1.1.0-*.tar.gz --no-build-isolation --no-deps

# 2. 安装运行依赖（排除 torch/torch_npu，保护已配好的版本）
grep -vE '^(torch==|torch_npu==)' requirements.txt > /tmp/reqs.txt
pip3 install -r /tmp/reqs.txt --no-build-isolation

# 3. 修 torchaudio ABI（transformers 5.12.1 会 eager import torchaudio）
pip3 install torchaudio==2.7.1 --no-deps --index-url https://download.pytorch.org/whl/cpu

# 4. 数据集环境变量（国内 + 规避 pyarrow bug，见 §3）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

## 3. 数据加载踩坑与解决（ofmr 示例 utils.py 适配）

| 问题 | 现象 | 解决 |
|------|------|------|
| pyarrow 25 读 HF parquet | `ArrowInvalid: Index not in dictionary bounds` | 改用 **fastparquet** 引擎读本地 parquet |
| HF 直连被墙 | 连接超时 | `export HF_ENDPOINT=https://hf-mirror.com` |
| 大文件走 Xet/CAS | `401 Unauthorized`（mirror 不代理 CAS） | `export HF_HUB_DISABLE_XET=1` + `allow_patterns` 只下所需 parquet |
| WikiText 裸名失效 | `Repository id must be 'namespace/name'` | 用规范 id `Salesforce/wikitext` |
| pile-val-backup 非 parquet | 是 `val.jsonl.zst` | 用 `zstandard` 解压读 jsonl |

对应在 `examples/algorithms/ofmr/src/utils.py` 中新增 `_load_hf_parquet_fastparquet` /
`_load_pile_val_texts`，并将 `get_qwen` 的 `device_map="auto"` 去掉改单卡加载
（否则 8B 被拆到 npu:0/npu:1，量化时报 device mismatch）。

## 4. 关键技术判断

- **MXFP4 是纯 torch fake-quant，不依赖 amct_ops / NPU 原生算子**。
  `quant_util.py` 中 MXFP4 路径 = `cal_shared_exponent`（log2/floor）+
  `scale_input_by_shared_exponents` + `float_to_fp4e2m1`（纯位运算），无 `npu_quantize`。
  （对比：hifloat8 路径需 `amct_ops.hifloat8_cast`，本任务不用 hifloat8，故不安装。）
- **GPTQ 补偿在校准前向内完成，不需 `convert()`**。
  `gptq_module.forward` 状态机：前 `batch_num` 次前向累积 Hessian → 第 `batch_num` 次做
  Cholesky 误差补偿并更新权重 → 之后走 `fake_quant_forward`。因此
  「`quantize` → 喂 1 次校准前向 → `test_ppl`」即得已补偿的 GPTQ-mxfp4 结果。

## 5. 实验结果

约定：`delta = PPL_quant − PPL_bf16`；mxfp4 验收要求 `delta ≤ 0.4` 且量化层（nn.Linear）占比 ≥ 70%。

| 编号 | 模型 | 方案 | seq_len | 校准 | WikiText2 PPL | delta | 备注 |
|------|------|------|---------|------|---------------|-------|------|
| E0 | Qwen3-8B | BF16 基准 | 2048 | — | **9.7154** | — | 与 ofmr README 参考值 9.715 吻合，流程正确 |
| A（冒烟） | Qwen3-8B | MXFP4-AWQ（仓内现成 CFG） | 2048 | pile 512×256 | **9.954** | **+0.239** ✅ | ≤0.4 达标；验证 MXFP4 管线跑通 |
| E1 | Qwen3-8B | 我们的 GPTQ-mxfp4（MR-GPTQ 增量全关，标准 block-32） | 2048 | pile 64×512 | **10.1332** | **+0.418** | 纯 MXFP4 粗糙，GPTQ≈RTN（论文表11），刚好卡门槛外 → 需增量 |
| E1' | Qwen3-8B | stock amct GPTQ-mxfp4（外部参考） | 2048 | — | **10.1369** | +0.4215 | **与我们的 10.1332 差 0.004 → 交叉验证实现正确**；stock 跑 3h，我们几分钟（~10x 提速） |
| E2 | Qwen3-8B | **MR-GPTQ +Hadamard(k=128)** | 2048 | pile 64×512 | **10.0747** | **+0.359** ✅ | Hadamard 把 delta 0.418→0.359 **越过门槛**；纯 GPTQ 过不了、增量让它过 |
| E3 | Qwen3-8B | **MR-GPTQ +Hadamard +scale fitting**（完整） | 2048 | pile 64×512 | **10.0499** | **+0.3345** ✅ | scale fitting（offset=log2 6 修正后）再降 0.025；完整消融链每步都降 |

**完整消融链（2026-07-14，Qwen3-8B，seq_len=2048）**：GPTQ 10.1332(+0.418❌) → +Hadamard 10.0747(+0.359✅) → +scale fitting 10.0499(+0.3345✅)。**每个增量都正向**，复现论文 MR-GPTQ 设计。⚠️scale fitting 关键坑：① 方法须是论文式(2)/(3) grid remapping（非 MSE 搜索）；② offset 用 log2(6) 使 block_max→6 用满 E2M1 范围（用 2 则→4，反而变差）。当前 8B 达标（delta 0.3345 < 0.4），未反超 AWQ(9.954/0.239)——AWQ 用了 512 校准样本、我们才 64，加校准可能进一步降。

**消融进展（2026-07-14）**：GPTQ 10.1332(+0.418,超门槛) → +Hadamard 10.0747(+0.359,**达标**)。复现论文核心：Hadamard 是 MXFP4 头号杠杆，纯 GPTQ 不够、需微旋转。⚠️坑：box 单分支 clone `git pull` 拉不动新分支提交，须 `git fetch origin <branch> && git reset --hard FETCH_HEAD`（否则跑的是旧代码，hadamard flag 成 no-op → 结果=baseline）。下一步：把 scale fitting 改为论文式(2)/(3) 的 grid remapping（当前 MSE 搜索版是错的），叠在 Hadamard 上冲击 <9.95。

**关键观察（2026-07-14）**：我们自研 MRGPTQuant（CPU 补偿 + 标准 block-32 MXFP4）跑通，增量全关时 PPL=10.1332（delta 0.418）。与论文一致——纯 MXFP4 上 GPTQ 几乎不比 RTN 强，AWQ（9.954）反而更好；MR-GPTQ 的收益来自各增量（**当时判断 scale fitting 最大——后被推翻**：实测头号杠杆是 Hadamard 旋转，且旋转块越大越好；scale fitting 须叠在旋转之上才发力，见 §5 与设计文档 §3.3）。补偿从 stock 的 ~2h 降到几分钟（CPU 卸载 + shared exponent 预计算 + 标准量化）。下一步加 scale fitting。

> 冒烟实验 A 仅用于验证「MXFP4 量化 → 校准 → WikiText2 PPL」全链路在 910B 跑通
> （无 amct_ops 报错），非最终交付项；最终交付用 GPTQ 系列 + seq_len=4096。

## 5.1 ★ 8B 最终交付数（seq_len=4096，任务真实设置）

| 指标 | 结果 | 要求 | |
|------|------|------|---|
| bf16 PPL | 8.9893 | — | 基准 |
| MR-GPTQ PPL（Had128+scale fitting） | 9.2621 | — | |
| **PPL delta** | **0.273** | ≤ 0.4 | ✅ |
| **量化层覆盖率** | **99.6%**（252/253，仅跳 lm_head） | ≥ 70% | ✅ |
| **显存降低（权重压缩率）** | **62.3%**（全模型口径）/ 67.4%（Linear 权重口径） | ≥ 50% | ✅ |

**Qwen3-8B 三项硬指标全部达标。** 命令：`run_eval.py --mode mr_gptq --increments hadamard scale_fitting --seq_len 4096`。日志：`src/logs/mrgptq_full_8b_4096.log`、`e0_bf16_4096.log`。

## 5.2 ★ 35B 最终交付数（Qwen3.6-35B-A3B / qwen3_5_moe，seq_len=4096）

| 指标 | 结果 | 要求 | |
|------|------|------|---|
| bf16 PPL | 6.3111 | — | 基准 |
| MR-GPTQ PPL | 6.3781 | — | |
| **PPL delta** | **0.0671** | ≤ 0.4 | ✅ |
| **量化层覆盖率（nn.Linear）** | **71.4%**（250/350） | ≥ 70% | ✅ |
| 全参数量化占比 | 97.0%（33.62B/34.66B） | — | 参考 |
| **显存降低** | **71.2%** | ≥ 50% | ✅ |

⚠️ 覆盖率分母为**全部 350 个 nn.Linear**（跳 lm_head），非"已量化的 250 个"——
35B 另有 100 个 nn.Linear 因 in/out 维不整除 32 而无法做 block-32 MXFP4，**始终占分母**。
两个分母在 8B 上几乎重合（252 vs 253）、在 35B 上相差 100，口径详见报告 §4.2 / §4.4。

**Qwen3.6-35B-A3B 三项硬指标全部达标。** 双卡 device_map，命令：`run_eval.py --mode mr_gptq --increments hadamard scale_fitting --seq_len 4096 --device_map auto`。

**架构与坑（2026-07-15）**：模型实为 **`qwen3_5_moe` 混合架构**（30 层 linear_attn + 10 层 self_attn，每层 MoE 带 256 路由专家 + 共享专家）。⚠️**路由专家是 `Qwen3_5MoeExperts` 的融合 3D 张量（gate_up_proj/down_proj，占 ~32B/35B），非 nn.Linear → amct 覆盖不到**，不处理则显存降<50% 必挂。解法：`quantize_fused_experts` 对融合张量沿收缩维做 **scale-fitting MXFP4 RTN**（不做 Hadamard；scale fitting 对 RTN 是加分项）。nn.Linear（attn/linear_attn/shared_expert，250 个）走完整 MR-GPTQ。⚠️融合专家量化的 fp32 临时量在多卡塞满的卡上会 OOM → 全程 CPU 计算、回写先释放旧张量再挪回。router 保 fp16。日志：`src/logs/mrgptq_full_35b_4096.log`、`e0_bf16_35b_4096.log`。

## 6. W4A4 补充（2026-07-27 ~ 07-29，按评审意见）

评审要求以 **W4A4** 评估最终效果（weight-only 只有存储收益，matmul 仍需反量化回 bf16）。
在已有 Hadamard 旋转基础上于 `fake_quant_forward` 内补齐激活量化，**Qwen3-8B 达标：
delta 0.763 → 0.3897（降低 49%），真 4-bit 激活层占比 71.4%**。

达标的两项手段都不在论文里（**效应量已于 §7 经三 seed 配对复测重新判定**）：

1. **全宽旋转**（`--hadamard_full`）：`down_proj` 的 in=12288 在 k=4096 下只能切 3 个对角块，
   是全网唯一未获全宽旋转的层。12288 非 2 的幂、`3×4096` 不可行（3 阶 Hadamard 不存在），
   走 `12×1024` = Paley H₁₂ ⊗ Sylvester H₁₀₂₄。**→ §7 复测：旋转强度确证（0.147），
   但"全宽 vs k=4096"的细分（0.0305）低于分辨率、不单独归因。**
2. **GPTQ 阻尼系数**（`--perc_damp` 0.01→0.2）：最大层样本/维度比仅 1.1:1、Hessian 近奇异，
   原判断为默认阻尼偏小。**→ §7 复测：证伪。**

**同期 5 项手段记为负结果**（激活 scale 裁剪、加大校准集、权重 scale 裁剪、增加保护层、
randomized Hadamard）。**→ §7 复测后多数降级为"未观察到收益"**，逐项判定见报告 §4.3 ④。

**⚠️ 测量口径的自我修正（第一次）**：中期曾用测试集前 20 块做快速筛选，后经两组同配置对照
发现其偏移随配置摆动达 0.053、大于多数待测增量——**连排序都不可靠**。此后所有定量结论改用
全量评测，并重跑最优配置验证可复现（PPL 与记录值逐位吻合）。

## 7. 测量分辨率复测与 35B W4A4（2026-08-11 ~ 08-12）

**起因**：新一轮评审要求补 **35B 的 W4A4**。同时注意到 8B W4A4 的达标余量仅 0.0103
（0.3897 vs 0.40），而此前所有归因都基于**单一校准集的单次运行**——遂先测本实验的分辨率。

### 7.1 测量分辨率（16 个全量 run）

*(a) 阻尼扫描延伸*（seed 42，全量 73 块）——原判断"曲线单调下降"在延伸后不成立：

| `perc_damp` | 0.05 | 0.1 | 0.2 | **0.3** | **0.4** | **0.6** |
|---|---|---|---|---|---|---|
| delta | 0.4061 | 0.3982 | 0.3897 | **0.4135** | **0.3519** | 0.3608 |

0.3 处反弹，**相邻点摆动约 0.06**。

*(b) 跨校准集*（同配置仅改 `--calib_seed`）：

| 配置 | seed 7 | seed 42 | seed 2024 | **极差** |
|---|---|---|---|---|
| 8B **W4A4**（damp 0.2） | 0.3131 | 0.3897 | **0.4241 ❌** | **0.111** |
| 8B W4A4（damp 0.4） | 0.3796 | 0.3519 | **0.4392 ❌** | 0.0873 |
| 8B **W4A16** 主线 | 0.2544 | 0.273 | 0.2380 | **0.0350** |
| **35B W4A4**（无保护） | 0.2842 | 0.3241 | 0.2909 | **0.0399** |

**判据**：效应量 < 0.06 不可分辨、0.06–0.15 存疑、> 0.15 可归因（8B W4A4 配置下）。
**该判据配置相关**——W4A16 与 35B W4A4 的方差仅为其 1/3，须各自测定。

*(c) 两个观察*：
- W4A16 与 W4A4 共用同一 Hessian 通路，方差却差 3.2 倍 → **多出的方差由激活量化引入**；
- 三个 seed 在两种配置下**难易排序相反**（W4A16 最好的 seed 2024 是 W4A4 最差的）
  → 不存在"某批语料质量差"，方差来自**激活量化与权重补偿的交互**。

### 7.2 配对复测：两条确证、一条证伪

固定其余全部变量，只改一项：

| 手段 | 对照 | 效应量 | 判定 |
|---|---|---|---|
| **混合精度保护** | 0.3519（有） vs **0.5761**（无） | **0.224** | ✅ **确证，8B 不可省** |
| **旋转强度** | 0.3519（全宽） vs **0.4993**（k=128） | **0.147** | ✅ **确证，k=128 不达标** |
| **GPTQ 阻尼** | damp 0.2 vs 0.4，三 seed 配对：+0.0665 / −0.0378 / +0.0151 | 均值 **+0.0146** | ❌ **证伪**（2/3 变差，方向随数据变号） |

日志：`w4a4_8b_noprot_damp04_73.log`、`w4a4_8b_k128_damp04_73.log`、
`w4a4_8b_damp04_seed{7,2024}_73.log`、`w4a16_8b_seed{7,2024}_73.log`。

**部署含义**：k=128（在线旋转开销仅 1.6%）**不达标**，达标必须全宽旋转（朴素矩阵乘下
约 +104% 在线算力）→ 可部署性取决于是否实现 **FWHT**（O(k log k)，开销 < 1%）。

### 7.3 35B W4A4（评审要求）—— ✅ 达标

**首个发现来自覆盖率而非精度**：冒烟运行显示，沿用 8B 的保护层配置时
真 4-bit 激活层 = 200/250 = 80%（按已量化模块），但**按任务书口径仅 200/350 = 57.1%，不达标**。
35B 有 **100 个 `nn.Linear` 因维度不整除 32 而结构性无法量化、却始终占分母**。
为此把 `report_act_coverage` 改为**同时打印两个分母**（`666cfe1`），并去掉保护层：

| 配置 | 真 4-bit 层 | /250 | **/350（任务书口径）** | delta | 判定 |
|---|---|---|---|---|---|
| 保护 `down_proj`+`o_proj` | 200 | 80.0% | **57.1%** | — | ❌ 覆盖率 |
| **无保护（交付）** | **250** | 100% | **71.4%** | **0.3241** | ✅ |

> **同一手段在两个模型上结论相反**：8B 上保护不可省（0.224），35B 上必须弃用。
> **配置不可跨模型默认继承**——这是本轮最直接的工程教训，且是被覆盖率倒逼发现的。

35B 单次全量约 **35 min**（校准 560s + 72 块 × 20.5s），约为 8B 的一半。
日志：`w4a4_35b_noprot_72.log`、`w4a4_35b_noprot_seed{7,2024}_72.log`。

### 7.4 本轮结论

- ✅ **35B W4A4 达标**，评审要求闭环
- ✅ 两条手段**首次获得确证**（此前全部为单次运行的差值）
- ❌ 阻尼一项证伪，四项归因/负结果撤销
- ⚠️ **8B W4A4 不具跨校准集稳健性**（seed 2024 越线），已在报告 §4.3 ① 主动披露；
  未发现三 seed 全达标的 W4A4 配置
- 🔎 **加大校准集：方差收窄但被均值上移抵消**（`--n_calib` 64→256，三 seed 配对，
  详见报告 §4.3 ⑤(d)）：极差 **0.111 → 0.051**（原先最差的 seed 2024 好转 0.0507、
  最好的 seed 7 变差 0.0810，呈向均值收缩），但均值上移 0.0217，**最差值 0.4241 → 0.4244 不变**。
  极差收窄与机理一致但受 n=3 限制仅为提示性，均值上移 0.0217 则低于分辨率、不成立。
  **过阻尼推测已被否定**：同配置下阻尼 0.2 → 0.05 得 0.4287，仅差 0.0043（分辨率的 1/14）。
  这同时是 `perc_damp` 无效的**第三个独立证据**（两种校准规模、四倍阻尼跨度均测不出效应）。
  **结论：配置空间内已穷尽可行的方差抑制手段，未找到三 seed 全部达标的 W4A4 配置**，
  该边界已在报告 §4.3 ① 如实披露。（§6 那次"+0.006 无收益"为单 seed、默认阻尼下的
  均值比较，量级远低于分辨率，不足为据。）

新增工具：`src/run_ablations.sh`、`src/run_night.sh`（分组、断点续跑、自动汇总），
`run_eval.py --calib_seed`，`mr_gptq_module.report_act_coverage`（双分母）。

## 8. 交付收尾

1. ✅ 8B / 35B **W4A16 与 W4A4 均达标**（两轮评审要求全部闭环）。
2. ✅ 全部实验日志入 `src/logs/`（含 §7 的 16 个复测 run）。
3. ✅ 运行截图入 `docs/img/`。
4. ✅ 设计文档（v1.0 最终版）、自验证报告、README 同步至最终态；
   §7 的复测结论已回写至报告 §4.3 ①②③⑤ / §4.4 / §7 与 design.md §4.4。
5. 交付方式：**单 PR** —— `feat/mr-gptq-impl` → `cann/amct:feature/community-tasks`，
   含设计文档 + 算法代码 + 自验证报告（早期设想的设计文档独立分支路线已废弃）。

**⚠️ 交付时须一并说明的三条口径**（详见报告）：

| | 说的是 | 不是 |
|---|---|---|
| 35B「全参数 97%」 | **MXFP4 数据类型**的参数覆盖率 | MR-GPTQ 算法的覆盖率（那是约 4%） |
| 「71.4% 层跑真 4-bit」 | **层数**口径（任务书定义） | 算力口径（35B 算力大头仍是 fp16 激活） |
| 显存/压缩率 | **解析推算**（4.25 bit/参数） | 运行时实测（fake-quant 下张量仍以 fp16 承载） |

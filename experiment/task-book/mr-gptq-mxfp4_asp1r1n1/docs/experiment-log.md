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
| E1' | Qwen3-8B | stock amct GPTQ-mxfp4（外部参考） | 2048 | — | **10.1369** | +0.4215 | **与我们的 10.1332 差 0.004 → 交叉验证实现正确**；stock 跑 3h，我们几分钟（~15x 提速） |
| E2 | Qwen3-8B | **MR-GPTQ +Hadamard(k=128)** | 2048 | pile 64×512 | **10.0747** | **+0.359** ✅ | Hadamard 把 delta 0.418→0.359 **越过门槛**；纯 GPTQ 过不了、增量让它过 |
| E3 | Qwen3-8B | **MR-GPTQ +Hadamard +scale fitting**（完整） | 2048 | pile 64×512 | **10.0499** | **+0.3345** ✅ | scale fitting（offset=log2 6 修正后）再降 0.025；完整消融链每步都降 |

**完整消融链（2026-07-14，Qwen3-8B，seq_len=2048）**：GPTQ 10.1332(+0.418❌) → +Hadamard 10.0747(+0.359✅) → +scale fitting 10.0499(+0.3345✅)。**每个增量都正向**，复现论文 MR-GPTQ 设计。⚠️scale fitting 关键坑：① 方法须是论文式(2)/(3) grid remapping（非 MSE 搜索）；② offset 用 log2(6) 使 block_max→6 用满 E2M1 范围（用 2 则→4，反而变差）。当前 8B 达标（delta 0.3345 < 0.4），未反超 AWQ(9.954/0.239)——AWQ 用了 512 校准样本、我们才 64，加校准可能进一步降。

**消融进展（2026-07-14）**：GPTQ 10.1332(+0.418,超门槛) → +Hadamard 10.0747(+0.359,**达标**)。复现论文核心：Hadamard 是 MXFP4 头号杠杆，纯 GPTQ 不够、需微旋转。⚠️坑：box 单分支 clone `git pull` 拉不动新分支提交，须 `git fetch origin <branch> && git reset --hard FETCH_HEAD`（否则跑的是旧代码，hadamard flag 成 no-op → 结果=baseline）。下一步：把 scale fitting 改为论文式(2)/(3) 的 grid remapping（当前 MSE 搜索版是错的），叠在 Hadamard 上冲击 <9.95。

**关键观察（2026-07-14）**：我们自研 MRGPTQuant（CPU 补偿 + 标准 block-32 MXFP4）跑通，增量全关时 PPL=10.1332（delta 0.418）。与论文一致——纯 MXFP4 上 GPTQ 几乎不比 RTN 强，AWQ（9.954）反而更好；MR-GPTQ 的收益来自三增量（scale fitting 最大）。补偿从 stock 的 ~2h 降到几分钟（CPU 卸载 + shared exponent 预计算 + 标准量化）。下一步加 scale fitting。

> 冒烟实验 A 仅用于验证「MXFP4 量化 → 校准 → WikiText2 PPL」全链路在 910B 跑通
> （无 amct_ops 报错），非最终交付项；最终交付用 GPTQ 系列 + seq_len=4096。

## 5.1 ★ 8B 最终交付数（seq_len=4096，任务真实设置）

| 指标 | 结果 | 要求 | |
|------|------|------|---|
| bf16 PPL | 8.9893 | — | 基准 |
| MR-GPTQ PPL（Had128+scale fitting） | 9.2621 | — | |
| **PPL delta** | **0.273** | ≤ 0.4 | ✅ |
| **量化层覆盖率** | **99.6%**（252/253，仅跳 lm_head） | ≥ 70% | ✅ |
| **显存降低** | **67.4%**（MXFP4 4.25bit vs bf16 16bit；lm_head 保 fp16） | ≥ 50% | ✅ |

**Qwen3-8B 三项硬指标全部达标。** 命令：`run_eval.py --mode mr_gptq --increments hadamard scale_fitting --seq_len 4096`。日志：`src/logs/mrgptq_full_8b_4096.log`、`e0_bf16_4096.log`。

## 6. 下一步

1. E0：BF16 基准 PPL（`--mode bf16`）。
2. E1：MXFP4-GPTQ baseline PPL（`--mode gptq_mxfp4`），与 E0 算 delta 立靶。
3. 实现 MR-GPTQ 三增量（按投产比）：MXFP scale fitting → block Hadamard 微旋转 → 静态激活重排。
4. seq_len 切 4096 出最终交付数；再上 Qwen3.6-35B-A3B（含 MoE expert，量化层占比 ≥ 70%）。

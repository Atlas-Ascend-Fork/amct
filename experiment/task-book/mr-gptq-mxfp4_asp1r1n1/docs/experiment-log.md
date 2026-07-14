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
| E1 | Qwen3-8B | MXFP4-GPTQ（自建 CFG） | 2048 | pile 64×512 | *待测* | *待测* | baseline 对照 |
| — | Qwen3-8B | **MR-GPTQ**（本任务） | 4096 | — | *待实现* | 目标 < E1 | 三增量 |

> 冒烟实验 A 仅用于验证「MXFP4 量化 → 校准 → WikiText2 PPL」全链路在 910B 跑通
> （无 amct_ops 报错），非最终交付项；最终交付用 GPTQ 系列 + seq_len=4096。

## 6. 下一步

1. E0：BF16 基准 PPL（`--mode bf16`）。
2. E1：MXFP4-GPTQ baseline PPL（`--mode gptq_mxfp4`），与 E0 算 delta 立靶。
3. 实现 MR-GPTQ 三增量（按投产比）：MXFP scale fitting → block Hadamard 微旋转 → 静态激活重排。
4. seq_len 切 4096 出最终交付数；再上 Qwen3.6-35B-A3B（含 MoE expert，量化层占比 ≥ 70%）。

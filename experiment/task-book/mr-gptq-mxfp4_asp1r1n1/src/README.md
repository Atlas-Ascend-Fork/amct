# 算法实现与评测脚本

MR-GPTQ（MXFP4）的完整实现与 WikiText2 PPL 评测。数据加载已封装环境规避逻辑，
无需修改仓内 `examples/`。

## 文件

| 文件 | 作用 |
|------|------|
| `mr_gptq_module.py` | **★ 核心交付**：MR-GPTQ 量化算子。子类化 amct `GPTQuant`，`import` 即运行时注册 `mr_gptq`，**零改 amct 源码** |
| `run_eval.py` | 入口：`--mode {bf16, gptq_mxfp4, mr_gptq}`，增量与 W4A4 开关均在此 |
| `eval_utils.py` | 模型加载（单卡 / 多卡 device_map）、校准前向、PPL 评测、覆盖率与显存统计 |
| `data.py` | WikiText2 / pile-val 加载（fastparquet、jsonl.zst、HF 镜像、禁 Xet 已内置） |
| `run_ablations.sh` | 消融批跑：分组（`s1`/`a2`/`b1`/`b2`/`c1`/`c2`）、断点续跑、自动汇总 delta |
| `run_night.sh` | 夜间队列：测量分辨率复测与 35B W4A4（8 个 run，约 7h） |
| `logs/` | 全部实验原始日志（自验证报告的证据） |

## 依赖（首次）

```bash
pip3 install fastparquet zstandard --no-build-isolation
# amct_pytorch 及其余依赖见 ../docs/experiment-log.md §2
```

## 运行

```bash
cd experiment/task-book/mr-gptq-mxfp4_asp1r1n1/src
export ASCEND_RT_VISIBLE_DEVICES=0        # 单卡；8B 跨卡会报 device mismatch
export PYTHONUNBUFFERED=1                 # 实时看逐层补偿进度

# E0：bf16 基准
python3 run_eval.py --model_path <Qwen3-8B> --mode bf16 --seq_len 4096 \
    2>&1 | tee logs/e0_bf16.log

# E1：MXFP4 + 仓内 GPTQ（baseline 对照）
python3 run_eval.py --model_path <Qwen3-8B> --mode gptq_mxfp4 --seq_len 4096 \
    2>&1 | tee logs/e1_gptq_mxfp4.log

# E2：MR-GPTQ 主线 W4A16（delta 0.273）
python3 run_eval.py --model_path <Qwen3-8B> --mode mr_gptq --seq_len 4096 \
    --increments hadamard scale_fitting \
    2>&1 | tee logs/e2_mr_gptq.log

# E3：MR-GPTQ W4A4 达标配置（delta 0.3897）
python3 run_eval.py --model_path <Qwen3-8B> --mode mr_gptq --seq_len 4096 \
    --hadamard_k 4096 --hadamard_full --perc_damp 0.2 \
    --act_protect down_proj o_proj \
    --increments hadamard scale_fitting act_quant act_scale_fit joint_comp \
    2>&1 | tee logs/e3_w4a4.log

# 35B：双卡，追加 --device_map auto
export ASCEND_RT_VISIBLE_DEVICES=0,1
python3 run_eval.py --model_path <35B> --mode mr_gptq --seq_len 4096 \
    --increments hadamard scale_fitting --device_map auto
```

## 主要参数

| 参数 | 说明 |
|------|------|
| `--seq_len` | PPL 分块长度。**交付数据一律用 4096**（任务要求）；2048 仅用于与公开工作同协议对照 |
| `--increments` | MR-GPTQ 增量开关。`hadamard` / `scale_fitting` 为主线；`act_quant` 起 W4A4，`act_scale_fit` / `joint_comp` 为其优化项 |
| `--hadamard_k` | 旋转块大小。论文默认 128 是**在线算子开销**的权衡；本项目为 fake-quant 精度评估，实测 **k=128 不达标（0.4993）、须全宽**。⚠️ 代价：全宽旋转在朴素矩阵乘下约 +104% 在线算力，可部署性取决于是否实现 FWHT（见报告 §4.3 ③(a)） |
| `--hadamard_full` | 每层旋转块取满 `in_features`。8B 上唯一受影响的是 `down_proj`(in=12288)。「k=128 → 全宽」整步**效应量 0.147，已确证** |
| `--perc_damp` | GPTQ Hessian 阻尼。⚠️ **经三 seed 配对复测已证伪**（2/3 变差、均值变差、方向随数据变号），交付配置固化取值但**不作为精度手段**，见报告 §4.3 ③(c) |
| `--act_protect` | W4A4 混合精度：层名含指定子串的层保激活高精度（子串匹配，注意勿误伤）。**8B 上不可省（效应量 0.224）；35B 上必须弃用**（会把任务书口径的占比压到 57.1%） |
| `--calib_seed` | 校准语料抽样种子（默认 42，交付数据以此为准）。⚠️ 换种子会改变结果——8B W4A4 三 seed 极差 **0.111**，W4A16 仅 0.0350，见报告 §4.3 ⑤ |
| `--n_calib` | 校准语料**条数**（不是块数！）。总 token ≈ 条数 × 平均条长；只调大 `--calib_block` 反而会减少总量 |
| `--calib_chunk` | 校准前向分块，仅降峰值显存（与整块数学等价）。放大 `--n_calib` 时配合使用 |
| `--max_eval_samples` | 只评前 N 块。⚠️**不要用于定量结论**——实测该子集偏移随配置摆动达 0.053，见报告 §4.3 ⑤ |

`act_clip` / `weight_clip` / `act_fit_rowwise` / `--hadamard_random` 为**未采纳的手段**，
默认关闭、代码保留作消融记录。⚠️ 其中多数的实测效应量低于本实验分辨率（0.06），
只能表述为「未观察到收益」；经复测仍站得住的只有 `--hadamard_random` 有害（0.138）。
逐项判定见报告 §4.3 ④。

## 说明

- 运行即打印：量化层覆盖率（`[coverage]` / `[actual-coverage]`）、W4A4 下的真 4-bit 激活层占比
  （`[act-coverage]`，**同时给两个分母，任务书口径为"/ 全部 nn.Linear"**）、
  融合专家量化（`[fused-experts]`）、显存降低（`[memory]` / `[memory-full]`）、
  逐层补偿进度、校准耗时、WikiText2 PPL（`Score:`）。
- `HF_ENDPOINT` / `HF_HUB_DISABLE_XET` 已在 `data.py` 内设默认值，无需手动 export。
- 日志统一 `tee` 到 `logs/`（进 git，作为自验证报告的原始记录）。

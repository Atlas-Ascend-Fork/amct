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
| `--hadamard_k` | 旋转块大小。论文默认 128 是**在线算子开销**的权衡；本项目为 fake-quant 精度评估、旋转不计推理成本，实测 4096 更优 |
| `--hadamard_full` | 每层旋转块取满 `in_features`。8B 上唯一受影响的是 `down_proj`(in=12288)——**W4A4 达标的关键之一** |
| `--perc_damp` | GPTQ Hessian 阻尼。最大层样本/维度比仅约 1.1:1、Hessian 近奇异，默认值偏小，**另一个达标关键** |
| `--act_protect` | W4A4 混合精度：层名含指定子串的层保激活高精度（子串匹配，注意勿误伤） |
| `--n_calib` | 校准语料**条数**（不是块数！）。总 token ≈ 条数 × 平均条长；只调大 `--calib_block` 反而会减少总量 |
| `--calib_chunk` | 校准前向分块，仅降峰值显存（与整块数学等价）。放大 `--n_calib` 时配合使用 |
| `--max_eval_samples` | 只评前 N 块。⚠️**不要用于定量结论**——实测该子集偏移随配置摆动达 0.053，见报告 §4.3 ⑤ |

`act_clip` / `weight_clip` / `act_fit_rowwise` / `--hadamard_random` 为**实测负结果**，
默认关闭、代码保留作消融记录，逐项分析见报告 §4.3 ④。

## 说明

- 运行即打印：量化层覆盖率（`[coverage]` / `[actual-coverage]`）、显存降低（`[memory]` / `[memory-full]`）、
  逐层补偿进度、校准耗时、WikiText2 PPL（`Score:`）。
- `HF_ENDPOINT` / `HF_HUB_DISABLE_XET` 已在 `data.py` 内设默认值，无需手动 export。
- 日志统一 `tee` 到 `logs/`（进 git，作为自验证报告的原始记录）。

# 测试脚本（自包含评测工具）

复现 baseline 与 MR-GPTQ 的 WikiText2 PPL。数据加载已封装环境规避逻辑，
无需修改仓内 `examples/`。

## 文件

| 文件              | 作用                                                                        |
| ----------------- | --------------------------------------------------------------------------- |
| `data.py`       | WikiText2 / pile-val 加载（fastparquet、jsonl.zst、HF 镜像、禁 Xet 已内置） |
| `eval_utils.py` | 单卡模型加载、GPTQ 校准前向、PPL 评测                                       |
| `run_eval.py`   | 入口：`--mode {bf16, gptq_mxfp4, mr_gptq}`                                |

## 依赖（首次）

```bash
pip3 install fastparquet zstandard --no-build-isolation
# amct_pytorch 及其余依赖见 ../docs/experiment-log.md §2
```

## 运行

```bash
cd experiment/task-book/mr-gptq-mxfp4_asp1r1n1/src
export ASCEND_RT_VISIBLE_DEVICES=0          # 单卡，避免 8B 跨卡报错
export PYTHONPATH=/mnt/workspace/gitCode/cann/amct   # 以源码方式跑 amct（改动即时生效）

# E0：bf16 基准
python3 run_eval.py --model_path /mnt/workspace/models/Qwen3-8B --mode bf16 \
    2>&1 | tee logs/e0_bf16.log

# E1：MXFP4 + GPTQ baseline
python3 run_eval.py --model_path /mnt/workspace/models/Qwen3-8B --mode gptq_mxfp4 \
    2>&1 | tee logs/e1_gptq_mxfp4.log

# MR-GPTQ（待 mr_gptq_module.py 实现并注册后）
python3 run_eval.py --model_path /mnt/workspace/models/Qwen3-8B --mode mr_gptq \
    2>&1 | tee logs/e2_mr_gptq.log
```

- 探索用 `--seq_len 2048`（快）；**最终交付数用 `--seq_len 4096`**（任务要求）。
- `--n_calib` 校准样本数，OOM 时调小（如 32）。
- `HF_ENDPOINT` / `HF_HUB_DISABLE_XET` 已在 `data.py` 内设默认值，无需手动 export。

日志统一 `tee` 到 `logs/`（进 git，作为自验证报告的原始记录）。

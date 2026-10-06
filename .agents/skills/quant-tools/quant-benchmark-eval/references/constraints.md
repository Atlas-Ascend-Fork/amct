# 关键约束（踩坑换来的 · 运行前必读）

每次评测前都要过一遍这份清单。编号稳定，脚本注释与本文互相引用。

## 1. attention 量化与 KV cache：按 wrapper 现状判断，不沿用历史结论

`attn-linear` 量化本身不应破坏 KV cache——它只量化 q/k/v/o 投影，不改变 HF `past_key_values` 机制。但部分 amct attention wrapper 只面向 blockwise / PPL 最小路径实现，会忽略 `past_key_values` 并返回 `(attn_output, None)`；这种 wrapper 一旦量化 attention，`generate` 只能 `use_cache=False`，每步全上下文重算，速度退化到分钟级 / 题。

**运行前必须按当前源码重新判断**：目标模型的 attention wrapper 是否已支持 `past_key_values` / `cache_position` / `attention_mask`。

- 已接通 → 允许 `attn-linear` 与 `use_cache=True` 同时用于生成式评测。
- 未接通 → 默认只量化 MLP，attn 保持 bf16，KV cache 正常工作。

补 / 改 KV cache 属于适配工作，本 skill 不默认改源码；只有用户明确要求，或用户确认本次 benchmark 必须先补 wrapper 时才进入源码修改。实现原则、最小正确路径与等价性验证清单见 `$model-adapter` 的「Attention 适配原则」第 6–10 条。

⚠️ **用户给的量化参数里含 attn-linear 时，必须先说明取舍并征得同意，不得静默跳过**——静默跳过等于把用户量化过的 attn 降级回 bf16，被评测对象就不再是完整 W8A8。三条路由用户选，不要替他决定：

| 选项 | 含义 | 代价 |
|---|---|---|
| (a) | 只量化 MLP，attn 保持 bf16 | 评测可行，但**不是完整 W8A8** |
| (b) | 连 attn 一起量化，wrapper 未接 KV cache | 是完整 W8A8，但会极慢 |
| (c) | 连 attn 一起量化，wrapper 已接 KV cache | 可评完整 attention 量化对象，但必须先 `generate` 冒烟 + 小样本限量确认速度 / 显存 |

走 (b) / (c) 必须显式加 `--allow_attn_quant`；脚本默认拦截 `attn-linear` / `attn-cache`。

## 2. 数据必须离线加载，且用 fastparquet 引擎

直接用 HF `datasets` 会有两个问题：远程 glob 在本机镜像上超时（`FSTimeoutError`）；pyarrow 读 MMLU-Pro 的 parquet 会崩溃（`Index not in dictionary bounds`）。

解决办法：先用 `snapshot_download` 把数据集仓下载到本地，再 monkeypatch `datasets.load_dataset`，改用 **fastparquet** 读本地 parquet、pandas 读本地 jsonl（见 `scripts/bench_offline.py`）。运行前用 `scripts/prefetch_data.py` 预下载一次即可，运行期保持 `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1`。

## 3. HFLM 直接接受内存中的模型对象

脚本已处理，无需自定义 LM 子类。运行时 `pretrained not of type str` 的 warning 属正常，可忽略。

## 4. PYTHONPATH 必须把 amct 前置，不能整个覆盖

```bash
export PYTHONPATH=$AMCT_HOME:$PYTHONPATH   # 前置（prepend）
```

若直接覆盖（`PYTHONPATH=$AMCT_HOME`），会丢掉 CANN 的 tbe / te，导致大模型在 aclop 编译阶段崩溃。

## 5. page cache 引起的 OOM

进程以 exit 137 / 143 退出、且 dmesg 显示 cgroup OOM 时，启动前先清一次缓存：

```bash
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

## 6. 速度与规模参考

短答案的 MMLU-Pro 约 6 秒 / 题（仅量化 MLP、KV cache 开启），全量 12032 题约需 20 小时。建议用 `--limit N` 限制每个学科的题数，例如 10 题 / 学科共 140 题、约 50 分钟。

长 CoT 任务（MATH500 / LiveCodeBench）明显更慢，且需要更大的 `--max_gen_toks`：MATH500 常用 1024+，LiveCodeBench 常用 2048+。

## 7. 多卡替换层时必须保留 accelerate hook

脚本已自动处理（取出旧 hook、`.to(dev)` 后重新挂回）。手动改装配逻辑时要注意，否则跨卡 `generate` 会报 `tensors on npu:0 and npu:1`。原理见 `scripts/assemble_and_eval.py` 注释。

## 8. 量化 MoE 跑长 prompt 会在 npu:0 上 OOM

量化专家 forward 有额外中间张量，而 npu:0 上又集中了前半层和 embedding，最先被填满：短输入冒烟能过，长 prompt 就 OOM。这是真实容量不足（非显存碎片），`max_split_size_mb` 无效；应改用 `device_map` 的 `max_memory` 把部分层挪离 npu:0，或缩短序列长度。

## 9. 在独立工作目录里运行，不要直接在 `scripts/` 下跑

amct 会往当前目录写 `amct_log/` 和 `fusion_result.json`。在一个专用运行目录（如 `<workdir>/<task>_run/`）里用完整路径调用脚本，保持 skill 的 `scripts/` 只存代码、不留运行产物。

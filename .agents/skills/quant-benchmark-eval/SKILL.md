---
name: quant-benchmark-eval
description: >-
  在 Ascend NPU 上,使用 lm-eval-harness/LiveCodeBench 官方 runner 对 amct 伪量化模型运行生成式下游
  benchmark (MMLU-Pro、MATH500、LiveCodeBench hidden-tests pass@1),并与 bf16 基线对比精度损失。本 skill 只做评测,不做量化,
  需要事先准备好 ptq 量化参数。当需要验证量化模型在下游任务上的准确率、对比量化与
  bf16 的差距,或获取 WikiText2 PPL 之外的下游精度指标时使用。Validate amct
  fake-quant model accuracy on MMLU-Pro vs bf16 on Ascend NPU.
---

# 量化 Benchmark 评测 (amct + NPU + lm-eval-harness)

本 skill 在 Ascend NPU 上对 **amct 伪量化** 大模型运行生成式下游 benchmark。它只负责
评测:量化(extract_ptq_data 与 ptq)应已由 amct 量化流程完成,本 skill 的工作是把
量化后的模型**装配成一个完整、可 generate 的模型对象**,再交给 lm-eval-harness 打分。

## 何时使用
- 需要跑 MMLU-Pro / MATH500 / LiveCodeBench 等下游 benchmark 来验证量化后的精度
- 需要对比 W8A8 与 bf16 在 MMLU-Pro 上的精度损失
- amct 目前只能测 WikiText2 PPL,而你需要下游任务的准确率

## 状态(如实标注覆盖度)

⚠️ **下表的验证结论均基于特定硬件:2× Ascend 910(每卡 64GB HBM,即 65536 MB)。** 其中的显存相关
判断(能否单卡放下、是否 OOM、要几张卡)和速度数字(约 6 秒/题等)都强依赖这套配置,
换用显存更大/更小、算力不同的卡后结论会变化——例如更大显存的卡上约束 8 的 OOM 可能
不再出现,单卡也可能放得下 35B。迁移到其他硬件时,请把这些当作参考量级,重新实测确认,
不要当成普适结论。

验证情况(硬件:2× Ascend 910 64GB):

| 场景 | 状态 |
|---|---|
| MMLU-Pro,8B W8A8 仅 mlp,单卡 | ✅ 已验证 |
| 35B MoE 双卡 bf16(跨卡 generate + HFLM + 打分) | ✅ 已验证 |
| 35B MoE 双卡 W8A8 装配 + 跨卡 generate 冒烟 | ✅ 已验证(需约束 7 的 hook 修复) |
| 35B MoE 双卡 W8A8 + 长 prompt 评测 | ⚠️ npu:0 OOM(见约束 8,属显存调优) |
| MATH500, lm-eval `hendrycks_math500`/别名 `math500` | ✅ 已接入(需按机器实测耗时) |
| LiveCodeBench code-generation-lite,官方 custom_evaluator hidden-tests pass@1 | ⚠️ 已接入生成官方 custom 输出;评分需安装并调用 lcb_runner |

## 关键约束(踩坑换来的 — 运行前必读)

1. **量化后的 attention 不支持 KV cache。** amct 的 `QuantQwen3Attn.forward` 签名为
   `(hidden_states, position_embeddings, **kwargs)`,会忽略 `past_key_value` 并返回
   `(attn_output, None)`。因此一旦量化了 attn,`generate` 就只能用 `use_cache=False`,
   每步计算量退化为 O(n²),约 7 分钟/题,基本不可用。**所以生成式评测只量化 MLP**:
   attn 保持 bf16,KV cache 正常工作,约 6 秒/题。

   ⚠️ **当用户提供的量化参数里含有 attn-linear 时,必须先向用户说明、征得同意,才能
   继续。** 不要擅自跳过 attn-linear 的加载——这等于把用户量化过的 attn 降级回 bf16,
   会改变被评测的对象、让结果不再代表完整的 W8A8 模型。要向用户讲清这个取舍:
   (a) 只量化 MLP、attn 保持 bf16 → 评测可行(约 6 秒/题),但**不是完整 W8A8**;
   (b) 连 attn 一起量化 → 是完整 W8A8,但受 KV-cache 限制,约 7 分钟/题、基本跑不完。
   由用户选择走哪条,不要默认替用户决定。

2. **数据必须离线加载,且用 fastparquet 引擎。** 直接用 HF `datasets` 会有两个问题:
   远程 glob 在本机镜像上会超时(`FSTimeoutError`),pyarrow 读 MMLU-Pro 的 parquet
   会崩溃("Index not in dictionary bounds")。解决办法是先用 `snapshot_download`
   把数据集仓下载到本地,再 monkeypatch `datasets.load_dataset`,改用 **fastparquet**
   读本地 parquet(见 `scripts/bench_offline.py`)。运行前先用 `scripts/prefetch_data.py`
   预下载一次即可。

3. **HFLM 直接接受内存中的模型对象**(脚本已处理,无需自定义 LM 子类)。运行时
   "pretrained not of type str" 的 warning 属正常,可忽略。

4. **PYTHONPATH 必须把 amct 前置(prepend),不能整个覆盖:**
   `PYTHONPATH=/mnt/workspace/gitCode/yaoguangxiu/amct:$PYTHONPATH`。若直接覆盖,会丢掉
   CANN 的 tbe/te,导致大模型在 aclop 编译阶段崩溃。

5. **page cache 引起的 OOM。** 如果进程以 exit 137/143 退出、且 dmesg 显示 cgroup OOM,
   在启动前先执行 `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` 清一次缓存。

6. **速度与规模参考。** 短答案的 MMLU-Pro 约 6 秒/题(仅量化 MLP、KV cache 开启);
   全量 12032 题约需 20 小时。建议用 `--limit N` 限制每个学科的题数,例如 10 题/学科
   共 140 题、约 50 分钟。长 CoT 任务(如 MATH500 / LiveCodeBench)会明显更慢,且
   需要更大的 `--max_gen_toks`(MATH500 常用 1024+,LiveCodeBench 常用 2048+)。

7. **多卡替换层时需保留 accelerate hook**(脚本已自动处理:取出旧 hook、`.to(dev)`
   后重新挂回)。若手动改装配逻辑需注意,否则跨卡 `generate` 会报
   "tensors on npu:0 and npu:1"。原理见 assemble_and_eval.py 注释。

8. **量化 MoE 跑长 prompt 评测会在 npu:0 上 OOM。** 量化专家 forward 有额外的中间张量,
   而 npu:0 上又集中了前半层和 embedding,最先被填满:短输入冒烟能过,长 prompt 就 OOM。
   这是真实容量不足(非显存碎片),`max_split_size_mb` 无效;应改用 device_map 的
   `max_memory` 把部分层挪离 npu:0,或缩短序列长度。

9. **在独立的工作目录里运行,不要直接在 scripts/ 下跑。** amct 会往当前目录写
   `amct_log/` 和 `fusion_result.json`。建议在类似 `/home/developer/<task>_run/` 的目录
   下运行(用 `python /path/to/scripts/assemble_and_eval.py` 调用),以保持 skill 的
   scripts/ 目录干净——只存代码,不留运行产物。

## 可移植性(目标:其他 Ascend + amct 机器)

本 skill 可以跨机器配置,但**并非平台无关**。具体边界如下:
- **硬依赖(无法去除):** amct(整个装配过程都依赖 amct 的 `MODEL_REGISTRY`、
  `build_quant_block`、`load_selected_layer_ptq_params` 以及 ptq 参数格式)、
  `torch_npu` 与 Ascend NPU(代码使用 `npu:*` 设备)、lm-eval 0.4.12。因此它无法原样
  运行在 GPU/CPU 上,也无法脱离 amct。
- **LiveCodeBench 官方分数依赖:** `lcb_runner`。本 skill 的
  `scripts/livecodebench_eval.py` 只在 Ascend 环境里生成官方 `custom_evaluator` 所需
  JSON(`question_id` + `code_list`),hidden-tests pass@1 由
  `python -m lcb_runner.runner.custom_evaluator` 计算。
- **可经 env 变量配置**(默认值指向原机器;`config.py` 读取它们):
  - `AMCT_HOME`    — amct 源码根目录(用于默认 `--bit_config`)。
  - `BENCH_ASSETS` — 本地数据集缓存根目录。
  - `HF_ENDPOINT`  — prefetch 用的 HF 镜像。
- **数据集注册表**采用数据驱动,集中在 `scripts/datasets.json`(prefetch 与离线 shim
  共用)。MMLU-Pro / MATH500 走 lm-eval;LiveCodeBench 走本 skill 自带
  `scripts/livecodebench_eval.py` 生成官方 custom 输出,再用 `lcb_runner` 评分。
- **模型与量化参数**始终通过 CLI 参数传入(`--model`、`--model_name`、`--param_dir`)。

部署到一台新的 Ascend+amct 机器时:设好上述三个 env 变量、执行
`pip install lm-eval==0.4.12`、预下载数据集,即可运行,路径部分无需改动代码。
可先运行 `python scripts/check_env.py` 做依赖检查;LiveCodeBench 官方评分环境再加
`--livecodebench_score` 检查 `lcb_runner`。

## 环境
```
source <ascend-toolkit>/set_env.sh                          # 例如 /home/developer/Ascend/ascend-toolkit
export ASCEND_RT_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1        # 35B 多卡用 0,1
export AMCT_HOME=/mnt/workspace/gitCode/yaoguangxiu/amct     # 按机器覆盖
export PYTHONPATH=$AMCT_HOME:$PYTHONPATH                     # 必须前置(见约束 4)
export BENCH_ASSETS=/home/developer/bench_assets             # 按机器覆盖
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
```
lm-eval 0.4.12(`pip install --user lm-eval==0.4.12`)、LiveCodeBench 官方 runner
(`pip install --user livecodebench` 或按官方仓 README 安装)、torch 2.7.1+cpu、
torch_npu、transformers、py3.12.9。原机器:2× Ascend 910(64GB)。8B 单卡可放下;
35B 需要 `device_map=auto`(双卡,不允许 offload)。

## 工作流

1. **预下载数据**(一次):
   - 轻量自检:`python scripts/self_check.py`
   - 环境检查:`python scripts/check_env.py`
   - MMLU-Pro:`python scripts/prefetch_data.py --task mmlu_pro`
   - MATH500:`python scripts/prefetch_data.py --task math500`
   - LiveCodeBench v6 单 release:`python scripts/prefetch_data.py --task livecodebench`
   - LiveCodeBench release_v6 累计集(更大):`python scripts/prefetch_data.py --task livecodebench_release_v6`

   缓存都落在 `$BENCH_ASSETS/<subdir>/`。`livecodebench` 默认只下 `test6.jsonl`
   以避免一次下载累计 4GB+ 数据;需要累计版本时显式使用 `livecodebench_release_v6`。
   离线读取用 fastparquet/jsonl;运行前保持
   `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1`。

2. **装配 + 评测**量化模型:
   - 8B dense,单卡:
     `python scripts/assemble_and_eval.py --model <path> --model_name qwen3 \
        --param_dir <ptq_params/mlp> --task mmlu_pro --limit 10 --max_gen_toks 512 \
        --output_dir /tmp/qbe_out`
   - 35B MoE,双卡(设 `ASCEND_RT_VISIBLE_DEVICES=0,1`):
     `... --model_name qwen3_5_moe --param_dir <ptq_params/moe> --device_map auto ...`
   - MATH500,同一装配脚本:
     `python scripts/assemble_and_eval.py --model <path> --model_name qwen3 \
        --param_dir <ptq_params/mlp> --task math500 --limit 50 --max_gen_toks 1024`
     其中 `--task math500` 是别名,实际映射到 lm-eval 的 `hendrycks_math500`。也可以
     直接传 `--task hendrycks_math500` 或 `--task minerva_math500` 做不同 prompt 口径。
   - 装配流程:先用 `_load_float()`(支持 device_map)加载模型,再逐层执行
     `build_quant_block`(KV-safe 目标:dense 用 mlp,MoE 用 moe)、
     `load_selected_layer_ptq_params`、开启量化、置 `eval_mode=True`,最后把量化块
     塞回层列表、**并放在该层原本所在的卡上**。装配完会先做一次 `generate` 冒烟。
     若出现 cpu/disk offload 会直接报错(offload 会破坏量化正确性)。
   - 加 `--bf16` 会跳过量化,用**同一套 harness** 跑原始模型,得到同口径的基线。
     两者都要跑,最后报告两者的差值。
   - 加 `--device_map auto` 走多卡,省略则单卡 npu:0。输入会经 `input_device()`
     路由到 embedding 所在的卡。
   - 默认禁止 `--quant_target attn-linear` / `attn-cache`;若已确认要评完整 attention
     量化对象,必须显式加 `--allow_attn_quant`。脚本会关闭 KV cache,速度会显著变慢。

3. **LiveCodeBench 官方 hidden-tests pass@1**:
   `python scripts/livecodebench_eval.py --model <path> --model_name qwen3 \
      --param_dir <ptq_params/mlp> --dataset_task livecodebench_release_v6 \
      --release_version release_v6 --max_gen_toks 2048 \
      --output_json /path/to/livecodebench_quant_custom_outputs.json \
      --raw_output_jsonl /path/to/livecodebench_quant_raw.jsonl`

   加 `--bf16` 跑基线。该脚本读取 `livecodebench/code_generation_lite` 的
   `question_content`、`starter_code` 等字段,生成 Python 代码并写出官方
   `custom_evaluator` JSON 格式:
   `[{ "question_id": "...", "code_list": ["..."] }]`。默认使用
   `livecodebench_release_v6` / `release_v6` 全量累计集;不要加 `--limit` 产出正式分数,
   因为官方 `custom_evaluator` 要求 custom 输出条数与所选 release 的 benchmark 条数一致。
   `--limit` 只用于生成链路冒烟。

   评分命令:
   `python -m lcb_runner.runner.custom_evaluator --scenario codegeneration \
      --release_version release_v6 \
      --custom_output_file /path/to/livecodebench_quant_custom_outputs.json`

   bf16 与量化各跑一次生成和官方评分,最后对比两个 pass@1。若需要与 LiveCodeBench 榜单
   严格对齐,还要确认 prompt 模板、release_version、样本全集、max_gen_toks、是否
   thinking 模式等设置与目标口径一致。

4. **报告结果**:对比量化模型与 bf16 的 exact_match / 官方 hidden-tests pass@1。报告时
   要写明样本量和 release_version。若使用 `--limit` 小样本冒烟,不能称为官方分数;差异
   落在统计噪声范围内应表述为"无可见损失",而不是"量化更好"。
   可用 `python scripts/compare_results.py --bf16 <bf16.log> --quant <quant.log> \
   --metric exact_match,none --sample_len <N> --limited` 汇总差值。

## 文件
- `scripts/config.py` — env 变量解析(AMCT_HOME/BENCH_ASSETS/HF_ENDPOINT)+
  加载 datasets.json;所有脚本共用。换机器改 env 变量,不改代码。提供
  `local_dataset_map()`;旧名 `local_parquet_map()` 保留为兼容别名。
- `scripts/utils.py` — 数据文件读取(parquet/jsonl)、依赖检查、LiveCodeBench 字段校验、
  attention 量化保护、结果日志解析等公共逻辑。
- `scripts/check_env.py` — 检查运行依赖与 lm-eval 版本;加 `--livecodebench_score`
  检查官方评分 runner。
- `scripts/self_check.py` — 不加载模型、不跑 NPU 的轻量自检,用于验证 skill 脚本接线。
- `scripts/datasets.json` — benchmark 注册表(repo_id、subdir、split→parquet/jsonl/glob)。
  prefetch + 离线 shim 的单一数据源。
- `scripts/prefetch_data.py` — snapshot_download 把数据集下到 $BENCH_ASSETS。
- `scripts/bench_offline.py` — monkeypatch load_dataset → 用 fastparquet 读本地 parquet,
  用 pandas 读本地 jsonl(映射来自 datasets.json)。
- `scripts/assemble_and_eval.py` — 装配 fake-quant 模型(KV-safe 目标、多卡保留 hook)、
  HFLM 评测、`--bf16` 基线。参数:--model/--model_name/--param_dir/--task/--limit/
  --device_map/--bit_config/--quant_target/--output_dir/--allow_attn_quant
  (默认按模型选 KV-safe 目标)。
- `scripts/livecodebench_eval.py` — 装配模型后生成 Python 代码,输出 LiveCodeBench 官方
  `custom_evaluator` JSON;hidden-tests pass@1 由官方 `lcb_runner` 计算。
- `scripts/compare_results.py` — 从 bf16/quant 日志或 JSON 中抽取指标并输出差值。

---
name: amct-llm-benchmark
description: |
  面向 AMCT 项目里 LLM 精度与性能基准测试的技能。用户要求做模型 PPL/Perplexity 评估、跑 MMLU 准确率、跑 MATH-500 数学推理评估、对比量化前后精度、跑 wikitext2/c4 等数据集、NPU 推理验证、下载 HF/ModelScope 模型做基准、对比 FP16 vs HiFloat8 W8A8、INT8、W4A16、W4A8、W4A4、FlatQuant、AWQ、GPTQ 等量化方案的精度损失，或验证量化算法在真实 LLM 上的端到端效果时，必须使用此 skill。涉及关键词：PPL、困惑度、benchmark、基准测试、量化对比、精度对比、wikitext、c4、MMLU、MATH-500、FlatQuant、W4A4、W4A8、fp8fp4、量化精度验证、HiFloat8 W8A8 验证、模型下载验证、推理 smoke test。
---

# AMCT LLM 基准测试

本 skill 用于在昇腾 NPU 上对 LLM 做端到端基准测试：模型获取 → 推理验证 → 数据集 PPL → 量化前后对比。整套流程已在 Qwen3-8B 上验证过（FP16 PPL 9.7153 vs HiFloat8 W8A8 9.7398，146 chunks），优先复用 `scripts/` 下的工具脚本而不是临时重写。

## 两条 HiFloat8 W8A8 路径——优先用 AMCT 主流程

| 路径 | 入口 | 实测 PPL | 何时用 |
|---|---|---|---|
| **AMCT 主流程** | `scripts/ppl_eval_amct.py` → `amct.quantize(m, HIFP8_CAST_CFG)` | **9.7398** | 跟主线版本对齐 / 想换其它 amct 算法（quantile/ofmr 等）只改 CFG |
| **教学版** | `scripts/ppl_eval.py --quant hif8_w8a8` → 自写 `Hifloat8FakequantLinearNPU` | **9.7398** | amct_pytorch 包导入坏的时候兜底 / 想加 INT8/W4A16/自定义量化扩展（只需写个 wrap 函数） |

两条路径**算法等价**（per-channel 对称权重 + per-tensor 激活，跳过 lm_head，NPU cast 算子相同），PPL 数值一致。差别只在工程入口和依赖：主流程依赖 `amct_pytorch` 包，教学版只用 `amct_ops`。速度都是 ~4.17s/chunk。

## ⚠️ 关于 amct_pytorch 源码 import

源码树的 `amct_pytorch/classic/graph_based/amct_pytorch/proto/` 只有 `.proto` 源文件，没有 protoc 生成的 `_pb2.py`。直接 import 会报：

```
ImportError: cannot import name 'calibration_config_pytorch_pb2' from
  partially initialized module '...proto' (most likely due to a circular import)
```

Python 的 "circular import" 是误报——真因是 `_pb2.py` 文件不存在。

**正规修复**（一次性）：

```bash
export PATH=/usr/local/protobuf-abi/bin:$PATH   # 让 cmake 找到 protoc
bash build.sh --pkg
cp build/lib/amct_pytorch/classic/graph_based/amct_pytorch/proto/*_pb2.py \
   amct_pytorch/classic/graph_based/amct_pytorch/proto/
```

跑完之后从仓库根目录 `import amct_pytorch` 就 OK 了。

**兜底**（没构建也想跑）：`ppl_eval_amct.py` 已经做了 try / fallback——如果源码 import 失败，自动切到已安装的 wheel `~/.local/lib/.../amct_pytorch/`，并打印一条提示。

教学版 `ppl_eval.py` 不依赖 amct_pytorch 包，从仓库根跑也可以。

## 工作流总览

```
[1] download_model.py       ModelScope 下载模型到本地
       ↓
[2] infer_smoke.py          NPU 推理 smoke test，确认权重+tokenizer+模板可用
       ↓
[3] ppl_eval.py             FP16 baseline PPL（WikiText-2，seqlen=2048）
       ↓
[4] ppl_eval.py --quant X   量化后 PPL，与 baseline 对比
       ↓
[5] 汇报                    PPL 差值、精度损失百分比、推理速度
```

每一步都是独立可重入的，调试期可以单跑某一步。

## 环境要求

- 昇腾 NPU（A2/A3/A5）+ `torch_npu`
- `amct_ops` 已通过 `bash amct_ops/ops_build.sh` 编译并 `pip install` —— `--quant hif8_w8a8` 直接调 `torch.ops.amct.encode/decode_to_hifloat8`，否则会 ImportError
- `pip install modelscope transformers datasets tqdm`
- WikiText-2 首次会从 HF Hub 下载（或读 `~/.cache/huggingface/datasets/wikitext` 缓存）

## 快速命令

```bash
SK=/mnt/workspace/gitCode/cann/amct/.claude/skills/amct-llm-benchmark/scripts
MODEL=/path/to/Qwen3-8B
cd /tmp  # 见上面"从哪儿跑"

# 1. 下载（如果模型还没本地化）
python $SK/download_model.py --model_id Qwen/Qwen3-8B --local_dir $MODEL

# 2. 推理验证
python $SK/infer_smoke.py --model_path $MODEL

# 3. 推荐：一次性跑 FP16 + HiFloat8 W8A8 (AMCT 主流程)
python $SK/ppl_eval_amct.py --model_path $MODEL                # both, 默认
python $SK/ppl_eval_amct.py --model_path $MODEL --quant fp16   # 只跑 FP16
python $SK/ppl_eval_amct.py --model_path $MODEL --quant hif8   # 只跑 HIFP8 cast

# 4. 教学版（最简实现，算法与主流程等价）
PYTHONPATH=$SK python $SK/ppl_eval.py --model_path $MODEL                       # FP16
PYTHONPATH=$SK python $SK/ppl_eval.py --model_path $MODEL --quant hif8_w8a8     # W8A8

# 探速：先 4 个 chunk 估时
PYTHONPATH=$SK python $SK/ppl_eval.py --model_path $MODEL --quant hif8_w8a8 --max_chunks 4
```

`PYTHONPATH=$SK` 仅教学版需要（脚本里 `from hif8_w8a8_wrap import ...`）。主流程 `ppl_eval_amct.py` 不需要。

## ⚠️ Subagent 上下文 / Bash 600s timeout

HiFloat8 W8A8 PPL 全量 ~10 min（4.17s/chunk × 146）。Subagent 的 Bash tool **单次最大 600s**，会被截断。两个应对：

**方案 A：分两段跑 + merge**（最常用，仅 2 次 bash 调用）

`--quant both` 模式会复用一次模型加载：FP16 全跑（~30s）+ in-place 量化（~10s）+ HiFloat8 part 1，全部在第一次 bash 调用里。

```bash
# Phase 1（FP16 baseline + HiFloat8 前半，~7 min）
python $SK/ppl_eval_amct.py --model_path $MODEL --quant both \
    --start_chunk 0 --max_chunks 73 --nll_dump /tmp/p1.json
# 输出：FP16 全量 PPL（直接报）+ HiFloat8 chunks 0-72 (--quant both 时 start_chunk/max_chunks 仅作用于 HiFloat8 部分)

# Phase 2（HiFloat8 后半，~7 min）
python $SK/ppl_eval_amct.py --model_path $MODEL --quant hif8 \
    --start_chunk 73 --max_chunks 73 --nll_dump /tmp/p2.json

# 合并出 HiFloat8 全量 PPL
python $SK/merge_ppl.py /tmp/p1.hif8.json /tmp/p2.hif8.json
```

省下一次 ~75s 的模型加载（iter-3 优化）。如果想 3 次调用更保守，用三次 `--quant fp16` / `--quant hif8 --start_chunk N --max_chunks M`。

**方案 B：从主会话跑（非 subagent）**
主会话的 `run_in_background: true` 是 harness 跟踪的真后台，subagent 退出不会杀进程。如果你是 subagent，把这段 NPU 工作回报给上层让它跑，比自己硬扛 timeout 强。

**⚠️ DO NOT** 在 subagent 里用 `Bash(run_in_background=True)` 跑长 PPL — 任务跟着 subagent 生命周期，subagent 退出时进程会被收掉。iteration-1 实测发现的真实 footgun。

## 各阶段要点

### [1] 模型下载（`download_model.py`）

- 默认目标 ModelScope；HF Hub 用 `huggingface_hub.snapshot_download` 类似封装也可
- 大模型建议加 `--include "*.safetensors" "*.json" "*.txt" "tokenizer*"` 跳过老的 `.bin` 减少下载量
- ModelScope CLI `modelscope download --model X --local_dir Y` 与脚本等价；脚本只是为了让 skill 流程统一

**Qwen3 系列实际尺寸**（用户经常把 Qwen1.5/Qwen2 的 1.8B/1.5B 误套用到 Qwen3）：

| 系列 | 可用尺寸 | 示例 model id |
|---|---|---|
| Qwen3 | 0.6B / 1.7B / 4B / 8B / 14B / 32B | `Qwen/Qwen3-1.7B` |
| Qwen2 / Qwen2.5 | 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B | `Qwen/Qwen2.5-7B` |
| Qwen1.5 | 0.5B / 1.8B / 4B / 7B / 14B / 32B / 72B / 110B | `Qwen/Qwen1.5-1.8B` |

**没有** `Qwen/Qwen3-1.8B` —— 用户报错 404 时，先猜他想的是 Qwen3-1.7B 还是 Qwen1.5-1.8B，确认再下载。

### [2] 推理验证（`infer_smoke.py`）

- 必跑：确认模型加载没坏、tokenizer 模板正确、NPU 推理 pipeline 通畅
- 默认两条 prompt（中英各一），生成 128 tokens；输出文本 + token 速率
- Qwen3 系列默认走 `apply_chat_template(enable_thinking=False)` 关闭思考模式，否则会输出 `<think>...</think>` 噪声；非 Qwen3 模型用 `TypeError` 兜底回退
- 加载 8B 模型约 60-90s，推理速度参考：8B FP16/NPU ≈ 6-8 tok/s（朴素生成，未启 vLLM）

### [3] PPL 评估（`ppl_eval.py`）

参考实现：`amct_pytorch/experimental/hifloat8/utils.py:test_ppl`，本 skill 脚本沿用相同协议：

| 参数 | 默认值 | 来源 |
|---|---|---|
| 数据集 | `wikitext` / `wikitext-2-raw-v1` / `test` | GPTQ/AWQ 论文标准 |
| seqlen | 2048 | 同上 |
| 切片方式 | 拼接全部 text 后非重叠分块 | 同上 |
| 损失 | `CrossEntropyLoss`，按 `seqlen` 累加 NLL | 同上 |
| PPL | `exp(sum_nll / (nsamples * seqlen))` | 同上 |

总切片数取决于 tokenizer，Qwen3 = 146；LLaMA = 290 左右；GPT-2 ≈ 287。

**陷阱**：
- 用 `tokenizer.apply_chat_template` 跑 PPL 是错的 —— PPL 评的是 base LM 的下一 token 预测，不应附加 chat 包装。脚本不走 chat template
- 别开 `--max_chunks` 跑正式数字 —— 前 N 个 chunk 不代表全集，Qwen3-8B FP16 前 4 chunk PPL=10.5 而全 146 chunk=9.72，差异巨大
- 必须确保 `eval()` 模式（脚本已设），否则 dropout 会让 PPL 跳动

### [4] 量化对比

调用约定：所有量化包装器都接收 `model`、原地替换 Linear、返回替换数量。

```python
def wrap_model_xxx(model: nn.Module, prefix: str = "") -> int:
    ...
```

`scripts/ppl_eval.py:get_quant_wrapper` 注册新量化方式只要加一个分支：

```python
if name == "int8_w8a8":
    from int8_w8a8_wrap import wrap_model_int8_w8a8
    return wrap_model_int8_w8a8
```

内置的 `hif8_w8a8`（见 `scripts/hif8_w8a8_wrap.py`）是 Cast 算法（最简单的 HiFloat8 W8A8）：
- 权重 per-output-channel 对称，`scale_w[i] = |w[i,:]|.amax() / 16`，每行归一到 HiFloat8 高精度区上限
- 激活直转（无 scale）
- 跳过 `lm_head` + embedding，只替换 `model.layers.*` 下的 Linear
- 用 `amct_ops.hifloat8_cast` 在 NPU 上跑 cast，比 `amct_pytorch/experimental/hifloat8/` 的 CPU OpenMP 版快 ~3.5x

其它变体（TAQ / ORQ / Quantile / INT8 / W4A16 / AWQ / GPTQ）的扩展模板见 `references/quant_variants.md`。

## 精度评估：PPL 之外（MMLU / MATH-500）

PPL 衡量的是语言建模能力，但量化对**下游任务准确率**的影响往往更值得关心。`scripts/` 里已经备好两个准确率评估脚本，机制和数据集都按业界标准实现，优先复用、不要重写：

### MMLU 5-shot（`mmlu_eval.py`）

判别式评估：拼 5 个 dev 样例做 few-shot，对测试题取最后一个位置在 ` A`/` B`/` C`/` D` 四个 token 上的 logits，argmax 即预测。单次前向、无生成，所以很快。

```bash
SK=/mnt/workspace/gitCode/cann/amct/.claude/skills/amct-llm-benchmark/scripts
python $SK/mmlu_eval.py --model_path $MODEL                      # FP16 baseline
python $SK/mmlu_eval.py --model_path $MODEL --quant both         # FP16 + HiFloat8 W8A8 一次加载
python $SK/mmlu_eval.py --model_path $MODEL --subjects abstract_algebra,anatomy  # 抽几科探速
```

- `--quant` 取值：`none`（默认，仅 FP16）/ `hif8_w8a8`（仅量化）/ `both`（两者一次加载，省一次 reload）
- `--mmlu_root` 默认 `/mnt/workspace/gitCode/cann/datasets/mmlu`，读 `all/dev-*.parquet` 和 `all/test-*.parquet`
- `--subjects` 逗号分隔指定科目，调试时只跑几科

### MATH-500（`math500_eval.py`）

生成式评估：用官方模板让模型解题、要求把答案写进 `\boxed{}`，再用 `math_verify`（基于 sympy 的等价判定，lm-eval-harness 标准）打分。Qwen3 走 `enable_thinking=True` 思考模式，采样参数用官方推荐值（temp=0.6/top_p=0.95/top_k=20），`max_new_tokens` 默认 8192（思考链很长）。

```bash
python $SK/math500_eval.py --model_path $MODEL --max_problems 5            # 5 题 smoke test
python $SK/math500_eval.py --model_path $MODEL --quant both                # 全 500 题，FP16 + W8A8
python $SK/math500_eval.py --model_path $MODEL --stratified_sample 50      # 按难度分层抽 50 题
```

- `--data_path` 默认 `/mnt/workspace/gitCode/cann/datasets/math500/test.jsonl`（500 题，7 类 5 难度）
- 依赖 `math_verify`（`pip install math_verify`）；缺了会在评分处 import 失败
- 生成式 + 长 CoT，比 MMLU/PPL 慢得多，全量 500 题要按 600s timeout 切分或放主会话后台跑（见上面 ⚠️ 约束）

## 低比特量化模式：W4A8 与 FlatQuant W4A4

HiFloat8 W8A8 之外，`scripts/` 还支持更激进的低比特方案。它们对精度影响更大，PPL 是首选验证手段。

> `amct.quantize → 校准 → amct.convert` 这套量化接口的通用调用配方（免校准 vs 需校准算法的区别、校准数据怎么组织）以 **amct-pytorch-quant-algo-dev skill 的「直接使用」章节为权威来源**。本节只描述各脚本在此之上的具体模式差异。

### fp8fp4 / W4A8（`ppl_eval_amct.py --quant fp8fp4`）

复用主流程脚本的第四种模式：激活 `float8_e4m3fn` × 权重 `float4_e2m1`，group=32，minmax 算法。和 `hif8`（免校准的 Cast 路径）不同，**fp8fp4 需要 pileval 校准**——脚本会自动 `amct.quantize → 用 pileval 数据集校准 → 评 PPL`。

```bash
python $SK/ppl_eval_amct.py --model_path $MODEL --quant fp8fp4                  # bf16 baseline + W4A8
python $SK/ppl_eval_amct.py --model_path $MODEL --quant fp8fp4 --calib_samples 64  # 调校准样本数（默认 64）
```

注意基线是 **bf16**（不是 fp16），因为 W4A8 路径按 bf16 加载模型。

### FlatQuant W4A4 / W4A8（`ppl_eval_flatquant.py`）

FlatQuant 通过可学习的仿射变换矩阵"压平"激活分布，再做低比特量化，是目前 W4A4 精度最好的方案之一。脚本依赖 `amct_pytorch.experimental.flatquant`，需要先训练变换矩阵（`cali_flat_quant`），再评 PPL。

```bash
python $SK/ppl_eval_flatquant.py --model_path $MODEL --mode w4a4              # int4 权重 × int4 激活，amct.convert → NPU
python $SK/ppl_eval_flatquant.py --model_path $MODEL --mode w4a8              # int4 权重 × int8 激活，仅 fake-quant
python $SK/ppl_eval_flatquant.py --model_path $MODEL --mode w4a4 --save_matrix /tmp/flat.pt  # 存变换矩阵复用
python $SK/ppl_eval_flatquant.py --model_path $MODEL --mode w4a4 --load_matrix /tmp/flat.pt  # 复用已训练矩阵跳过训练
```

两个模式的本质差异（脚本 docstring 写明）：
- `w4a4`：int4 权重（channel）× int4 激活（token），调 `amct.convert` 部署到 NPU 的 `dynamic_w4a4` 算子，是真正的端到端部署路径
- `w4a8`：int4 权重 × int8 激活，**仅 fake-quant**。因为 amct 的 flatquant NPU 部署算子只支持 W4A4，`int8×int4` 组合也不在 amct 默认白名单里，脚本在内存里临时 monkey-patch 白名单（`vars.py ALLOWED_DTYPE_MAP`），不改 wheel，跳过 `convert`，在 convert 前的 fake-quant 模型上评 PPL

关键参数：`--epochs`（变换矩阵训练轮数，默认 15）、`--nsamples`（校准样本，默认 128）、`--cali_bsz`（校准 batch，默认 4）、`--flat_lr`（学习率，默认 3e-3）。训练矩阵开销不小，探速时配 `--max_chunks` 和 `--skip_baseline`。

## 输出报告格式

跑完一组对比后，固定按这个表格汇报：

```
| 配置 | PPL | Δ vs FP16 | 推理耗时 | s/chunk |
|---|---|---|---|---|
| FP16 baseline | X.XXXX | — | XXs | X.XX |
| 量化方案 A | X.XXXX | +X.XXXX (+X.XX%) | XXs | X.XX |
```

附加信息：
- 替换了多少个 Linear、跳过了哪些层
- 量化策略的关键参数（scale 公式、是否需要校准）
- 速度比较 baseline / 量化 / 是否引入额外 cast 开销

如果差异很小（PPL Δ < 0.05），明确说"在测量噪声范围内"，不要过度解读。

## 性能与调试参考

| 指标 | Qwen3-8B 实测（NPU 单卡 FP16） |
|---|---|
| 模型加载 | ~80s |
| 量化 wrap | hif8_w8a8 NPU: 10s；CPU 版: 37s |
| PPL eval 速度（FP16） | 0.19s/chunk，全量 ~28s |
| PPL eval 速度（hif8 W8A8, NPU op） | 4.17s/chunk，全量 ~10 min |
| PPL eval 速度（hif8 W8A8, CPU op） | 15.3s/chunk，全量 ~37 min |
| 单 prompt 生成（128 tok） | ~17s，~7 tok/s |

如果实测明显偏离这个范围，先查：
1. 模型 dtype 是否真的是 fp16（`print(model.dtype)`）
2. `--device` 是不是 npu 而不是 cpu
3. amct_ops 是否真的走了 NPU 算子（CPU 路径会回退到 OpenMP）
4. NPU 是否被其它进程占用（`npu-smi info`）

## 常见陷阱

- **PPL 公式按 seqlen 加权**：`neg_log_likelihood = loss * seqlen` 不能省，否则平均后 PPL 偏小一个量级
- **`shift_labels` 切片**：`testenc[:, i*seqlen:(i+1)*seqlen][:, 1:]` 必须从 `testenc` 切而不是 `batch`（device 不同），否则 cross-entropy 报错
- **`apply_chat_template` 的 `enable_thinking`**：仅 Qwen3 支持，其它模型用 `TypeError` 捕获回退到不带这个参数
- **AMCT 包导入失败**：`from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule` 可能因 protobuf .py 缺失而循环导入挂掉；skill 脚本里特意没依赖 `BaseQuantizeModule`，直接继承 `nn.Module`，绕开这个问题
- **NPU 内存不足**：Qwen3-8B FP16 ~16GB 显存，量化后加 cast buffer 还会更高；如果 OOM，先 `torch.npu.empty_cache()` 然后看 `npu-smi info` 是否有遗留进程
- **wikitext 缓存路径**：HF datasets 首次拉取走 hub，离线环境要预下载到 `~/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/`

## 给 AI 助手的工作流提示

被触发后，先和用户确认四个边界：

1. **模型来源**：已经在本地（给路径）还是要下载（给 model_id）？如果是 Qwen3-1.8B 这种不存在的 id，先猜想用户实际想要哪个（见上面 Qwen 系列尺寸表）
2. **量化方案**：要对比哪些（HiFloat8 W8A8 / INT8 / W4A16 / AWQ / GPTQ / 不量化只跑 baseline）？默认走 AMCT 主流程 `ppl_eval_amct.py`
3. **数据集**：默认 wikitext-2-raw-v1，需要换 c4/ptb 也支持
4. **样本量**：默认全量 146 chunks（Qwen3-8B 约 10 min/方案），探速可以 `--max_chunks 4`

确认后按工作流顺序跑：

- **在 subagent 上下文**：如果 PPL 单次 > 600s，用 `--start_chunk + --max_chunks + --nll_dump` 分两段，最后 `merge_ppl.py` 合并。**禁止** `run_in_background:true` —— subagent 退出会杀进程
- **在主会话上下文**：直接 `run_in_background:true`，harness 跟踪长任务

不要自己写 PPL 公式 —— `scripts/ppl_core.py:eval_ppl` 已经按 AMCT 标准实现并和 `amct_pytorch/experimental/hifloat8/utils.py` 一致，且是**唯一来源**：ppl_eval.py / ppl_eval_amct.py / ppl_eval_flatquant.py 全部 `from ppl_core import eval_ppl`，不要在某个脚本里复制一份再改。要改公式（如 sliding window、stride、token-level masking）先解释为什么标准公式不够用，再改 `ppl_core.py:eval_ppl` 一处，三个脚本自动同步。

不要把"前 4 chunk 的 PPL"当作结论汇报 —— 那只是探速值。正式数字必须跑完整 chunks。前 4 chunk vs 全 146 chunks 在 Qwen3-8B 上能差 10% PPL。

不要在 `lm_head` 上做 W8A8 量化 —— 会显著拉高 PPL，且不是常见做法。两条路径都已经过滤（教学版 `wrap_model_hif8_w8a8` 按名字过滤；主流程 `HIFP8_CAST_CFG` 用 `skip_layers={'lm_head'}`）。

不要从仓库根目录跑 `ppl_eval_amct.py` —— 见上面 ⚠️ 约束。脚本已自动 `cd /tmp`，但**别在 shell 里 `cd amct/` 后 `python script.py`**。
